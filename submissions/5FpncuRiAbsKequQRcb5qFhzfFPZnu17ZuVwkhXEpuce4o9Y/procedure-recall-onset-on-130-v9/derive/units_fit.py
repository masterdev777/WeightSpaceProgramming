"""Fit rewritten MLP units (set_rows on gate_proj / up_proj, set_cols on down_proj) on rows of the model's own text, with
the model frozen (teacher forcing; "keep" rows: KL to the unedited model). The units are the ones with the largest
summed loss gradient over a sample of the rows (--units N), never one that the parent chain already rewrote
(--exclude-units PARENT_EDIT.json, the whole chain is read), never in the last layer (--skip-last-mlp).
    python units_fit.py train MODEL_DIR ROWS.train.jsonl OUT_PREFIX --heldout ROWS.held.jsonl --rank 0 --units 128 \
        --skip-last-mlp --exclude-units PARENT_EDIT.json --lr 1e-4 --accum 8 --steps S --warmup 2 --max-tokens 1536
    python units_fit.py export OUT_PREFIX CASE NAME --model MODEL_DIR     # -> $WSP_CASES/CASE/edits/NAME.json + .npz
(The same fitter also trains per-unit / per-head gains with --unit-gains / --gains; this method does not use them.)"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

WRITERS = {"o": "self_attn.o_proj", "g": "linear_attn.out_proj", "d": "mlp.down_proj"}
HOOKED = {"o": "self_attn.o_proj", "g": "linear_attn.out_proj", "d": "mlp"}
EOT = "<|im_end|>"


def writer_modules(layers, kinds, skip_last_mlp):
    out = []
    for li, layer in enumerate(layers):
        for k in kinds:
            mod = layer
            try:
                for part in HOOKED[k].split("."):
                    mod = getattr(mod, part)
            except AttributeError:
                continue
            if k == "d" and skip_last_mlp and li == len(layers) - 1:
                continue
            out.append((li, k, mod))
    return out


def train(a):
    import torch
    from wsp.probe import decoder, load_model

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    DEV = os.environ.get("PT_DEVICE", "cuda")
    model, tok = load_model(a.model, DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    if a.ckpt:
        model.gradient_checkpointing_enable()
    owner, layers = decoder(model)
    d = model.config.get_text_config().hidden_size
    mods = writer_modules(layers, a.kinds, a.skip_last_mlp) if a.rank > 0 else []
    P = torch.nn.ParameterDict()
    hooks, ON = [], [True]
    for li, k, mod in mods:
        key = f"{k}{li}"
        P[key + "_B"] = torch.nn.Parameter(torch.zeros(d, a.rank, device=DEV))
        P[key + "_C"] = torch.nn.Parameter(torch.randn(d, a.rank, device=DEV) / math.sqrt(d))

        def hook(m, i, y, key=key):
            if not ON[0]:
                return y
            B, C = P[key + "_B"].to(y.dtype), P[key + "_C"].to(y.dtype)
            return y - 0.5 * ((y @ C) @ B.T + (y @ B) @ C.T)

        hooks.append(mod.register_forward_hook(hook))
    print(f"{len(mods)} writers, rank {a.rank}: {sum(p.numel() for p in P.values())} trainable values", flush=True)
    if a.gains:  # one log-gain per attention head (full attention and Gated DeltaNet), applied to the head's slice of the mixer output
        cfg = model.config.get_text_config()
        for li, layer in enumerate(layers):
            if not a.gain_layers[0] <= li <= a.gain_layers[1]:
                continue
            for k, nh in (("o", cfg.num_attention_heads), ("g", cfg.linear_num_value_heads)):
                mod = layer
                try:
                    for part in WRITERS[k].split("."):
                        mod = getattr(mod, part)
                except AttributeError:
                    continue
                key = f"s{k}{li}"
                P[key] = torch.nn.Parameter(torch.zeros(nh, device=DEV))

                def pre(m, args, key=key, nh=nh):
                    if not ON[0]:
                        return None
                    x = args[0]
                    return (x * P[key].exp().to(x.dtype).repeat_interleave(x.shape[-1] // nh),)

                mod.register_forward_pre_hook(pre)
        print(f"with head gains: {sum(p.numel() for p in P.values())} trainable values", flush=True)
    if a.unit_gains:  # one log-gain per MLP unit (exported as scale_rows on up_proj), applied to the unit's activation
        lo, hi = a.gain_layers
        for li, layer in enumerate(layers):
            if not lo <= li <= hi:
                continue
            key = f"m{li}"
            P[key] = torch.nn.Parameter(torch.zeros(layer.mlp.down_proj.weight.shape[1], device=DEV))

            def upre(m, args, key=key):
                if not ON[0]:
                    return None
                return (args[0] * P[key].exp().to(args[0].dtype),)

            layer.mlp.down_proj.register_forward_pre_hook(upre)
        print(f"with unit gains (layers {lo}-{hi}): {sum(p.numel() for p in P.values())} trainable values", flush=True)

    cap = {}
    owner.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    head = model.get_output_embeddings()

    def nll(pi, ti, backward=None):
        """Mean negative log-likelihood of the target tokens. With backward=scale, the gradient of scale * nll is
        accumulated (the output head is applied in chunks, so the full-vocabulary logits never exist at once)."""
        ids = torch.tensor([pi + ti], device=DEV)
        model(input_ids=ids, logits_to_keep=1, use_cache=False)
        h = cap.pop("h")[0]
        hd = h.detach().requires_grad_(backward is not None)
        tgt, n0, n = ids[0], len(pi), len(ti)
        total, g = 0.0, torch.zeros_like(hd) if backward is not None else None
        for c in range(n0 - 1, n0 - 1 + n, 512):
            e = min(c + 512, n0 - 1 + n)
            l = torch.nn.functional.cross_entropy(head(hd[c:e]).float(), tgt[c + 1:e + 1], reduction="sum")
            if backward is not None:
                g += torch.autograd.grad(l, hd)[0]
            total += float(l)
        if backward is not None:
            h.backward(g * (backward / n))
        return total / n

    def distill(pi, ti, backward=None, same=False):
        """KL(teacher sharpened || patched) per token; teacher = the unpatched model, its logits divided by tau on the
        first a.distill_until target tokens (all when 0) and unchanged after."""
        ids = torch.tensor([pi + ti], device=DEV)
        ON[0] = False
        with torch.no_grad():
            model(input_ids=ids, logits_to_keep=1, use_cache=False)
        ht = cap.pop("h")[0]
        ON[0] = True
        model(input_ids=ids, logits_to_keep=1, use_cache=False)
        h = cap.pop("h")[0]
        hd = h.detach().requires_grad_(backward is not None)
        n0, n = len(pi), len(ti)
        pos = torch.arange(n0 - 1, n0 - 1 + n, device=DEV)
        if a.distill_positions and n > a.distill_positions:  # all sharpened positions + a random sample of the rest
            cut = a.distill_until or n
            keep = torch.cat([pos[:cut], pos[cut:][torch.randperm(max(n - cut, 0), device=DEV)[: max(a.distill_positions - cut, 64)]]])
            pos = keep.sort().values
        total, g = 0.0, torch.zeros_like(hd) if backward is not None else None
        for c in range(0, len(pos), 384):
            p = pos[c:c + 384]
            tau = torch.ones(len(p), 1, device=DEV) if same else torch.where((p - (n0 - 1)) < (a.distill_until or n), a.distill_tau, 1.0)[:, None]
            with torch.no_grad():
                t = torch.log_softmax(head(ht[p]).float() / tau, -1)
            q = torch.log_softmax(head(hd[p]).float(), -1)
            l = (t.exp() * (t - q)).sum()
            if backward is not None:
                g += torch.autograd.grad(l, hd)[0]
            total += float(l)
        if backward is not None:
            h.backward(g * (backward / len(pos)))
        return total / len(pos)

    def encode(row):
        pi = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        ti = tok(row["target"] + ("" if row.get("no_eot") else EOT), add_special_tokens=False)["input_ids"][: a.max_tokens]  # no_eot: mid-thinking targets
        return pi, ti, float(row.get("w", 1.0)), row.get("mode", "fit")

    def loss(pi, ti, mode, backward=None):
        """mode "keep": stay with the unpatched model on this answer (KL at temperature 1); else the main objective."""
        if mode == "keep":
            return distill(pi, ti, backward=backward, same=True)
        return (distill if a.distill_tau else nll)(pi, ti, backward=backward)

    rows = [json.loads(x) for x in open(a.train)]
    data = [encode(r) for r in rows]
    held = [encode(json.loads(x)) for x in open(a.heldout)] if a.heldout else []
    print(f"{len(data)} training answers ({sum(len(t) for _, t, _, _ in data) / 1e6:.2f}M target tokens, {sum(m == 'keep' for *_, m in data)} keep rows), {len(held)} held out", flush=True)

    IDX = {}
    if a.units:  # MLP units with the largest summed loss gradient over a sample of the training answers
        ws = [w for layer in layers for w in (layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)]
        for w in ws:
            w.requires_grad_(True)
        for pi, ti, _, m in random.sample(data, min(a.sal_samples, len(data))):
            loss(pi, ti, m, backward=1.0)
        score = torch.stack([layer.mlp.gate_proj.weight.grad.float().pow(2).sum(1) + layer.mlp.up_proj.weight.grad.float().pow(2).sum(1)
                             + layer.mlp.down_proj.weight.grad.float().pow(2).sum(0) for layer in layers])  # [L, units]
        for w in ws:
            w.requires_grad_(False)
            w.grad = None
        if a.skip_last_mlp:
            score[-1] = 0
        if a.exclude_units:  # units the parent chain already rewrote (set_rows on gate/up, set_cols on down): never chosen again
            import json as _j, numpy as _np
            from pathlib import Path as _P
            sp, nex = _P(a.exclude_units).resolve(), 0
            while sp and sp.exists():
                spec = _j.load(open(sp)); z = _np.load(sp.with_name("edit.npz")) if sp.with_name("edit.npz").exists() else None
                for o in spec["ops"]:
                    m = re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", o["tensor"])
                    # a unit is rewritten by set_rows / dir_ablate side "in" restricted to rows (gate/up) or by
                    # set_cols / dir_ablate side "out" restricted to cols (down_proj; the parent chain writes some of its units this way)
                    is_row = o["op"] == "set_rows" or (o["op"] == "dir_ablate" and o.get("side", "out") == "in")
                    sel = o.get("rows") if is_row else o.get("cols") if o["op"] in ("set_cols", "dir_ablate") else None
                    if not m or sel is None or is_row == (m.group(2) == "down_proj"):
                        continue
                    if isinstance(sel, str) and "#" in sel:
                        sel = z[sel.split("#")[1]]
                    sel = _np.asarray(sel).ravel()
                    if sel.dtype == bool:
                        sel = _np.nonzero(sel)[0]
                    idx = torch.as_tensor(sel.astype("int64"))
                    nex += int((score[int(m.group(1)), idx] > 0).sum()); score[int(m.group(1)), idx] = 0
                par = spec.get("parent")   # a spec path ("parent/edit.json") or a published edit's name (<name>/edit/edit.json)
                sp = ((sp.parent / par) if par.endswith(".json") else (sp.parent / par / "edit" / "edit.json")).resolve() \
                    if par and (par.endswith(".json") or re.fullmatch(r"[A-Za-z0-9_-]+", par)) else None
            print(f"excluded units of the parent chain: {nex} scores zeroed", flush=True)
        top = torch.topk(score.flatten(), a.units).indices
        for li in range(len(layers)):
            sel = (top[top // score.shape[1] == li] % score.shape[1]).sort().values
            if len(sel):
                IDX[li] = sel
        print("units per layer:", {li: len(v) for li, v in IDX.items()}, flush=True)
        for li, sel in IDX.items():
            mlp, key = layers[li].mlp, f"u{li}"
            P[key + "_G"] = torch.nn.Parameter(torch.zeros(len(sel), d, device=DEV))
            P[key + "_U"] = torch.nn.Parameter(torch.zeros(len(sel), d, device=DEV))
            P[key + "_D"] = torch.nn.Parameter(torch.zeros(d, len(sel), device=DEV))

            def fwd(x, mlp=mlp, key=key, sel=sel):
                g, u = mlp.gate_proj(x), mlp.up_proj(x)
                h = mlp.act_fn(g) * u
                if not ON[0]:
                    return mlp.down_proj(h)
                hs = mlp.act_fn(g[..., sel] + x @ P[key + "_G"].to(x.dtype).T) * (u[..., sel] + x @ P[key + "_U"].to(x.dtype).T)
                return mlp.down_proj(h) + (hs - h[..., sel]) @ mlp.down_proj.weight[:, sel].T + hs @ P[key + "_D"].to(x.dtype).T

            mlp.forward = fwd
        torch.save({li: v.cpu() for li, v in IDX.items()}, a.out + ".units.pt")
        print(f"with units: {sum(p.numel() for p in P.values())} trainable values", flush=True)

    def evaluate():
        if not held:
            return float("nan")
        with torch.no_grad():
            return float(np.mean([w * loss(pi, ti, m) for pi, ti, w, m in held if m != "keep"]))

    opt = torch.optim.AdamW(P.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    log = {"args": vars(a), "eval": [], "train": []}
    print(f"step 0 heldout nll {evaluate():.4f}", flush=True)
    order, t0, ema = [], time.time(), None
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            if not order:
                order = random.sample(range(len(data)), len(data))
            pi, ti, w, m = data[order.pop()]
            tot += loss(pi, ti, m, backward=w / a.accum) * w / a.accum
        torch.nn.utils.clip_grad_norm_(P.parameters(), 1.0)
        opt.step()
        sched.step()
        ema = tot if ema is None else 0.95 * ema + 0.05 * tot
        if (step + 1) % a.log_every == 0:
            log["train"].append([step + 1, ema])
            print(f"step {step + 1} train nll (ema) {ema:.4f}  {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            e = evaluate()
            log["eval"].append([step + 1, e])
            print(f"step {step + 1} heldout nll {e:.4f}", flush=True)
            torch.save({k: v.detach().cpu() for k, v in P.items()}, f"{a.out}.step{step + 1}.pt")
    torch.save({k: v.detach().cpu() for k, v in P.items()}, a.out + ".pt")
    json.dump(log, open(a.out + ".log.json", "w"))
    print("TRAIN_DONE", flush=True)


def export(a):
    import torch

    P = torch.load(a.out + (f".step{a.step}" if a.step else "") + ".pt")
    if a.avg:  # average several fits (same keys) in parameter space = log space for gains
        Ps = [P] + [torch.load(x + ".pt") for x in a.avg]
        P = {k: sum(q[k].double() for q in Ps) / len(Ps) for k in P if all(k in q for q in Ps)}
        print(f"averaged {len(Ps)} fits")
    from wsp import case as WC
    root = WC.cdir(a.case) / "edits"
    root.mkdir(parents=True, exist_ok=True)   # a fresh case has no edits/ yet
    arrays, ops = {}, []
    if a.stack:  # the ops of an existing spec of the case come first (same parent)
        base = json.load(open(root / f"{a.stack}.json"))
        z = np.load(root / f"{a.stack}.npz")
        for op in base["ops"]:
            op = dict(op)
            for k in ("u", "cols", "rows", "values"):
                if isinstance(op.get(k), str):
                    key = op[k].split("#")[1]
                    arrays["s_" + key] = z[key]
                    op[k] = f"{a.name}.npz#s_{key}"
            ops.append(op)
    for key in sorted(k for k in P if k[0] == "s"):
        kind, li = key[1], int(key[2:])
        g = (P[key].double() * a.scale).exp()
        shape1 = 2048
        hd = shape1 // len(g)
        for h in range(len(g)):
            if abs(float(g[h]) - 1) < a.min_gain:
                continue
            arrays[f"{key}_{h}"] = np.arange(h * hd, (h + 1) * hd, dtype=np.int16)
            ops.append({"op": "scale_cols", "tensor": f"model.language_model.layers.{li}.{WRITERS[kind]}.weight",
                        "cols": f"{a.name}.npz#{key}_{h}", "factor": round(float(g[h]), 5)})
    ng = sum(1 for o in ops if o["op"] == "scale_cols")
    if ng:
        gs = [o["factor"] for o in ops if o["op"] == "scale_cols"]
        print(f"{ng} head gains, range {min(gs):.3f} .. {max(gs):.3f}")
    nu = 0
    for key in sorted((k for k in P if k[0] == "m"), key=lambda k: int(k[1:])):
        li, lg = int(key[1:]), P[key].double().numpy() * a.scale
        edges = np.linspace(lg.min(), lg.max() + 1e-12, a.levels + 1)
        lev = np.clip(np.digitize(lg, edges) - 1, 0, a.levels - 1)
        for j in range(a.levels):
            rows = np.nonzero(lev == j)[0]
            if not len(rows):
                continue
            f = float(np.exp(lg[rows].mean()))
            if abs(f - 1) < a.min_gain:
                continue
            arrays[f"{key}_{j}"] = rows.astype(np.int64)
            ops.append({"op": "scale_rows", "tensor": f"model.language_model.layers.{li}.mlp.up_proj.weight",
                        "rows": f"{a.name}.npz#{key}_{j}", "factor": round(f, 5)})
            nu += len(rows)
    if nu:
        fs = [o["factor"] for o in ops if o["op"] == "scale_rows"]
        print(f"{nu} unit gains in {len(fs)} scale_rows ops ({a.levels} levels/layer), range {min(fs):.3f} .. {max(fs):.3f}")
    ukeys = sorted({k[:-2] for k in P if k[0] == "u"}, key=lambda k: int(k[1:]))
    if ukeys:
        from safetensors import safe_open
        IDX = torch.load(a.out + ".units.pt")
        import glob
        W = {}
        for f in glob.glob(str(Path(a.model) / "*.safetensors")):
            with safe_open(f, "pt") as sf:
                for n in sf.keys():
                    if ".mlp." in n and "language_model.layers" in n:
                        W[n] = sf
                        W[n] = sf.get_tensor(n).float()
        for key in ukeys:
            li = int(key[1:])
            sel = IDX[li]
            t = f"model.language_model.layers.{li}.mlp."
            arrays[f"{key}_rows"] = sel.numpy().astype(np.int64)
            arrays[f"{key}_G"] = (W[t + "gate_proj.weight"][sel] + P[key + "_G"].float()).numpy().astype(np.float16)
            arrays[f"{key}_U"] = (W[t + "up_proj.weight"][sel] + P[key + "_U"].float()).numpy().astype(np.float16)
            arrays[f"{key}_D"] = (W[t + "down_proj.weight"][:, sel] + P[key + "_D"].float()).numpy().astype(np.float16)
            ops += [{"op": "set_rows", "tensor": t + "gate_proj.weight", "rows": f"{a.name}.npz#{key}_rows", "values": f"{a.name}.npz#{key}_G"},
                    {"op": "set_rows", "tensor": t + "up_proj.weight", "rows": f"{a.name}.npz#{key}_rows", "values": f"{a.name}.npz#{key}_U"},
                    {"op": "set_cols", "tensor": t + "down_proj.weight", "cols": f"{a.name}.npz#{key}_rows", "values": f"{a.name}.npz#{key}_D"}]
        print(f"{sum(len(IDX[int(k[1:])]) for k in ukeys)} rewritten units in {len(ukeys)} layers")
    keys = sorted({k[:-2] for k in P if k.endswith("_B")})
    n_dirs, alphas = 0, []
    for key in keys:
        B, C = P[key + "_B"].double(), P[key + "_C"].double()
        Q, R = torch.linalg.qr(torch.cat([B, C], 1))  # M = 0.5 (B C^T + C B^T) = Q S Q^T on span[B C]
        r = B.shape[1]
        S = 0.5 * (R[:, :r] @ R[:, r:].T + R[:, r:] @ R[:, :r].T)
        ev, V = torch.linalg.eigh(S)
        U = Q @ V
        kind, li = key[0], int(key[1:])
        keep = [i for i in range(len(ev)) if abs(float(ev[i])) >= a.min_alpha]
        keep = sorted(keep, key=lambda i: -abs(float(ev[i])))[: a.max_dirs]
        for j, i in enumerate(keep):
            arrays[f"{key}_{j}"] = U[:, i].numpy().astype(np.float16)
            ops.append({"op": "dir_ablate", "tensor": f"model.language_model.layers.{li}.{WRITERS[kind]}.weight",
                        "u": f"{a.name}.npz#{key}_{j}", "alpha": round(float(ev[i]) * a.scale, 6), "side": "out"})
            alphas.append(float(ev[i]) * a.scale)
            n_dirs += 1
    from wsp import case as WC   # the parent as the toolkit's own case records it (wsp.case init copies the start verbatim)
    conf = WC.conf(a.case)
    spec = {"name": a.name, "base": conf.get("profile", "qwen35-0.8b-think"), "ops": ops}
    if conf.get("parent"):
        spec["parent"] = os.path.relpath(WC.cdir(a.case) / conf["parent"], root)
    json.dump(spec, open(root / f"{a.name}.json", "w"), indent=1)
    np.savez(root / f"{a.name}.npz", **arrays)
    al = np.abs(np.array(alphas)) if alphas else np.zeros(1)
    print(f"{a.name}: {n_dirs} directions on {len(keys)} writers, |alpha| median {np.median(al):.3f} max {al.max():.3f}, "
          f"payload {sum(v.nbytes for v in arrays.values()) / 1024:.0f} KiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("model")
    t.add_argument("train")
    t.add_argument("out")
    t.add_argument("--heldout")
    t.add_argument("--rank", type=int, default=4)
    t.add_argument("--kinds", nargs="+", default=["o", "g", "d"])
    t.add_argument("--skip-last-mlp", action="store_true")
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--max-tokens", type=int, default=6000)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--ckpt", action="store_true")
    t.add_argument("--units", type=int, default=0, help="MLP units to rewrite (0 = none)")
    t.add_argument("--sal-samples", type=int, default=64)
    t.add_argument("--exclude-units", help="published edit.json of the parent: units its chain rewrote are not chosen")
    t.add_argument("--gains", action="store_true", help="train one output gain per attention head (scale_cols)")
    t.add_argument("--unit-gains", action="store_true", help="train one gain per MLP unit (scale_rows on up_proj)")
    t.add_argument("--gain-layers", type=int, nargs=2, default=[0, 99], help="layer range of the gains")
    t.add_argument("--distill-tau", type=float, default=0.0, help="match the unpatched model at this temperature (0 = likelihood training)")
    t.add_argument("--distill-until", type=int, default=0, help="sharpen only the first N target tokens (0 = all)")
    t.add_argument("--distill-positions", type=int, default=1536, help="positions per answer used by the distillation loss (0 = all)")
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export")
    e.add_argument("out")
    e.add_argument("case")
    e.add_argument("name")
    e.add_argument("--stack")
    e.add_argument("--step", type=int, default=0)
    e.add_argument("--scale", type=float, default=1.0)
    e.add_argument("--min-alpha", type=float, default=0.0)
    e.add_argument("--max-dirs", type=int, default=64)
    e.add_argument("--min-gain", type=float, default=0.005)
    e.add_argument("--model", help="model dir the patch was trained on (needed to export rewritten units)")
    e.add_argument("--avg", nargs="*", default=[], help="more fits (OUT prefixes) averaged with this one")
    e.add_argument("--levels", type=int, default=32, help="quantisation levels per layer for unit gains")
    a = ap.parse_args()
    (train if a.cmd == "train" else export)(a)


if __name__ == "__main__":
    main()
