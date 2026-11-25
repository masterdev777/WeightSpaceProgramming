"""Gains fitted by likelihood: one multiplicative gain per MLP unit (row of up_proj)
and, optionally, per attention head (column block of o_proj / out_proj), fitted by next-token loss on finished, correct
answers the starting model wrote (its own answers, and rescues: continuations from inside its failed answers). Exported as scale_rows /
scale_cols ops stacked on a case's parent, so the toolkit builds and runs it like any edit.

  python derive/fit_gains.py fit --model DIR --rows A.jsonl [B.jsonl ...] --out fits/F1 [--rescues R.jsonl ...]
        [--rescue-branches if] [--per-point 2] [--anchor-nll answer] [--close-tether 1.0] [--layers 14-21] [--heads]
        [--lr 2e-3] [--epochs 1] [--seed 1] [--subsample 0.8] [--loss sampler] [--tether 1.0] [--exclude ids.json ...]
  python derive/fit_gains.py export --fits fits/F1 [fits/F2 ...] --k 3.0 --name A30 --spec-dir cases/CASE/edits
        --parent ../parent/edit.json [--levels 32] [--shuffle SEED]

Rows are wsp.gen / wsp.case run output (id, k, source, correct, cut, think_closed, finish, text). The prompt is rebuilt
exactly as wsp.gen builds it (chat template, thinking on); the loss covers the generated tokens and the end-of-turn
token. Gains start at 1 (log-gain 0); the export averages the fits' log-gains, scales them by the dose k, quantises
each layer's unit gains to --levels levels and writes one op per level (and one per head).
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def select(rows, per_problem, seed, subsample, sources, exclude, max_tokens):
    """Finished, correct, thinking closed; up to per_problem per problem, at random (not by length)."""
    rng = random.Random(seed)
    by = {}
    for r in rows:
        if r["id"] in exclude or (sources and r.get("source") not in sources):
            continue
        if not (r.get("correct") and r.get("think_closed") and r.get("finish") == "stop" and not r.get("cut")):
            continue
        if max_tokens and r.get("n_tokens", 0) > max_tokens:
            continue
        by.setdefault(r["id"], []).append(r)
    keep = []
    for pid in sorted(by):
        rs = by[pid]
        rng.shuffle(rs)
        keep += rs[:per_problem]
    rng.shuffle(keep)
    if subsample < 1:
        keep = keep[: max(1, int(round(len(keep) * subsample)))]
    return keep


def select_rescues(rows, seed, subsample, exclude, max_share=0.45, per_point_max=1, per_answer_max=2):
    """Rescue rows (derive/branch.py): at most per_point_max per branch point and per_answer_max per failed answer;
    no source above max_share."""
    rng = random.Random(seed)
    rows = [r for r in rows if r.get("rescue") and r["id"] not in exclude]
    rng.shuffle(rows)
    per_point, per_answer, keep = {}, {}, []
    for r in rows:
        pt, ans = (r["id"], r["k"], r["branch"], r["point"]), (r["id"], r["k"])
        if per_point.get(pt, 0) >= per_point_max or per_answer.get(ans, 0) >= per_answer_max:
            continue
        per_point[pt] = per_point.get(pt, 0) + 1
        per_answer[ans] = per_answer.get(ans, 0) + 1
        keep.append(r)
    by = {}
    for r in keep:
        by.setdefault(r.get("source"), []).append(r)
    cap = max(1, int(max_share * len(keep))) if len(by) > 1 else len(keep)
    keep = [r for v in by.values() for r in v[:cap]]
    rng.shuffle(keep)
    if subsample < 1:
        keep = keep[: max(1, int(round(len(keep) * subsample)))]
    return keep


def heads_of(layer, cfg):
    """(module, n_heads, head_dim) of the attention output projection of a decoder layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn.o_proj, cfg.num_attention_heads, cfg.head_dim
    return layer.linear_attn.out_proj, cfg.linear_num_value_heads, cfg.linear_value_head_dim


def cmd_fit(a):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from wsp import data
    from wsp.probe import decoder, load_model

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = set()
    for p in a.exclude or []:
        exclude |= set(json.load(open(p)))
    rows = select(read_rows(a.rows), a.per_problem, a.seed, a.subsample, set(a.sources or []), exclude,
                  a.max_answer_tokens)
    if a.rescues:
        resc_rows = read_rows(a.rescues)
        if a.rescue_branches:
            resc_rows = [r for r in resc_rows if r.get("branch") in set(a.rescue_branches)]
        resc = select_rescues(resc_rows, a.seed, a.subsample, exclude, per_point_max=a.per_point,
                              per_answer_max=max(2, a.per_point))
        print(f"[fit] {len(resc)} rescues {dict(collections.Counter(r.get('source') for r in resc))} "
              f"{dict(collections.Counter(r['branch'] for r in resc))}", flush=True)
        rng = random.Random(a.seed + 100)
        rng.shuffle(rows)
        anchors = rows[: len(resc)]
        rows = [x for pair in zip(resc, anchors) for x in pair] + resc[len(anchors):]
        a.accum = 2
    if a.limit:
        rows = rows[: a.limit]
    by_src = {}
    for r in rows:
        by_src[r.get("source")] = by_src.get(r.get("source"), 0) + 1
    print(f"[fit] {len(rows)} answers {by_src}", flush=True)

    model, tok = load_model(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config.get_text_config()
    owner, layers = decoder(model)
    names = {m: n for n, m in model.named_modules()}
    L = parse_layers(a.layers)
    dev = next(model.parameters()).device
    mlp = {i: torch.zeros(layers[i].mlp.up_proj.out_features, device=dev, requires_grad=True) for i in L}
    head = {}
    on = [True]  # gains switched off for the start model's pass (tether)
    hooks = []
    meta = {"layers": L, "mlp_tensor": {}, "head_tensor": {}, "head_dim": {}, "n_heads": {}}
    for i in L:
        up = layers[i].mlp.up_proj
        meta["mlp_tensor"][i] = names[up] + ".weight"
        hooks.append(up.register_forward_hook(lambda m, inp, o, i=i: o * torch.exp(mlp[i]).to(o.dtype) if on[0] else o))
        if a.heads:
            proj, H, hd = heads_of(layers[i], cfg)
            if proj.in_features != H * hd:
                raise SystemExit(f"layer {i}: {proj.in_features} inputs, expected {H} x {hd}")
            head[i] = torch.zeros(H, device=dev, requires_grad=True)
            meta["head_tensor"][i], meta["head_dim"][i], meta["n_heads"][i] = names[proj] + ".weight", hd, H

            def pre(m, args, i=i, hd=hd):
                if not on[0]:
                    return None
                x = args[0]
                return (x * torch.exp(head[i]).to(x.dtype).repeat_interleave(hd),) + tuple(args[1:])
            hooks.append(proj.register_forward_pre_hook(pre))
    params = list(mlp.values()) + list(head.values())
    opt = torch.optim.Adam(params, lr=a.lr)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    lm_head = model.get_output_embeddings()
    idx = data.by_id()
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    V = lm_head.weight.shape[0]
    close_id = tok.convert_tokens_to_ids("</think>")
    nl_ids = torch.zeros(V, dtype=torch.bool, device=dev)       # tokens that end a line (where the model decides
    for t in range(min(V, len(tok))):                             # to go on, re-check or close the thinking)
        if tok.decode([t]).endswith("\n"):
            nl_ids[t] = True

    def encode(r):
        """-> ids, start of the completion, start of the loss, start of the visible answer (or None)."""
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[r["id"]], None)}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        p = tok(prompt, add_special_tokens=False).input_ids
        text = r["text"]
        if r.get("prefix"):            # a rescue: the prefix is context only
            pre = tok(r["prefix"], add_special_tokens=False).input_ids
            c = pre + tok(text[len(r["prefix"]):], add_special_tokens=False).input_ids + [eot]
            lstart = len(p) + len(pre)
        else:
            c = tok(text, add_special_tokens=False).input_ids + [eot]
            lstart = len(p)
        ids = (p + c)[: a.max_len]
        e = text.find("</think>")
        ans = None if e < 0 or r.get("prefix") else len(p) + len(tok(text[: e + 8], add_special_tokens=False).input_ids)
        if a.anchor_nll == "answer" and not r.get("prefix"):
            lstart = ans if ans is not None else len(ids)          # no loss on any thinking token
        return torch.tensor(ids, device=dev), len(p), lstart, ans

    def chunk_loss(h, t):
        return F.cross_entropy(lm_head(h).float(), t, reduction="sum")

    def chunk_loss_sampler(h, t, seen):
        z = lm_head(h).float()
        z = (z - a.pen * seen.to(z.dtype)) / a.temp
        return F.cross_entropy(z, t, reduction="sum")

    step, t0, run, ntok_all = 0, time.time(), [], 0
    total = len(rows) * a.epochs
    for ep in range(a.epochs):
        for j, r in enumerate(rows):
            ids, plen, lstart, ans = encode(r)
            if len(ids) - lstart < 2:
                continue
            tgt_lp = None
            nl_pos, nl_lp, close_lp0 = None, None, None
            if a.tether and ans is not None and ans < len(ids) - 1:
                ae = min(len(ids) - 1, ans + a.tether_max)
                with torch.no_grad():
                    on[0] = False
                    h0 = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
                    tgt_lp = torch.log_softmax(lm_head(h0[ans:ae]).float(), -1).to(torch.bfloat16)
                    if (a.think_tether or a.close_tether) and r.get("source") in ("gsm8k", "math", "mmlupro"):
                        tt = ids[plen:ans]
                        m = nl_ids[tt]
                        pos = (torch.nonzero(m).flatten() + plen)[: a.think_tether_max]
                        if len(pos):
                            nl_pos = pos
                            full = torch.log_softmax(lm_head(h0[pos]).float(), -1)
                            nl_lp = full.to(torch.bfloat16) if a.think_tether else None
                            close_lp0 = full[:, close_id].clone()
                            del full
                    on[0] = True
                    del h0
            hs = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
            loss, n = 0.0, 0
            if a.loss == "sampler":
                V = lm_head.weight.shape[0]
                first = torch.full((V,), 1 << 30, dtype=torch.long, device=dev)
                comp = ids[plen:]
                pos = torch.arange(plen, len(ids), device=dev)
                first.scatter_reduce_(0, comp, pos, reduce="amin")
            if lstart >= len(ids) - 1:
                continue
            for s in range(lstart - 1, len(ids) - 1, a.chunk):
                e = min(s + a.chunk, len(ids) - 1)
                if a.loss == "sampler":
                    t = torch.arange(s, e, device=dev)
                    seen = first[None, :] <= t[:, None]
                    loss = loss + checkpoint(chunk_loss_sampler, hs[s:e], ids[s + 1:e + 1], seen, use_reentrant=False)
                else:
                    loss = loss + checkpoint(chunk_loss, hs[s:e], ids[s + 1:e + 1], use_reentrant=False)
                n += e - s
            total_loss = loss / n
            if tgt_lp is not None:
                lp = torch.log_softmax(lm_head(hs[ans:ans + len(tgt_lp)]).float(), -1)
                kl = (tgt_lp.float().exp() * (tgt_lp.float() - lp)).sum(-1).mean()
                total_loss = total_loss + a.tether * kl
            if nl_pos is not None and (nl_lp is not None or close_lp0 is not None):
                lp2 = torch.log_softmax(lm_head(hs[nl_pos]).float(), -1)
                if nl_lp is not None:
                    kl2 = (nl_lp.float().exp() * (nl_lp.float() - lp2)).sum(-1).mean()
                    total_loss = total_loss + a.think_tether * kl2
                if close_lp0 is not None and a.close_tether:
                    # hold the readiness to end the thinking where it is: the log-probability of the close token at
                    # every line end of the thinking (a rare event the full-distribution KL hardly sees)
                    total_loss = total_loss + a.close_tether * ((lp2[:, close_id] - close_lp0) ** 2).mean()
            (total_loss / a.accum).backward()
            run.append(float(loss) / n)
            ntok_all += n
            if (j + 1) % a.accum == 0 or j == len(rows) - 1:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if (j + 1) % 20 == 0 or j == len(rows) - 1:
                done = ep * len(rows) + j + 1
                el = time.time() - t0
                spread = max(float(p.detach().abs().max()) for p in params)
                print(f"[fit] {done}/{total} loss {sum(run[-20:]) / len(run[-20:]):.4f} max|log g| {spread:.4f} "
                      f"{ntok_all / el:.0f} tok/s, {el / 60:.1f} min, eta {el / done * (total - done) / 60:.1f} min",
                      flush=True)
    for h in hooks:
        h.remove()
    torch.save({"mlp": {i: t.detach().cpu() for i, t in mlp.items()},
                "head": {i: t.detach().cpu() for i, t in head.items()}, "meta": meta,
                "args": vars(a), "n_answers": len(rows), "by_source": by_src}, out / "state.pt")
    print(f"[fit] saved {out / 'state.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def cmd_export(a):
    import numpy as np
    import torch

    from wsp import profiles
    from wsp.edits import decoder_prefix, tensor_index

    states = [torch.load(Path(f) / "state.pt", weights_only=False) for f in a.fits]
    meta = states[0]["meta"]
    rng = np.random.default_rng(a.shuffle) if a.shuffle is not None else None
    # module names (as the loaded class names them) -> tensor names of the checkpoint the subnet builds from
    index = tensor_index(profiles.base_dir(a.base))
    prefix = decoder_prefix(index)

    def tensor(name):
        t = prefix + "layers." + name.split("layers.", 1)[1]
        if t not in index:
            raise SystemExit(f"{name} -> {t}: not a tensor of the base checkpoint")
        return t

    meta["mlp_tensor"] = {i: tensor(n) for i, n in meta["mlp_tensor"].items()}
    meta["head_tensor"] = {i: tensor(n) for i, n in meta["head_tensor"].items()}
    ops, arrays = [], {}

    def add(op, key, arr):
        arrays[key] = np.asarray(arr, dtype=np.int32)
        op[("rows" if op["op"] == "scale_rows" else "cols")] = f"{a.name}.npz#{key}"
        ops.append(op)

    moved = 0
    for i in meta["layers"]:
        lg = a.k * torch.stack([s["mlp"][i] for s in states]).mean(0).numpy().astype(np.float64)
        if rng is not None:
            lg = rng.permutation(lg)
        order = np.argsort(lg)
        for q, grp in enumerate(np.array_split(order, a.levels)):
            if len(grp) == 0:
                continue
            m = float(lg[grp].mean())
            if abs(m) < a.min_log:
                continue
            moved += len(grp)
            add({"op": "scale_rows", "tensor": meta["mlp_tensor"][i], "factor": float(math.exp(m))}, f"m{i}_{q}",
                np.sort(grp))
        if i in meta["head_tensor"] and all(i in s["head"] for s in states):
            hg = a.k * torch.stack([s["head"][i] for s in states]).mean(0).numpy().astype(np.float64)
            if rng is not None:
                hg = rng.permutation(hg)
            hd = meta["head_dim"][i]
            for h, v in enumerate(hg):
                if abs(v) >= a.min_log:
                    add({"op": "scale_cols", "tensor": meta["head_tensor"][i], "factor": float(math.exp(v))},
                        f"h{i}_{h}", np.arange(h * hd, (h + 1) * hd))
    d = Path(a.spec_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{a.name}.npz", **arrays)
    spec = {"base": a.base, "parent": a.parent, "ops": ops} if a.parent else {"base": a.base, "ops": ops}
    (d / f"{a.name}.json").write_text(json.dumps(spec))
    size = sum(v.nbytes for v in arrays.values())
    allg = np.concatenate([a.k * torch.stack([s["mlp"][i] for s in states]).mean(0).numpy() for i in meta["layers"]])
    print(f"[export] {d / a.name}.json: {len(ops)} ops, {moved} units moved, payload {size / 1e6:.2f} MB, "
          f"unit gains {math.exp(allg.min()):.3f}-{math.exp(allg.max()):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", required=True)
    f.add_argument("--rows", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--layers", default="4-21")
    f.add_argument("--heads", action="store_true")
    f.add_argument("--lr", type=float, default=2e-3)
    f.add_argument("--epochs", type=int, default=1)
    f.add_argument("--accum", type=int, default=1)
    f.add_argument("--per-problem", type=int, default=2)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--subsample", type=float, default=1.0)
    f.add_argument("--max-len", type=int, default=12800)
    f.add_argument("--max-answer-tokens", type=int, default=0)
    f.add_argument("--chunk", type=int, default=2048)
    f.add_argument("--sources", nargs="*")
    f.add_argument("--exclude", nargs="*")
    f.add_argument("--limit", type=int)
    f.add_argument("--rescues", nargs="*", help="derive/branch.py output: rescues, each paired with one anchor")
    f.add_argument("--rescue-branches", nargs="*", help="only rescues of these branch kinds (open, wrong, if)")
    f.add_argument("--loss", choices=["plain", "sampler"], default="plain",
                   help="sampler: likelihood under the profile's sampler (presence penalty on tokens already in the "
                        "answer, then temperature)")
    f.add_argument("--pen", type=float, default=1.5)
    f.add_argument("--temp", type=float, default=0.6)
    f.add_argument("--tether", type=float, default=0.0, help="weight of KL(start || edit) on anchors' visible answers")
    f.add_argument("--tether-max", type=int, default=1024)
    f.add_argument("--anchor-nll", choices=["all", "answer"], default="all",
                   help="answer: anchors' likelihood only on the visible answer (no thinking token is trained)")
    f.add_argument("--per-point", type=int, default=1, help="rescues kept per branch point")
    f.add_argument("--think-tether", type=float, default=0.0,
                   help="weight of KL(start || edit) at line ends inside calc anchors' thinking")
    f.add_argument("--think-tether-max", type=int, default=512)
    f.add_argument("--close-tether", type=float, default=0.0,
                   help="weight of (log p_edit(</think>) - log p_start(</think>))^2 at line ends of calc anchors' thinking")
    e = sub.add_parser("export")
    e.add_argument("--fits", nargs="+", required=True)
    e.add_argument("--k", type=float, default=1.0)
    e.add_argument("--name", required=True)
    e.add_argument("--spec-dir", required=True)
    e.add_argument("--parent", default="../parent/edit.json")
    e.add_argument("--base", default="qwen35-0.8b-think")
    e.add_argument("--levels", type=int, default=32)
    e.add_argument("--min-log", type=float, default=1e-4)
    e.add_argument("--shuffle", type=int)
    a = ap.parse_args()
    {"fit": cmd_fit, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
