"""Preference fit of per-unit / per-head gains on answers written after the same thinking.

Pairs come from derive/branch.py output (--kinds if [--if-all]): rows that share a branch point (id, k, point) have
the same prefix (prompt side: the model's own thinking up to </think>) and differ only in the visible answer. A pair is
(an answer that passes the item's checker, one that fails), both finished. Loss per pair, on the answer tokens only:
  -log sigmoid(beta * ((logp(c) - logp0(c)) - (logp(r) - logp0(r)))) + nll_w * NLL(c) / |c|
logp0: the start model (gains off). With --keep ROWS --keep-w W, each pair's step also holds one of the start model's
own finished answers of other families (calculation practice) at the start model: W * KL(p0 || p) over the vocabulary,
averaged over its first --keep-len generated positions (thinking and answer). Parameters: one multiplicative gain
per MLP unit (row of up_proj) and, with --heads, per attention head (column block of o_proj / out_proj), in --layers.
Saves state.pt in fit_gains format: average several seeds and dose them with
`fit_gains.py export --fits F1 F2 ... --k K`.

  python derive/pref_gains.py --model DIR --branches BR.jsonl ... --out fits/P1 [--layers 4-21] [--heads]
        [--beta 0.1] [--nll-w 0.2] [--max-pairs 2] [--epochs 2] [--accum 4] [--lr 2e-3] [--seed 1] [--exclude ids.json]
        [--keep CALC.jsonl --keep-w 30 --keep-len 4096] [--chunk 512]
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_gains import heads_of, parse_layers, read_rows  # noqa: E402


def make_pairs(rows, max_pairs, rng, exclude, allow_think=False, len_match=0.0, bad_cut=False, points=None, min_cont=0):
    """11-22 (jude): allow_think keeps thinking-stage continuations (math 'wrong'/'open' branches, whose rescues all close
    the thinking); len_match R > 0 pairs a rescue only with a failure whose continuation length is within a factor R
    (so the preference cannot become a length lever)."""
    groups = collections.defaultdict(lambda: ([], []))
    for r in rows:
        cont = r["text"][len(r.get("prefix", "")):]
        if points and r.get("point") not in points:
            continue
        cut_fail = bad_cut and r.get("finish") != "stop" and not r.get("correct")      # an unfinished continuation = a failure
        if r["id"] in exclude or (r.get("finish") != "stop" and not cut_fail) or (not allow_think and "</think>" in cont):
            continue
        if min_cont and len(cont) < min_cont:
            continue
        good, bad = groups[(r["id"], r["k"], r.get("point"))]
        (good if r.get("correct") else bad).append(r)
    pairs = []
    for key, (good, bad) in groups.items():
        if not good or not bad:
            continue
        rng.shuffle(good)
        rng.shuffle(bad)
        if len_match and len_match > 0:
            L = lambda r: max(1, len(r["text"]) - len(r.get("prefix", "")))
            cand = [(g, b) for g in good for b in bad if max(L(g), L(b)) / min(L(g), L(b)) <= len_match]
            rng.shuffle(cand)
            pairs.extend(cand[:max_pairs])
            continue
        for i in range(min(max_pairs, max(len(good), len(bad)))):
            pairs.append((good[i % len(good)], bad[i % len(bad)]))
    rng.shuffle(pairs)
    return pairs, len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--branches", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="4-21")
    ap.add_argument("--heads", action="store_true")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll-w", type=float, default=0.2)
    ap.add_argument("--max-pairs", type=int, default=2)
    ap.add_argument("--allow-think", action="store_true", help="keep thinking-stage continuations (math branch rescues)")
    ap.add_argument("--len-match", type=float, default=0.0, help="pair only continuations within this length ratio (0 = off)")
    ap.add_argument("--window", type=int, default=0,
                    help="EARLY-DECISION credit: only the first N tokens after the branch point enter the preference and the NLL "
                         "(0 = the whole continuation). The outcome of a maths trace is sealed early; the rest is noise.")
    ap.add_argument("--bad-cut", action="store_true", help="unfinished (cut) continuations count as failures")
    ap.add_argument("--points", type=float, nargs="*", help="use only branch points with these values")
    ap.add_argument("--min-cont", type=int, default=0, help="skip continuations shorter than this many characters")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--exclude", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--keep", nargs="*", help="the start model's own finished answers (wsp.gen rows, any task) held at "
                    "the start model: KL(p0 || p) over the vocabulary at their positions (thinking and answer), mean "
                    "per position, one row per pair")
    ap.add_argument("--keep-w", type=float, default=0.0)
    ap.add_argument("--keep-len", type=int, default=4096, help="generated tokens of a keep row that count")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from wsp import data
    from wsp.probe import decoder, load_model

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = set()
    for p in a.exclude or []:
        exclude |= set(json.load(open(p)))
    pairs, n_groups = make_pairs(read_rows(a.branches), a.max_pairs, rng, exclude, a.allow_think, a.len_match,
                                 a.bad_cut, set(a.points) if a.points else None, a.min_cont)
    if a.limit:
        pairs = pairs[: a.limit]
    by_src = collections.Counter(g.get("source") for g, _ in pairs)
    print(f"[pref] {len(pairs)} pairs from {n_groups} branch points {dict(by_src)}", flush=True)
    keep = []
    if a.keep and a.keep_w > 0:
        keep = [r for r in read_rows(a.keep) if r["id"] not in exclude and r.get("finish") == "stop"
                and r.get("think_closed")]
        rng.shuffle(keep)
        print(f"[pref] {len(keep)} keep rows {dict(collections.Counter(r.get('source') for r in keep))}, "
              f"weight {a.keep_w}", flush=True)
    if not pairs:
        raise SystemExit("no pairs")

    model, tok = load_model(a.model)
    if a.window:      # a pair whose two windows start with the same tokens says nothing about the early decision
        def _w(r): return tok(r["text"][len(r["prefix"]):], add_special_tokens=False).input_ids[: a.window]
        kept = []
        for g_, b_ in pairs:
            wg, wb = _w(g_), _w(b_)
            same = next((i for i, (x, y) in enumerate(zip(wg, wb)) if x != y), min(len(wg), len(wb)))
            if same < 0.5 * min(len(wg), len(wb)) and min(len(wg), len(wb)) >= 64:
                kept.append((g_, b_))
        print(f"[pref] window {a.window}: {len(kept)} of {len(pairs)} pairs differ within the first half of the window", flush=True)
        pairs = kept
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config.get_text_config()
    owner, layers = decoder(model)
    names = {m: n for n, m in model.named_modules()}
    L = parse_layers(a.layers)
    dev = next(model.parameters()).device
    mlp = {i: torch.zeros(layers[i].mlp.up_proj.out_features, device=dev, requires_grad=True) for i in L}
    head = {}
    on = [True]
    hooks = []
    meta = {"layers": L, "mlp_tensor": {}, "head_tensor": {}, "head_dim": {}, "n_heads": {}}
    for i in L:
        up = layers[i].mlp.up_proj
        meta["mlp_tensor"][i] = names[up] + ".weight"
        hooks.append(up.register_forward_hook(lambda m, inp, o, i=i: o * torch.exp(mlp[i]).to(o.dtype) if on[0] else o))
        if a.heads:
            proj, H, hd = heads_of(layers[i], cfg)
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

    def encode(r):
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[r["id"]], None)}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        p = tok(prompt, add_special_tokens=False).input_ids
        pre = tok(r["prefix"], add_special_tokens=False).input_ids
        ans = tok(r["text"][len(r["prefix"]):], add_special_tokens=False).input_ids + [eot]
        if a.window:
            ans = ans[: a.window]
        ids = (p + pre + ans)[: a.max_len]
        return torch.tensor(ids, device=dev), len(p) + len(pre)

    def encode_keep(r):
        prompt = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[r["id"]], None)}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True)
        p = tok(prompt, add_special_tokens=False).input_ids
        gen = (tok(r["text"], add_special_tokens=False).input_ids + [eot])[: a.keep_len]
        return torch.tensor((p + gen)[: a.max_len], device=dev), len(p)

    def lp_chunk(h, t):
        return -F.cross_entropy(lm_head(h).float(), t, reduction="sum")

    def seq_logp(ids, start, grad):
        """Sum of log p of the answer tokens (positions start..end) under the current gains (on[0])."""
        if grad:
            hs = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        else:
            with torch.no_grad():
                hs = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        tot = 0.0
        for s in range(start - 1, len(ids) - 1, a.chunk):
            e = min(s + a.chunk, len(ids) - 1)
            if grad:
                tot = tot + checkpoint(lp_chunk, hs[s:e], ids[s + 1:e + 1], use_reentrant=False)
            else:
                with torch.no_grad():
                    tot = tot + lp_chunk(hs[s:e], ids[s + 1:e + 1])
        return tot, len(ids) - start

    def kl_chunk(h_off, h_on):
        lo = F.log_softmax(lm_head(h_off).float(), -1)
        ln = F.log_softmax(lm_head(h_on).float(), -1)
        return (lo.exp() * (lo - ln)).sum()

    def seq_kl(ids, start):
        """Mean over the generated positions of KL(p0 || p) over the whole vocabulary (exact, not sampled)."""
        on[0] = False
        with torch.no_grad():
            h0 = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        on[0] = True
        h1 = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        tot, n = 0.0, len(ids) - start
        for s in range(start - 1, len(ids) - 1, a.chunk):
            e = min(s + a.chunk, len(ids) - 1)
            tot = tot + checkpoint(kl_chunk, h0[s:e], h1[s:e], use_reentrant=False)
        return tot / max(1, n)

    t0, run, acc_run, step = time.time(), [], [], 0
    kl_run, ki = [], 0
    total = len(pairs) * a.epochs
    for ep in range(a.epochs):
        rng.shuffle(pairs)
        for j, (good, bad) in enumerate(pairs):
            gi, gs = encode(good)
            bi, bs = encode(bad)
            if gs >= len(gi) - 1 or bs >= len(bi) - 1:
                continue
            on[0] = False
            lg0, _ = seq_logp(gi, gs, False)
            lb0, _ = seq_logp(bi, bs, False)
            on[0] = True
            lg, ng = seq_logp(gi, gs, True)
            lb, _ = seq_logp(bi, bs, True)
            margin = a.beta * ((lg - lg0.detach()) - (lb - lb0.detach()))
            loss = -F.logsigmoid(margin) + a.nll_w * (-lg / max(1, ng))
            if keep:
                ki_ids, ki_s = encode_keep(keep[ki % len(keep)])
                ki += 1
                kl = seq_kl(ki_ids, ki_s)
                loss = loss + a.keep_w * kl
                kl_run.append(float(kl))
            (loss / a.accum).backward()
            run.append(float(loss))
            acc_run.append(float(margin > 0))
            if (j + 1) % a.accum == 0 or j == len(pairs) - 1:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            done = ep * len(pairs) + j + 1
            if done % 40 == 0 or done == total:
                el = time.time() - t0
                spread = max(float(q.detach().abs().max()) for q in params)
                kls = f" keep-kl {sum(kl_run[-40:]) / len(kl_run[-40:]):.5f}" if kl_run else ""
                print(f"[pref] {done}/{total} loss {sum(run[-40:]) / len(run[-40:]):.4f} "
                      f"pref-acc {sum(acc_run[-40:]) / len(acc_run[-40:]):.2f}{kls} max|log g| {spread:.4f} "
                      f"{el / 60:.1f} min, eta {el / done * (total - done) / 60:.1f} min", flush=True)
    for h in hooks:
        h.remove()
    torch.save({"mlp": {i: t.detach().cpu() for i, t in mlp.items()},
                "head": {i: t.detach().cpu() for i, t in head.items()}, "meta": meta,
                "args": vars(a), "n_answers": len(pairs),
                "by_source": dict(by_src)}, out / "state.pt")
    print(f"[pref] saved {out / 'state.pt'} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
