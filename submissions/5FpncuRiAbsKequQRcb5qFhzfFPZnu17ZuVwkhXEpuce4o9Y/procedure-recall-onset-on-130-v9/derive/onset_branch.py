"""onset_branch.py = the published branch2.py of #107 (0009568831-0002) and #106 (0009568726-0005), the same file in both,
credited, plus the kind 'onset' (#138, 0009578577-0019, this team) and the usage line below naming this file: rows of
if_onset.py are continued from their own prefix (inside the visible answer, just before the word that broke the
request); the original failed answer is written too (j = -1) as that onset's own failure.

Rescues: continuations a model writes from inside its own failed practice answers, kept when they end right.

  python derive/onset_branch.py --model DIR --rows A.jsonl [B.jsonl ...] --out rescues.jsonl
        [--k 4] [--salt rdb] [--points 0.5 0.75] [--max-new 12000] [--if-max-new 4000] [--min-think 1000]
        [--penalty prefix|none] [--kinds open wrong if] [--limit N] [--gpu-mem 0.9]

Failed answers (wsp.gen / wsp.case run rows of that same model):
  open   thinking still open when the answer was cut at the token cap;
  wrong  finished with the thinking closed but wrong (or without an answer), after >= --min-think thinking tokens;
  if     an instruction-following answer (ifsynth, tuluif) that finished but fails its checker.
Branch points: open: the last paragraph break at or before each --points share of the answer; wrong: the same within
the thinking; if: right after </think> (only the visible answer is sampled again).
Sampling: the chat prompt plus the answer's own text up to the branch point is the prompt; --k continuations with the
profile's sampler. In the full run the presence penalty also covers the prefix's tokens (they were generated):
--penalty prefix emulates that with a logit bias of -penalty on the prefix's tokens (its 1024 most frequent: vLLM's
limit per request) and no penalty inside the continuation; --penalty none leaves the prefix unpenalised and keeps vLLM's penalty on the continuation.
Each row: id, k, source, branch (open / wrong / if), point, j, prefix (text up to the branch point), text (prefix +
continuation), cont_tokens, finish, the grade of prefix + continuation (wsp.grade.score), rescue (bool, see below).
A rescue closes the thinking inside the continuation (open / wrong), stops by itself, is graded right and is at least
200 tokens long; for 'if' the new visible answer is right, at least 20 tokens and at least half the original's length.
On multiple choice a point counts only when at least 2 of its continuations are right (a lucky letter is not a rescue).
"""
import argparse
import collections
import json
import sys
import time


def read_rows(paths):
    out = []
    for p in paths:
        with open(p) as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def cut_at_break(text, upto):
    """Index just after the last paragraph break at or before char position upto (None when there is none)."""
    i = text.rfind("\n\n", 0, upto)
    return None if i <= 0 else i + 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="qwen35-0.8b-think")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--salt", default="rdb")
    ap.add_argument("--points", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--max-new", type=int, default=12000)
    ap.add_argument("--if-max-new", type=int, default=4000)
    ap.add_argument("--min-think", type=int, default=1000)
    ap.add_argument("--penalty", choices=["prefix", "none"], default="prefix")
    ap.add_argument("--kinds", nargs="+", default=["open", "wrong", "if"])
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--if-all", action="store_true", help="'if': branch every finished answer, passing ones too")
    ap.add_argument("--close-at", type=int, default=28000,
                    help="'close': answers cut with the thinking open are closed (</think>) at the last paragraph "
                         "break before this many generated tokens, and only the visible answer is sampled")
    ap.add_argument("--close-text", default="\n</think>\n\n", help="'close': text appended at the break")
    ap.add_argument("--pc-stem", default="", help="'postclose': text appended after the first </think> before sampling the answer (e.g. a boxed-result stem); default none")
    ap.add_argument("--mid-cue", default=None,
                    help="'open' / 'wrong': text inserted at the branch point ({problem} becomes the problem); its "
                         "tokens carry no presence-penalty bias")
    ap.add_argument("--cue", default=None,
                    help="'if': text inserted at the end of the thinking, just before </think> ({request} becomes the "
                         "request); its tokens carry no presence-penalty bias")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from wsp import data, profiles
    from wsp.common import stable_seed
    from wsp.grade import score

    prof = profiles.get(a.profile)
    idx = data.by_id()
    tok = AutoTokenizer.from_pretrained(a.model)
    samp = dict(prof["sampling"])
    pen = float(samp.pop("presence_penalty", 0.0))
    samp.pop("max_tokens", None)

    jobs = []  # (row, kind, point, prefix)
    for r in read_rows(a.rows):
        item = idx.get(r["id"])
        if item is None or (a.sources and r.get("source") not in a.sources):
            continue
        text = r["text"]
        src = r.get("source")
        if "onset" in a.kinds and r.get("onset_kind"):
            # if_onset.py row: continue from inside the visible answer, just before the word that broke the request
            jobs.append((r, "onset", float(r["onset_frac"]), r["prefix"]))
            continue
        if "close" in a.kinds and r.get("cut") and not r.get("think_closed"):
            ids = tok(text, add_special_tokens=False).input_ids
            upto = len(tok.decode(ids[: a.close_at]))
            c = cut_at_break(text, upto)
            if c:
                jobs.append((r, "close", float(a.close_at), text[:c].rstrip("\n") + a.close_text.replace("\\n", "\n")))
            continue
        if "open" in a.kinds and r.get("cut") and not r.get("think_closed"):
            for p in a.points:
                c = cut_at_break(text, int(len(text) * p))
                if c:
                    jobs.append((r, "open", p, text[:c]))
        elif ("wrong" in a.kinds and r.get("finish") == "stop" and r.get("think_closed") and not r.get("correct")
              and src not in ("ifsynth", "tuluif")):
            think = text[: text.find("</think>")]
            if len(tok(think, add_special_tokens=False).input_ids) >= a.min_think:
                for p in a.points:
                    c = cut_at_break(think, int(len(think) * p))
                    if c:
                        jobs.append((r, "wrong", p, text[:c]))
        elif ("right" in a.kinds and r.get("finish") == "stop" and r.get("think_closed") and r.get("correct") is True
              and src not in ("ifsynth", "tuluif")):
            # value-curve diagnostic: continuations of a CORRECT trace from an early point (how often does a good start survive?)
            think = text[: text.find("</think>")]
            if len(tok(think, add_special_tokens=False).input_ids) >= a.min_think:
                for p in a.points:
                    c = cut_at_break(think, int(len(think) * p))
                    if c:
                        jobs.append((r, "right", p, text[:c]))
        elif ("if" in a.kinds and src in ("ifsynth", "tuluif") and r.get("finish") == "stop" and r.get("think_closed")
              and (a.if_all or not r.get("correct"))):
            e = text.find("</think>") + len("</think>")
            jobs.append((r, "if", 1.0, text[:e]))
        elif ("postclose" in a.kinds and src not in ("ifsynth", "tuluif") and r.get("think_closed")
              and (text.count("</think>") >= 2 or r.get("kind") == "finished_no_answer")):
            # an answer that went wrong after its first close (thinking re-opened, or no result): keep the thinking
            # up to the first close and sample only the answer again (optionally after a result stem)
            e = text.find("</think>") + len("</think>")
            stem = a.pc_stem.replace("\\n", "\n") if a.pc_stem else ""
            jobs.append((r, "postclose", 1.0, text[:e] + stem))
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"[branch] {len(jobs)} branch points: {dict(collections.Counter(j[1] for j in jobs))}", flush=True)
    if not jobs:
        open(a.out, "w").close()
        return 0

    prompts, params, meta = [], [], []
    longest = 0
    for r, kind, p, prefix in jobs:
        chat = tok.apply_chat_template([{"role": "user", "content": data.user_message(idx[r["id"]], None)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=prof["thinking"])
        pre_ids = tok(prefix, add_special_tokens=False).input_ids   # the bias covers the generated text only
        prefix0 = prefix
        if kind in ("open", "wrong") and a.mid_cue:
            prefix = prefix + a.mid_cue.replace("\\n", "\n").replace("{problem}", idx[r["id"]]["problem"].strip())
        if kind == "if" and a.cue:
            e = prefix.rfind("</think>")
            cue = a.cue.replace("\\n", "\n").replace("{request}", idx[r["id"]]["problem"].strip())
            prefix = prefix[:e].rstrip("\n") + "\n\n" + cue + "\n" + prefix[e:]
        longest = max(longest, len(tok(chat, add_special_tokens=False).input_ids) + len(pre_ids))
        mx = a.if_max_new if kind in ("if", "close", "postclose", "onset") else a.max_new
        # vLLM takes at most 1024 logit-bias tokens per request: the prefix's most frequent tokens carry the penalty
        top = [t for t, _ in collections.Counter(pre_ids).most_common(1024)]
        extra = {"presence_penalty": pen} if a.penalty == "none" else \
            {"presence_penalty": 0.0, "logit_bias": {int(t): -pen for t in top}}
        for j in range(a.k):
            prompts.append(chat + prefix)
            params.append(SamplingParams(seed=stable_seed(r["id"], r["k"], kind, p, j, a.salt), max_tokens=mx,
                                         **samp, **extra))
            meta.append((r, kind, p, j, prefix, len(pre_ids), prefix0))
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=longest + max(a.max_new, a.if_max_new) + 256,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
    t0 = time.time()
    outs = llm.generate(prompts, params)
    print(f"[branch] {len(outs)} continuations in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for (r, kind, p, j, prefix, npre, prefix0), o in zip(meta, outs):
        c = o.outputs[0]
        full = prefix + c.text
        g = score(idx[r["id"]], full, c.finish_reason, prof["thinking"])
        n = len(c.token_ids)
        if kind == "close":
            ok = g["correct"] and c.finish_reason == "stop"
        elif kind == "postclose":
            ok = (g["correct"] and c.finish_reason == "stop" and "</think>" not in c.text and "<think>" not in c.text)
        elif kind == "onset":       # the visible answer = the text after the LAST close, as the grader reads it
            e0 = full.rfind("</think>") + len("</think>")
            vis_new = len(tok(full[e0:], add_special_tokens=False).input_ids)
            vis_old = len(tok(r["text"][r["text"].rfind("</think>") + len("</think>"):], add_special_tokens=False).input_ids)
            ok = g["correct"] and c.finish_reason == "stop" and vis_new >= 20 and vis_new >= vis_old / 2
        elif kind == "if":
            vis_new = len(tok(c.text, add_special_tokens=False).input_ids)
            vis_old = len(tok(r["text"][r["text"].find("</think>") + len("</think>"):], add_special_tokens=False).input_ids)
            ok = g["correct"] and c.finish_reason == "stop" and vis_new >= 20 and vis_new >= vis_old / 2
        else:
            ok = g["correct"] and c.finish_reason == "stop" and "</think>" in c.text and n >= 200
        rows.append({"id": r["id"], "k": r["k"], "source": r.get("source"), "point": p, "j": j,
                     "prefix_tokens": npre, "cont_tokens": n, "finish": c.finish_reason, **g, "branch": kind,
                     "orig_kind": r.get("kind"), "rescue": bool(ok), "prefix": prefix, "text": full,
                     **({"prefix0": prefix0} if prefix0 != prefix else {})})
    if "onset" in a.kinds:
        for r, kind, p, prefix in jobs:
            if kind == "onset":
                rows.append({"id": r["id"], "k": r["k"], "source": r.get("source"), "point": p, "j": -1,
                             "prefix_tokens": None, "cont_tokens": None, "finish": "stop", "correct": False,
                             "branch": "onset", "orig_kind": "original", "rescue": False, "prefix": prefix,
                             "text": r["text"]})
    # multiple choice: a point counts only with at least 2 right continuations
    right = collections.Counter((x["id"], x["k"], x["point"]) for x in rows if x["rescue"])
    for x in rows:
        if x["rescue"] and x["source"] == "mmlupro" and right[(x["id"], x["k"], x["point"])] < 2:
            x["rescue"] = False
    with open(a.out, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    by = collections.defaultdict(lambda: [0, 0, set(), set()])
    for x in rows:
        b = by[(x["source"], x["branch"], x["point"])]
        b[0] += 1
        b[1] += x["rescue"]
        b[2].add((x["id"], x["k"]))
        if x["rescue"]:
            b[3].add((x["id"], x["k"]))
    for key in sorted(by):
        n, res, ans, ans_ok = by[key]
        print(f"[branch] {key}: {res}/{n} continuations right; answers rescued {len(ans_ok)}/{len(ans)} "
              f"({100 * len(ans_ok) / max(1, len(ans)):.0f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
