"""Forced recall for maths problems: the model's thinking is started with one sentence in its own voice that states a
general way of working (PREFIX below; it never refers to the question, an option or a hint), and the model continues
freely with the profile's sampling settings. Answers (sentence included) are graded as usual, so the right ones are
natural text after the PLAIN question. Only maths problems (source "math") of the set are used.

    python recall_gen.py CASE MODEL SET OUT --n 2 [--var plan|concise] [--shard i/k] [--plain]     (one GPU per process)
--plain: the same problems and seeds without the sentence (the comparison for the ceiling)."""
import sys

PREFIX = {
    "plan": "First, recall how to solve this kind of problem: I restate exactly what is asked, choose a method, work it out "
            "step by step with exact values (fractions and radicals rather than decimals), check the result against every "
            "condition of the problem, and give it in simplest form in \\boxed{}.\n\n",
    "concise": "First, recall how to work efficiently: I keep the working short, do not repeat a check I have already done, "
               "and once a result is checked I write it in \\boxed{}.\n\n",
}


def main():
    from vllm import LLM, SamplingParams
    from wsp import case as C, data, profiles
    from wsp.common import stable_seed, write_jsonl
    from wsp.grade import score
    a = sys.argv[1:]
    case, model, s, out = a[:4]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 2
    var = a[a.index("--var") + 1] if "--var" in a else "plan"
    pre = "" if "--plain" in a else PREFIX[var]
    i, k = map(int, a[a.index("--shard") + 1].split("/")) if "--shard" in a else (0, 1)
    _, items = C.set_items(case, s)
    items = [r for r in items if str(r.get("source", "")).startswith("math")][i::k]
    prof = profiles.get(C.conf(case)["profile"])
    samp = dict(prof["sampling"])
    llm = LLM(model=str(C.model_path(case, model)), dtype="bfloat16", max_model_len=samp["max_tokens"] + 2048,
              gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()
    reqs = []
    for r in items:
        p = tok.apply_chat_template([{"role": "user", "content": data.user_message(r)}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=prof["thinking"]) + pre
        reqs += [(r, j, p) for j in range(n)]
    outs = llm.generate([p for *_, p in reqs], [SamplingParams(seed=stable_seed(r["id"], j, "pv"), **samp)
                                                 for r, j, _ in reqs])
    rows = []
    for (r, j, _), o in zip(reqs, outs):
        c = o.outputs[0]
        text = pre + c.text
        rows.append({"id": r["id"], "k": j, "source": r.get("source"), "gold": r.get("answer"),
                     **score(r, text, c.finish_reason, prof["thinking"]), "text": text})
    write_jsonl(out, rows)
    print(f"{len(rows)} answers, right {sum(x['correct'] for x in rows) / len(rows):.3f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    main()
