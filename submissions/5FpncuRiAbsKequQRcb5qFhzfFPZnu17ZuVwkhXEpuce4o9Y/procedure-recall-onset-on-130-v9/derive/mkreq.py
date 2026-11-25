"""Harder instruction requests, recombined from the practice sets' own material (no new text is written).

  python derive/mkreq.py --case CASE --out PATH.json [--side train|dev|holdout] [--seed 0]
                         [--n-rep 360] [--n-word 480] [--n-tulu 480]

Material
  * instruction sentences: every single-instruction request of practice:ifsynth is a practice word problem followed
    by one instruction sentence; cutting the word problem off leaves (sentence, checker arguments) pairs for 20
    instruction types. The repeat sentence is the one practice:tuluif uses.
  * bodies: practice word problems, practice multiple-choice and math prompts (worded as the toolkit words them), and
    requests of practice:tuluif together with the one instruction each already carries.
Requests
  rep    body + the repeat sentence; for a third of them one more instruction that may stand next to it comes first
         and is part of what has to be repeated (kinds repw / repc / repm / rept by the body: word problem,
         multiple choice, math, practice:tuluif request)
  word   a word problem + 2 or 3 instruction sentences of different types
  tulu   a practice:tuluif request (one instruction) + 1 or 2 more
Two instruction types that cannot hold together (ALONE and APART below, the practice sets' own combinations obey the
same table) are never put into one request.
Side "train" takes its bodies from outside the case's dev and holdout splits and outside every set under
cases/<case>/sets. Sides "dev" and "holdout" take theirs from that split of the case and nowhere else, and the
instruction sentences that "train" leaves out (every fifth): such a file shares neither bodies nor sentences with a
train file. Every request is a practice item in the toolkit's own format (the task and the grader of
practice:ifsynth), with source "ifsynth_x<kind>".
"""
import argparse, glob, json, os, random
from pathlib import Path

from wsp import case as C, data

REPEAT = (" First, repeat the request without change, then give your answer (do not say anything before repeating the "
          "request; the request you need to repeat does not include this sentence)")
RID = "combination:repeat_prompt"
# types that stand next to almost nothing: each is combined only with the types listed for it
ALONE = {RID: {"detectable_format:title", "keywords:existence", "punctuation:no_comma"},
         "combination:two_responses": {"detectable_format:title", "keywords:existence", "keywords:forbidden_words",
                                       "punctuation:no_comma"},
         "detectable_format:json_format": {"keywords:existence", "keywords:forbidden_words"}}
# groups of types of which a request carries at most one
APART = [{"change_case:capital_word_frequency", "change_case:english_capital", "change_case:english_lowercase"},
         {"detectable_format:multiple_sections", "detectable_format:number_highlighted_sections"},
         {"detectable_format:title", "startend:quotation"},
         {"length_constraints:nth_paragraph_first_word", "length_constraints:number_paragraphs"},
         {"length_constraints:number_paragraphs", "length_constraints:number_sentences"}]


def together(a, b):
    """May the instruction types a and b stand in one request?"""
    if a == b:
        return False
    for x, y in ((a, b), (b, a)):
        if x in ALONE and y not in ALONE[x]:
            return False
    return not any(a in g and b in g for g in APART)


def sentences(side):
    """(sentence, checker arguments) per instruction type, cut out of the single-instruction requests of ifsynth."""
    words = sorted((r["problem"].strip() for r in data.load("practice:gsm8k_train")), key=len, reverse=True)
    pool = {}
    for r in data.load("practice:ifsynth"):
        if len(r["instruction_id_list"]) != 1:
            continue
        body = next((w for w in words if r["problem"].startswith(w)), None)
        if body is None or not r["problem"][len(body):].strip():
            continue
        dev = int(r["id"].rsplit("_", 1)[1]) % 5 == 0
        if dev == (side == "dev"):
            pool.setdefault(r["instruction_id_list"][0], []).append((r["problem"][len(body):], r["kwargs"][0]))
    return pool


def excluded(case):
    ex = set(C.set_ids(case, "dev")) | set(C.set_ids(case, "holdout"))
    for f in glob.glob(str(Path(os.environ.get("WSP_CASES", "cases")) / case / "sets" / "*.json")):
        ex |= set(json.load(open(f)))
    return ex


def build(a):
    rng = random.Random(f"{a.side}:{a.seed}")
    pool = sentences("train" if a.side == "train" else "dev")
    types = sorted(pool)
    task = data.load("practice:ifsynth")[0]["task"]            # the practice set's own task: its grader checks these too
    ex = excluded(a.case) if a.side == "train" else None
    own = None if a.side == "train" else set(C.set_ids(a.case, a.side))

    def fresh(spec):
        rows = sorted((r for r in data.load(spec) if (r["id"] not in ex if own is None else r["id"] in own)),
                      key=lambda r: r["id"])
        rng.shuffle(rows)
        return rows
    word, mc, math_, tulu = (fresh(f"practice:{s}") for s in ("gsm8k_train", "mmlupro", "math_train", "tuluif"))
    tulu = [r for r in tulu if RID not in r["instruction_id_list"]]
    near = [t for t in types if together(RID, t) and t != "punctuation:no_comma"]   # a request to repeat may hold commas
    items = []

    def pick(have, k):
        """k more instruction types that may stand next to each other and to the ones in `have`."""
        got = []
        for t in rng.sample(types, len(types)):
            if len(got) == k:
                break
            if all(together(t, h) for h in [*have, *got]):
                got.append(t)
        return got

    def add(kind, text, ids, kws, bodies):
        items.append({"id": f"ifsynth_x{a.side[0]}{a.seed}_{kind}_{len(items):04d}", "source": "ifsynth_x" + kind, "task": task,
                      "problem": text, "instruction_id_list": ids, "kwargs": kws, "answer": None, "kind": kind,
                      "bodies": bodies})

    # rep: bodies of four kinds, so that what has to be copied is one line, several lines, formulas, or a long request
    tnear = [r for r in tulu if r["instruction_id_list"][0] in near]
    n_t = min(len(tnear), a.n_rep * 3 // 10)
    n_w, n_m = (a.n_rep - n_t) * 4 // 9, (a.n_rep - n_t) * 3 // 9
    plan = ([("w", r) for r in word[:n_w]] + [("c", r) for r in mc[:n_m]] + [("m", r) for r in math_[:a.n_rep - n_t - n_w - n_m]]
            + [("t", r) for r in tnear[:n_t]])
    word, tulu = word[n_w:], [r for r in tulu if r not in tnear[:n_t]]
    for sub, r in plan:
        body = r["problem"].strip() if sub in "wt" else data.user_message(r)
        ids, kws = (list(r["instruction_id_list"]), list(r["kwargs"])) if sub == "t" else ([], [])
        if sub != "t" and rng.random() < 1 / 3:
            t = rng.choice(near)
            s, kw = rng.choice(pool[t])
            body, ids, kws = body + s, [t], [kw]
        add("rep" + sub, body + REPEAT, [*ids, RID], [*kws, {"prompt_to_repeat": body}], [r["id"]])
    for r in word[:a.n_word]:
        ts = pick([], rng.choice([2, 3, 3]))
        ss = [rng.choice(pool[t]) for t in ts]
        add("word", r["problem"].strip() + "".join(s for s, _ in ss), ts, [kw for _, kw in ss], [r["id"]])
    for r in tulu[:a.n_tulu]:
        ts = pick(r["instruction_id_list"], rng.choice([1, 1, 2]))
        if not ts:
            continue
        ss = [rng.choice(pool[t]) for t in ts]
        add("tulu", r["problem"].rstrip() + "".join(s for s, _ in ss), [*r["instruction_id_list"], *ts],
            [*r["kwargs"], *(kw for _, kw in ss)], [r["id"]])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--side", choices=["train", "dev", "holdout"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=360); ap.add_argument("--n-word", type=int, default=480)
    ap.add_argument("--n-tulu", type=int, default=480)
    a = ap.parse_args()
    items = build(a)
    json.dump(items, open(a.out, "w"))
    n, ni = {}, {}
    for it in items:
        n[it["kind"]] = n.get(it["kind"], 0) + 1
        ni[len(it["instruction_id_list"])] = ni.get(len(it["instruction_id_list"]), 0) + 1
    print(f"{len(items)} requests -> {a.out}: {n}; instructions per request {dict(sorted(ni.items()))}")


if __name__ == "__main__":
    main()
