"""Where a failed instruction answer first breaks its request: the ONSET (material for onset_branch.py).

  python derive/if_onset.py --rows ANS.jsonl [ANS2.jsonl ...] --out ONSET.jsonl [--sources ifsynth tuluif]

Rows: a model's own answers to practice instruction requests (wsp.gen / shard_gen rows: id, k, source, text, finish,
think_closed, correct). For every answer that finished with its thinking closed and fails its checker (prompt level,
strict: the toolkit's instruction checkers, as wsp.grade), each failing instruction is classed:
  growing  more text can never repair it once broken: no_comma, english_lowercase, english_capital, forbidden_words,
           and the 'less than' forms of number_words, number_sentences, frequency, letter_frequency,
           capital_word_frequency. Onset = the shortest prefix of the visible answer (cut at a word boundary) that
           already breaks it (checked with the instruction's own checker; case checks: the first letter of the
           wrong case, since the language test is not prefix-stable).
  missing  more text could still repair it: the 'at least' forms, existence, postscript, end_checker, placeholders,
           highlighted / multiple sections, title, number_paragraphs. Onset = the start of the answer's last paragraph.
  start    decided by how the answer begins (json_format, repeat_prompt, quotation,
           nth_paragraph_first_word, two_responses, number_bullet_lists): no onset, left out.
The answer's onset is its earliest 'growing' onset if it has one, else its 'missing' onset. Output (one row per answer):
id, k, source, onset_kind, broken (instruction ids), onset_char (in the visible answer), onset_frac, prefix (the full
text up to the onset: thinking + '</think>' + the visible answer before the word that breaks the request), rest (the
original text after the prefix)."""
import argparse
import json
import re
import sys

from wsp import data
from wsp.grade import answer_part

import lm_eval.tasks.ifeval.instructions_registry as REG

GROWING = {"punctuation:no_comma", "change_case:english_lowercase", "change_case:english_capital",
           "keywords:forbidden_words"}
RELATION = {"length_constraints:number_words": "relation", "length_constraints:number_sentences": "relation",
            "keywords:frequency": "relation", "keywords:letter_frequency": "let_relation",
            "change_case:capital_word_frequency": "capital_relation"}
MISSING = {"startend:end_checker", "detectable_content:postscript", "keywords:existence",
           "detectable_content:number_placeholders", "detectable_format:number_highlighted_sections",
           "detectable_format:multiple_sections", "detectable_format:title", "length_constraints:number_paragraphs"}


def checker(iid, kw, prompt):
    ins = REG.INSTRUCTION_DICT[iid](iid)
    kw = {k: v for k, v in (kw or {}).items() if v}
    ins.build_description(**kw)
    args = ins.get_instruction_args()
    if args and "prompt" in args:
        ins.build_description(prompt=prompt)
    return lambda s: bool(s.strip()) and bool(ins.check_following(s))


def klass(iid, kw):
    if iid in GROWING:
        return "growing"
    if iid in RELATION:
        return "growing" if (kw or {}).get(RELATION[iid]) == "less than" else "missing"
    if iid in MISSING:
        return "missing"
    return "start"


def word_cuts(ans):
    """character positions where a word starts (the cut points a prefix may end at)"""
    return [m.start() for m in re.finditer(r"(?<=\s)\S", ans)] + [len(ans)]


def growing_onset(iid, kw, ans, ok):
    """the start of the word whose inclusion first breaks the instruction (None if no prefix breaks it)"""
    if iid == "change_case:english_lowercase":
        m = re.search(r"[A-Z]", ans)
    elif iid == "change_case:english_capital":
        m = re.search(r"[a-z]", ans)
    else:
        m = None
    if iid in ("change_case:english_lowercase", "change_case:english_capital"):
        if not m:
            return None
        cut = ans.rfind(" ", 0, m.start()) + 1
        return max(0, cut)
    cuts = word_cuts(ans)
    # end of each word = next word start; prefixes ans[:end]; find the first end where the check fails
    ends = cuts[1:] if cuts and cuts[0] == 0 else cuts
    lo, hi = 0, len(ends) - 1
    if hi < 0 or ok(ans[: ends[hi]]):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(ans[: ends[mid]]):
            lo = mid + 1
        else:
            hi = mid
    end = ends[lo]
    # the cut = start of the word that ends at 'end'
    start = ans.rfind(" ", 0, max(0, end - 1)) + 1 if end > 0 else 0
    nl = ans.rfind("\n", 0, max(0, end - 1)) + 1
    return max(start, nl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="*", default=[], help="only these sources (default: every row whose item has instructions)")
    ap.add_argument("--min-prefix", type=int, default=1, help="leave out onsets at visible-answer character < this")
    ap.add_argument("--cuts", nargs="+", default=["word"], choices=["word", "sentence"],
                    help="word: just before the breaking word; sentence: the start of its sentence / line (earlier)")
    a = ap.parse_args()
    idx = data.by_id()
    rows = [json.loads(l) for p in a.rows for l in open(p) if l.strip()]
    stats = {"answers": 0, "failed": 0, "growing": 0, "missing": 0, "start_only": 0, "no_onset": 0}
    out = []
    for r in rows:
        if a.sources and r.get("source") not in a.sources or r.get("finish") != "stop" or not r.get("think_closed"):
            continue
        stats["answers"] += 1
        if r.get("correct"):
            continue
        item = idx.get(r["id"])
        text = r["text"]
        e = text.rfind("</think>")          # the answer the request is graded on: after the LAST close
        if item is None or e < 0 or not item.get("instruction_id_list"):
            continue
        stats["failed"] += 1
        head = text[: e + len("</think>")]
        vis = text[e + len("</think>"):]
        lead = len(vis) - len(vis.lstrip())
        ans = vis.strip()
        g_on, m_on, broken = [], [], []
        for iid, kw in zip(item["instruction_id_list"], item["kwargs"]):
            ok = checker(iid, kw, item["problem"])
            if ok(ans):
                continue
            broken.append(iid)
            c = klass(iid, kw)
            if c == "growing":
                o = growing_onset(iid, kw, ans, ok)
                if o is not None:
                    g_on.append(o)
            elif c == "missing":
                p = ans.rfind("\n\n")
                m_on.append(p + 2 if p > 0 else max(0, ans.rfind(". ", 0, len(ans) - 2) + 2))
        if g_on:
            kind, on = "growing", min(g_on)
        elif m_on:
            kind, on = "missing", min(m_on)
        else:
            stats["start_only" if broken else "no_onset"] += 1
            continue
        if on < a.min_prefix:
            stats["no_onset"] += 1
            continue
        stats[kind] += 1
        ons = {}
        if "word" in a.cuts:
            ons["word"] = on
        if "sentence" in a.cuts:
            s0 = max(ans.rfind(". ", 0, on) + 2 if ans.rfind(". ", 0, on) >= 0 else 0, ans.rfind("\n", 0, on) + 1)
            if s0 < on:
                ons["sentence"] = s0
        for ck, o in ons.items():
            if o <= 0:      # a branch point at the very start of the visible answer is no onset: it would only
                continue    # regenerate the whole answer after the thinking
            cut = e + len("</think>") + lead + o
            out.append({"id": r["id"], "k": r.get("k", 0), "source": r["source"], "onset_kind": kind, "cut": ck,
                        "broken": broken, "onset_char": o, "onset_frac": round(o / max(1, len(ans)), 3) + (0.0001 if ck == "sentence" else 0),
                        "prefix": text[:cut], "rest": text[cut:], "text": text, "finish": r.get("finish"),
                        "think_closed": True, "correct": False})
    with open(a.out, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print(f"[onset] {stats} -> {len(out)} onsets -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
