"""The case's sets for this method, written as $WSP_CASES/CASE/sets/NAME.json (each disjoint from the case's other splits):
  recall_pool  1024 maths practice problems (practice:math_train): the forced-recall questions
  math_hold    384 other maths practice problems (none with the text of a recall_pool problem): the maths screen, never fitted
  keepNM       96 gsm8k_train + 128 ifsynth + 64 mmlupro + 95 bbhsynth (5 of each of the 19 families) = 383 problems, no
               maths: the parent's own answers there become keep rows (the edit stays with the parent everywhere but maths)
  keepPZ       24 more generated puzzles of every family (456): more keep rows on puzzles, where the procedure must not act
  puzzle_hold  10 generated puzzles of every family (190): the puzzle check, never fitted or kept
    python recall_sets.py CASE [--seed 7]"""
import argparse
import collections
import json
import os
import random
from pathlib import Path

from wsp import case as C, data

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("case"); ap.add_argument("--seed", default="7")
a = ap.parse_args()
d = Path(os.environ.get("WSP_CASES", "cases")) / a.case / "sets"; d.mkdir(parents=True, exist_ok=True)
rng = random.Random(f"math-recall-sets:{a.seed}")
taken = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
items = {r["id"]: r for r in data._practice("math_train")}
maths = sorted(i for i in items if i not in taken); rng.shuffle(maths)
pool = maths[:1024]
seen = {data.fingerprint(items[i]["problem"]) for i in pool}
hold = [i for i in maths[1024:] if data.fingerprint(items[i]["problem"]) not in seen][:384]
keep = []
for src, n in (("gsm8k_train", 96), ("ifsynth", 128), ("mmlupro", 64)):
    xs = sorted(r["id"] for r in data._practice(src) if r["id"] not in taken); rng.shuffle(xs); keep += xs[:n]
fam = collections.defaultdict(list)
for r in data._practice("bbhsynth"):
    if r["id"] not in taken:
        fam[r["subject"]].append(r["id"])
puz, kb, kpz = [], [], []
for f in sorted(fam):
    xs = sorted(fam[f]); rng.shuffle(xs); puz += xs[:10]; kb += xs[10:15]; kpz += xs[15:39]
keep += kb
for name, ids in (("recall_pool", pool), ("math_hold", hold), ("keepNM", keep), ("keepPZ", kpz), ("puzzle_hold", puz)):
    rng.shuffle(ids); json.dump(ids, open(d / f"{name}.json", "w")); print(f"{name}: {len(ids)}")
