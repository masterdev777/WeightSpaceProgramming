"""Where the model's sampled answers go (recall_gen.py output files): right / finished wrong / cut, by item family, and the
number of problems with mixed outcomes.
    python answer_kinds.py SAMPLES.*.jsonl"""
import collections
import json
import sys

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__); sys.exit(0)

by, fam = collections.defaultdict(list), collections.defaultdict(lambda: collections.Counter())
for f in sys.argv[1:]:
    for line in open(f):
        r = json.loads(line)
        k = "right" if r["correct"] and not r["cut"] else ("cut" if r["cut"] else "wrong")
        by[r["id"]].append(k)
        fam[r["id"].split("_", 1)[1].rsplit("_", 1)[0]][k] += 1
for f, c in sorted(fam.items()):
    n = sum(c.values()); print(f"  {f:42s} right {100 * c['right'] / n:5.1f}  wrong {100 * c['wrong'] / n:5.1f}  cut {100 * c['cut'] / n:5.1f}")
tot = collections.Counter(k for v in by.values() for k in v); n = sum(tot.values())
mixed = sum(1 for v in by.values() if len(set(x == "right" for x in v)) > 1)
print(f"{len(by)} problems, {n} samples: right {100 * tot['right'] / n:.1f}%, wrong {100 * tot['wrong'] / n:.1f}%, cut {100 * tot['cut'] / n:.1f}%; mixed {mixed} ({100 * mixed / max(1, len(by)):.0f}%)")
