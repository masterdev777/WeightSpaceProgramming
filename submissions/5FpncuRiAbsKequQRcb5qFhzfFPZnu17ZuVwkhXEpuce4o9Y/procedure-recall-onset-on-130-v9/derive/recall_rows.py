"""Training rows for procedure recall (units_fit.py format).

fit rows : the model's own right forced-recall answers (recall_gen.py on recall_pool), placed after the PLAIN question:
           the edit has to produce, unprompted, the thinking the procedure sentence started. First --tokens tokens of the
           answer (no end-of-turn when cut there). Up to --per answers per problem; a share --held of the problems is
           held out to measure the fit.
keep rows: the parent's own answers to other problems (prun.py runs of keepNM / keepPZ), as "keep" rows (KL to the
           unedited model): the edit stays with the parent there.
    python recall_rows.py CASE RECALL.jsonl OUT_PREFIX --keep RUN [RUN ...] [--keep-w 3] [--tokens 160] [--per 2] [--held 0.1]
           [--drop-mentions]
-> OUT_PREFIX.train.jsonl, OUT_PREFIX.held.jsonl"""
import argparse
import collections
import json
import random
import re

from transformers import AutoTokenizer

from wsp import case as C, data, profiles

ap = argparse.ArgumentParser()
ap.add_argument("case"); ap.add_argument("fit"); ap.add_argument("out")
ap.add_argument("--keep", nargs="*", default=[])
ap.add_argument("--fams")
ap.add_argument("--tokens", type=int, default=1536)
ap.add_argument("--per", type=int, default=2)
ap.add_argument("--held", type=float, default=0.1)
ap.add_argument("--keep-exclude", default="", help="families whose keep answers are left out (comma list)")
ap.add_argument("--keep-max", type=int, default=0, help="at most this many keep rows (0 = all)")
ap.add_argument("--keep-w", type=float, default=1.0, help="weight of every keep row (the KL term), default 1")
ap.add_argument("--drop-mentions", action="store_true", help="drop answers that refer to a hint / note / given fact")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

prof = profiles.get(C.conf(a.case)["profile"])
tok = AutoTokenizer.from_pretrained(str(C.model_path(a.case, "base")))
idx = data.by_id()
rng = random.Random(a.seed)
MENTION = re.compile(r"useful (fact|approach)|\bhint\b|the note\b|as (given|stated|noted) (in|above)|the (provided|given) (fact|rule|approach)", re.I)


def prompt0(item):
    return tok.apply_chat_template([{"role": "user", "content": data.user_message(item)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=prof["thinking"])


def row(r, mode=None):
    ids = tok.encode(r["text"], add_special_tokens=False)
    cut = len(ids) > a.tokens
    out = {"id": r["id"], "prompt": prompt0(idx[r["id"]]), "target": tok.decode(ids[: a.tokens]) if cut else r["text"],
           "w": 1.0, "no_eot": cut}
    if mode:
        out["mode"] = mode
    return out


fams = set(a.fams.split(",")) if a.fams else None
by = collections.defaultdict(list)
dropped = collections.Counter()
for line in open(a.fit):
    r = json.loads(line)
    fam = idx[r["id"]].get("subject")
    if fams and fam not in fams:
        continue
    if not r["correct"] or r["cut"]:
        continue
    if a.drop_mentions and MENTION.search(r["text"]):
        dropped[fam] += 1
        continue
    by[r["id"]].append(r)
ids = sorted(by)
rng.shuffle(ids)
nh = int(round(a.held * len(ids)))
held_ids = set(ids[:nh])
train, held = [], []
per_fam = collections.Counter()
for i in ids:
    for r in rng.sample(by[i], min(a.per, len(by[i]))):
        (held if i in held_ids else train).append(row(r))
        per_fam[idx[i].get("subject", idx[i].get("source"))] += i not in held_ids
keep = []
kx = set(filter(None, a.keep_exclude.split(",")))
for f in a.keep:
    keep += [row(r, "keep") for r in map(json.loads, open(f)) if idx[r["id"]].get("subject") not in kx]
rng.shuffle(keep)
if a.keep_max:
    keep = keep[: a.keep_max]
for r in keep:
    r["w"] = a.keep_w
train += keep
rng.shuffle(train)
for name, rows in (("train", train), ("held", held)):
    with open(f"{a.out}.{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
print(f"{len(train) - len(keep)} fit rows ({len(ids) - nh} problems), {len(keep)} keep rows, {len(held)} held-out rows")
print("fit rows per family:", dict(sorted(per_fam.items())))
if dropped:
    print("dropped (mentions a hint):", dict(dropped))
