"""Sample a model on a file of recombined requests (derive/mkreq.py) and grade every answer with the toolkit's checker.

  python derive/genreq.py --case CASE --model base|EDIT --items REQ.json --out RUN.jsonl [--n 4] [--salt s0]
  python derive/genreq.py --items REQ.json --table RUN.jsonl [RUN2.jsonl ...]     # follow rates per kind and per type

Rows are the toolkit's own (wsp.gen): id, k, source, n_tokens, text, correct, cut, pred (one result per instruction).
"""
import argparse, json


def table(runs, items, label=None):
    idx = {it["id"]: it for it in items}
    rows = [json.loads(l) for f in runs for l in open(f)]
    rows = [r for r in rows if r["id"] in idx]

    def line(name, rs):
        by = {}
        for r in rs:
            by.setdefault(r["id"], []).append(bool(r["correct"]) and not r["cut"])
        mixed = sum(0 < sum(v) < len(v) for v in by.values())
        p = sum(map(sum, by.values())) / max(1, len(rs))
        print(f"  {name:44s} {len(by):5d} requests {len(rs):6d} answers  followed {100 * p:5.1f}%  mixed {mixed:4d}  "
              f"cut {100 * sum(r['cut'] for r in rs) / max(1, len(rs)):4.1f}%  tokens {sum(r['n_tokens'] for r in rs) / max(1, len(rs)):6.0f}")
    print(f"{label or ' '.join(runs)}:")
    line("ALL", rows)
    for k in sorted({idx[r["id"]]["kind"] for r in rows}):
        line("kind " + k, [r for r in rows if idx[r["id"]]["kind"] == k])
    for n in sorted({len(idx[r["id"]]["instruction_id_list"]) for r in rows}):
        line(f"{n} instruction(s)", [r for r in rows if len(idx[r["id"]]["instruction_id_list"]) == n])
    per = {}
    for r in rows:
        if not isinstance(r.get("pred"), list):
            continue
        for t, ok in zip(idx[r["id"]]["instruction_id_list"], r["pred"]):
            a = per.setdefault(t, [0, 0]); a[0] += bool(ok); a[1] += 1
    print("  per instruction (answers that closed their thinking):")
    for t, (k, n) in sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"    {t:48s} {100 * k / n:5.1f}%  of {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case"); ap.add_argument("--model", default="base"); ap.add_argument("--items", required=True)
    ap.add_argument("--out"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--salt", default="s0")
    ap.add_argument("--max-tokens", type=int, default=12000); ap.add_argument("--table", nargs="*")
    a = ap.parse_args()
    items = json.load(open(a.items))
    if a.table:
        table(a.table, items)
        return
    from wsp import case as C, profiles
    from wsp.gen import generate
    generate(C.model_path(a.case, a.model), items, profiles.get(C.conf(a.case)["profile"]), n=a.n, out=a.out,
             salt=a.salt, max_tokens=a.max_tokens)
    table([a.out], items, label=f"{a.case} {a.model}")
    print("GENREQ_DONE", flush=True)


if __name__ == "__main__":
    main()
