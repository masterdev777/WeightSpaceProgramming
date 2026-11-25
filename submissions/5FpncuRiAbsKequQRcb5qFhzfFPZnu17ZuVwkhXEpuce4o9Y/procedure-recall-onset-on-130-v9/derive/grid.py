"""The table of a screen: every edit of a case with a paired result on one set, best first.

    python grid.py table CASE SET
        -> each edit's paired result on SET (parent -> edit accuracy, gain +- SE, rescued / lost, cut share)
"""
import argparse
import json
from pathlib import Path


def table(a):
    rows = []
    for p in sorted((Path(a.cases) / a.case / "results").glob(f"*__{a.set}.json")):
        r = json.load(open(p)).get("paired")
        if r:
            rows.append((p.name.split("__")[0], r))
    rows.sort(key=lambda x: -x[1]["gain"])
    print(f"{'edit':26s} {'parent -> edit':>16s} {'gain +- SE':>14s} {'rescued/lost':>13s} {'cut':>15s}")
    for n, r in rows:
        print(f"{n:26s} {100 * r['base_acc']:6.1f} -> {100 * r['edit_acc']:5.1f}  {100 * r['gain']:+5.1f} +- {100 * r['se']:.1f}"
              f"  {r['rescues']:5d}/{r['losses']:<5d}  {100 * r['base_cut']:5.1f} -> {100 * r['edit_cut']:5.1f}")


def main():
    import os
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("case")
    t.add_argument("set")
    a = ap.parse_args()
    a.cases = os.environ.get("WSP_CASES", "cases")
    table(a)


if __name__ == "__main__":
    main()
