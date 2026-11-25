"""merge_gains.py OUT_DIR --set K1 FIT1 FIT2 ... --set K2 FITa FITb ...
Merge several preference-gain fits (pref_gains*.py state.pt files with the same layers / heads) into ONE state.pt whose log gains are
sum_j K_j * mean(fits of set j). Gains on the same columns compose multiplicatively, so exporting the merged state once with
`fit_gains.py export --k 1` gives the same model as stacking the sets one after the other, with one payload instead of several."""
import sys
from pathlib import Path
import torch

def main():
    out = Path(sys.argv[1]); args = sys.argv[2:]; sets = []; cur = None
    for x in args:
        if x == "--set": cur = None; continue
        if cur is None: cur = [float(x), []]; sets.append(cur)
        else: cur[1].append(x)
    merged = None
    for k, fits in sets:
        st = [torch.load(Path(f) / "state.pt", weights_only=False) for f in fits]
        meta = st[0]["meta"]
        for s in st[1:]:
            assert s["meta"]["layers"] == meta["layers"], "different layers"
        mlp = {i: k * torch.stack([s["mlp"][i] for s in st]).mean(0) for i in meta["layers"]}
        head = {i: k * torch.stack([s["head"][i] for s in st]).mean(0) for i in st[0]["head"]}
        if merged is None:
            merged = {"mlp": mlp, "head": head, "meta": meta, "args": {"merged_from": [(k, fits) for k, fits in sets]},
                      "n_answers": sum(s.get("n_answers", 0) for s in st), "by_source": {}}
        else:
            assert merged["meta"]["layers"] == meta["layers"] and set(merged["head"]) == set(head), "sets differ in layers / heads"
            for i in mlp: merged["mlp"][i] = merged["mlp"][i] + mlp[i]
            for i in head: merged["head"][i] = merged["head"][i] + head[i]
            merged["n_answers"] += sum(s.get("n_answers", 0) for s in st)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out / "state.pt")
    mx = max(float(t.abs().max()) for t in list(merged["mlp"].values()) + list(merged["head"].values()))
    print(f"merged {len(sets)} sets -> {out / 'state.pt'} (max |log g| {mx:.4f})")

if __name__ == "__main__":
    main()
