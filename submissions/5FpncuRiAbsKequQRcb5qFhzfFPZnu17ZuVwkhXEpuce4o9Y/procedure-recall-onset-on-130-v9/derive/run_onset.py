"""run_onset.py: the violation-onset layer as one detached run (METHOD.md, section 4).

  nohup python derive/run_onset.py --gpus 0,1,2,3 [--case k] [--keep-w W] [--k K] [--req-sets N] [--no-upload] [--no-build] > $WSP_WORK/onset_run.out 2>&1 &

The case (default k) is already initialised on the start (`python -m wsp.case init ...`). Every step logs one line to
$WSP_WORK/onset/status.log and keeps its outputs in $WSP_WORK/onset/, so a restarted run skips what is done (finished
continuation shards and finished fits included). A vLLM job (genreq, continuations; the answer shards in shard_gen.py)
whose output is written but whose worker hangs at shutdown is killed after 300 s and counts as finished. Steps:
  1  the start's answers to 1,920 practice instruction requests (ifsynth + tuluif outside dev / holdout) x4, 12,000 tokens
  1b with --keep-w > 0: the start's answers to 480 practice reasoning puzzles (bbhsynth) x2, 8,000 tokens (keep rows)
  2  #94's recombined requests (mkreq seeds 0 to N-1, --req-sets N, default 3) x2
  3  onsets: derive/if_onset.py (the request's checker on growing prefixes of the visible answer; word and sentence cuts)
  4  K continuations (--k, default 8) from every branch point: derive/onset_branch.py (one share of the onsets per GPU)
  5  8 preference fits (#107's pref_gains2.py, seeds 201-208, rounds of one per GPU; with --keep-w the keep rows are
     held at the start), log-mean merge
  6  export at the fixed strength 1.0 and build; the calling driver (run_a.py) packages, checks and uploads
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--case", default="k")
    ap.add_argument("--keep-w", type=float, default=0.0, help="weight of the KL keep on the start's own puzzle answers")
    ap.add_argument("--no-upload", action="store_true", help="stop after the build (always: the calling driver uploads)")
    ap.add_argument("--smoke", action="store_true", help="a ~10 min check of every step at tiny sizes (no upload)")
    ap.add_argument("--k", type=int, default=8, help="continuations per branch point")
    ap.add_argument("--req-sets", type=int, default=3, help="sets of recombined requests (mkreq seeds 0..N-1)")
    ap.add_argument("--no-build", action="store_true", help="export only (a driver that packages the spec itself)")
    a = ap.parse_args()
    if a.smoke:
        a.no_upload = True
    N_IDS, N_KEEP, K, EPOCHS, SEEDS = (32, 16, 2, 1, [201, 202]) if a.smoke else (1920, 480, a.k, 2, list(range(201, 209)))
    RS = list(range(1 if a.smoke else a.req_sets))
    MKREQ = ["--n-rep", 12, "--n-word", 16, "--n-tulu", 16] if a.smoke else []
    gl = a.gpus.split(",")
    n = len(gl)
    W = Path(os.environ["WSP_WORK"]) / "onset"
    (W / "fits").mkdir(parents=True, exist_ok=True)
    (W / "req").mkdir(parents=True, exist_ok=True)
    L = W / "status.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}"
        print(line, flush=True)
        with open(L, "a") as f:
            f.write(line + "\n")

    def fail(msg):
        log(f"FAILED: {msg}")
        sys.exit(1)

    HANG_S = int(os.environ.get("ONSET_HANG_S", 300))    # a job whose done line is printed but which is still alive this long after is hung at shutdown

    def killtree(pid):
        """SIGKILL a job and every process under it, deepest first (the jobs stay in this script's process group, so a
        driver that stops this script's group still stops them)"""
        def kids(q):
            out = []
            for t in os.listdir(f"/proc/{q}/task") if os.path.isdir(f"/proc/{q}/task") else []:
                try:
                    out += [int(c) for c in open(f"/proc/{q}/task/{t}/children").read().split()]
                except OSError:
                    pass
            return out
        order, todo = [], [pid]
        while todo:
            q = todo.pop()
            order.append(q)
            todo += kids(q)
        for q in reversed(order):
            try:
                os.kill(q, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def wait_all(ps, logs, done_re):
        """return codes of the jobs; a job whose log shows done_re (printed after its output is written) and which is
        still alive HANG_S s later had its vLLM worker hang at shutdown: its group is killed and it counts as finished
        (aldiendix's watchdog idea, credited)"""
        rcs, seen = [None] * len(ps), [None] * len(ps)
        while any(rc is None for rc in rcs):
            for i, p in enumerate(ps):
                if rcs[i] is not None:
                    continue
                rcs[i] = p.poll()
                if rcs[i] is not None or not done_re:
                    continue
                if seen[i] is None:
                    try:
                        if re.search(done_re, open(logs[i], errors="ignore").read()[-6000:], flags=re.M):
                            seen[i] = time.time()
                    except OSError:
                        pass
                elif time.time() - seen[i] > HANG_S:
                    log(f"a job printed its done line {HANG_S} s ago but did not exit: killed ({logs[i]})")
                    killtree(p.pid)
                    p.wait()
                    rcs[i] = 0
            if any(rc is None for rc in rcs):
                time.sleep(10)
        return rcs

    def run(args, out=None, env=None, ok=None, done_re=None):
        """one command; its output to the file `out` (or the status log); fails the run if it fails"""
        with open(out or L, "a" if out is None else "w") as f:
            p = subprocess.Popen([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT, env=env)
            rc = wait_all([p], [out or L], done_re)[0]
        if rc:
            fail(ok or " ".join(map(str, args[:3])))

    def par(jobs, what, done=None, done_re=None):
        """jobs = [(args, out file, gpu)], run together, one per GPU; done(i) is called for every job that ended well
        (before a failure stops the run), so a restarted run keeps the finished ones"""
        ps = []
        for args, out, g in jobs:
            f = open(out, "w")
            ps.append((subprocess.Popen([str(x) for x in args], stdout=f, stderr=subprocess.STDOUT,
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": g}), f))
        bad = wait_all([p for p, _ in ps], [out for _, out, _ in jobs], done_re)
        for _, f in ps:
            f.close()
        for i, b in enumerate(bad):
            if not b and done:
                done(i)
        if any(bad):
            fail(what)

    def lines(p):
        return sum(1 for _ in open(p)) if Path(p).exists() else 0

    from wsp import case as C, data
    mp = Path(C.model_path(a.case, "base"))
    if not mp.is_dir():
        fail("start model")
    os.environ["REQ_ITEMS"] = str(W / "req" / "all.json")
    log(f"start model {mp}; GPUs {a.gpus}; keep-w {a.keep_w}")

    # 1. the start's answers to practice instruction requests outside dev / holdout (a fixed amount of work)
    if not (W / "ids.json").exists():
        ex = set(C.set_ids(a.case, "dev")) | set(C.set_ids(a.case, "holdout"))
        per = {}                         # an equal share from each practice set (fixed shuffle per set), whatever their sizes
        for s in ("ifsynth", "tuluif"):
            per[s] = sorted(x["id"] for x in data.load(f"practice:{s}") if x["id"] not in ex)
            random.Random(2107).shuffle(per[s])
        half = N_IDS // 2
        take = {s: per[s][:half] for s in per}
        for s, o in (("ifsynth", "tuluif"), ("tuluif", "ifsynth")):   # a short set is filled from the other one
            if len(take[s]) < half:
                take[o] = per[o][:N_IDS - len(take[s])]
        ids = take["ifsynth"] + take["tuluif"]
        random.Random(2107).shuffle(ids)
        json.dump(ids, open(W / "ids.json", "w"))
    if not (W / "ans.jsonl").exists():
        run([PY, D / "shard_gen.py", "--model", mp, "--ids", W / "ids.json", "--n", 4, "--salt", "onA", "--out",
             W / "ans.jsonl", "--max-tokens", 12000, "--gpus", a.gpus], out=W / "ans.log", ok="answers")
    log(f"1 answers: {lines(W / 'ans.jsonl')}")
    if a.keep_w > 0:
        if not (W / "keep_ids.json").exists():
            ids = sorted(x["id"] for x in data.load("practice:bbhsynth"))
            random.Random(125).shuffle(ids)
            json.dump(ids[:N_KEEP], open(W / "keep_ids.json", "w"))
        if not (W / "keep_bb.jsonl").exists():
            run([PY, D / "shard_gen.py", "--model", mp, "--ids", W / "keep_ids.json", "--n", 2, "--salt", "onK", "--out",
                 W / "keep_bb.jsonl", "--max-tokens", 8000, "--gpus", a.gpus], out=W / "keep.log", ok="keep answers")
        log(f"1b keep answers (practice puzzles): {lines(W / 'keep_bb.jsonl')}")

    # 2. recombined requests (#94's mkreq / genreq, credited)
    if not (W / "req" / "all.json").exists():
        for s in RS:
            run([PY, D / "mkreq.py", "--case", a.case, "--out", W / "req" / f"r{s}.json", "--seed", s, *MKREQ], ok="mkreq")
        allr = [x for s in RS for x in json.load(open(W / "req" / f"r{s}.json"))]
        json.dump(allr, open(W / "req" / "all.json", "w"))
    if not (W / "req" / "all.base.jsonl").exists():
        run([PY, D / "genreq.py", "--case", a.case, "--model", "base", "--items", W / "req" / "all.json", "--out",
             W / "req" / "all.base.jsonl", "--n", 2, "--salt", "onR"], out=W / "genreq.log",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": a.gpus}, ok="genreq", done_re=r"^GENREQ_DONE")
    log(f"2 recombined answers: {lines(W / 'req' / 'all.base.jsonl')}")

    # 3. onsets
    if not (W / "onset.jsonl").exists():
        run([PY, D / "with_req.py", D / "if_onset.py", "--rows", W / "ans.jsonl", W / "req" / "all.base.jsonl",
             "--out", W / "onset.jsonl", "--cuts", "word", "sentence"], ok="onsets")
    if not lines(W / "onset.jsonl"):
        fail("onsets")
    log(f"3 onsets: {lines(W / 'onset.jsonl')}")

    # 4. continuations from every onset, one share per GPU
    if not (W / "br.jsonl").exists():
        rows = open(W / "onset.jsonl").read().splitlines()
        parts = json.load(open(W / "br_parts.json")) if (W / "br_parts.json").exists() else n   # a restart keeps the split
        for g in range(parts):
            if not (W / f"br_{g}.done").exists():
                part = rows[g::parts]
                open(W / f"onset_{g}.jsonl", "w").write("\n".join(part) + ("\n" if part else ""))
        json.dump(parts, open(W / "br_parts.json", "w"))
        todo = [g for g in range(parts) if not (W / f"br_{g}.done").exists()]   # shards already continued are kept
        for i in range(0, len(todo), n):
            batch = todo[i:i + n]
            par([([PY, D / "with_req.py", D / "onset_branch.py", "--model", mp, "--rows", W / f"onset_{g}.jsonl",
                   "--out", W / f"br_{g}.jsonl", "--kinds", "onset", "--k", K, "--if-max-new", 4000, "--salt", "onb"],
                  W / f"br_{g}.log", gl[j]) for j, g in enumerate(batch)], "continuations",
                done=lambda j, batch=batch: open(W / f"br_{batch[j]}.done", "w").close(),
                done_re=r"continuations right; answers rescued|^\[branch\] 0 branch points")
        with open(W / "br.jsonl", "w") as f:
            for g in range(parts):
                f.write(open(W / f"br_{g}.jsonl").read())
    log(f"4 continuation rows: {lines(W / 'br.jsonl')}")

    # 5. eight fits (rounds of one per GPU), then the log mean
    if not (W / "fits" / "ONm" / "state.pt").exists():
        seeds = SEEDS
        keep = ["--keep", W / "keep_bb.jsonl", "--keep-w", a.keep_w] if a.keep_w > 0 else []
        def fitted(s):          # a fit that saved its state (a restarted run does not fit it again)
            lg = W / f"fit_{s}.log"
            return (W / "fits" / f"ON{s}" / "state.pt").exists() and lg.exists() and "[pref] saved" in lg.read_text()
        todo = [s for s in seeds if not fitted(s)]
        for i in range(0, len(todo), n):
            par([([PY, D / "with_req.py", D / "pref_gains2.py", "--model", mp, "--branches", W / "br.jsonl", "--out",
                   W / "fits" / f"ON{s}", "--layers", "4-21", "--heads", "--beta", 0.1, "--nll-w", 0.2,
                   "--max-pairs", 2, "--bad-cut", "--epochs", EPOCHS, *keep, "--seed", s], W / f"fit_{s}.log", gl[j])
                 for j, s in enumerate(todo[i:i + n])], "fits")
        pairs = next((ln for ln in open(W / f"fit_{seeds[0]}.log") if ln.startswith("[pref]") and "pairs" in ln), "")
        log(pairs.strip())
        run([PY, D / "merge_gains.py", W / "fits" / "ONm", "--set", 1.0, *[W / "fits" / f"ON{s}" for s in seeds]],
            ok="merge")
    log("5 fits merged")

    # 6. export at the fixed strength 1.0 and build (METHOD.md section 4, step 5)
    name = "ON1p0"
    edits = Path(os.environ["WSP_CASES"]) / a.case / "edits"
    if not (edits / f"{name}.json").exists():
        run([PY, D / "fit_gains.py", "export", "--fits", W / "fits" / "ONm", "--k", 1.0, "--name", name, "--spec-dir",
             edits, "--parent", "../" + C.conf(a.case)["parent"]], ok="export 1.0")
    if a.no_build:
        log(f"6 exported {name} (not built here: --no-build)")
    else:
        C.model_path(a.case, name)
        log(f"6 exported and built {name}")
    log("DONE (exported; the calling driver packages, checks and uploads it)")


if __name__ == "__main__":
    main()
