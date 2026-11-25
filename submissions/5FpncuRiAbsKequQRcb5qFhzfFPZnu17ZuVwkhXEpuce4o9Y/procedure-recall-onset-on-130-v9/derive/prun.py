"""`python -m wsp.case run`, split over several GPUs (one vLLM per GPU): same problems, same per-(problem, sample)
seeds (salt s0), same run file and the same paired result, about N times faster on N GPUs.

    python prun.py CASE MODEL SET [--n 2] [--gpus 0,1,2,3] [--force]

MODEL is "base" or an edit of the case (built once before the GPUs start); SET a case set or subset (dev, dev128,
holdout256, ...). Sampling overrides ("model@key=value") are not supported here: use python -m wsp.case run.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


DONE = re.compile(r"\[gen\] \d+ answers, \d+ tokens in")   # wsp.gen's line once a shard's answers are generated
HANG_S = 300   # a shard whose log says its answers are done but whose output is still missing this long after is hung


def start(cmd, env, log, mode="w"):
    """One shard in its own process group (so a hung vLLM worker can be killed with it)."""
    return subprocess.Popen(cmd, env=env, stdout=open(log, mode), stderr=subprocess.STDOUT, start_new_session=True)


def watch(p, part, log):
    """Wait for a shard: its return code, or "hung" when it printed its '[gen] N answers' line but wrote no output
    within HANG_S seconds (seen when a vLLM worker does not exit after generating); the hung group is killed."""
    import signal
    import time
    done_at = None
    while True:
        rc = p.poll()
        if rc is not None:
            return rc
        if done_at is None:
            try:
                if DONE.search(open(log, errors="ignore").read()[-4000:]):
                    done_at = time.time()
            except OSError:
                pass
        elif not part.exists() and time.time() - done_at > HANG_S:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            return "hung"
        time.sleep(10)


def main():
    from wsp import case as C
    from wsp.common import read_jsonl, write_jsonl

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("model")
    ap.add_argument("set")
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    name, sampling = C.split_model(a.model)
    if sampling:
        raise SystemExit("sampling overrides: use python -m wsp.case run")
    tag, items = C.set_items(a.case, a.set)
    out = C.run_path(a.case, a.model, tag)
    if a.force or not out.exists():
        model = C.model_path(a.case, name)
        profile = C.conf(a.case)["profile"]
        gpus = [g for g in a.gpus.split(",") if g != ""]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{a.model}__{tag}.", dir=out.parent))
        jobs = []
        for gi, g in enumerate(gpus):
            chunk = items[gi::len(gpus)]
            if not chunk:
                continue
            ids, part, log = tmp / f"ids{g}.json", tmp / f"part{g}.jsonl", tmp / f"gen{g}.log"
            ids.write_text(json.dumps([r["id"] for r in chunk]))
            cmd = [sys.executable, "-m", "wsp.gen", "--profile", profile, "--model", str(model), "--ids", str(ids),
                   "--n", str(a.n), "--out", str(part)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
            jobs.append((g, part, log, cmd, env, start(cmd, env, log)))
        print(f"[prun] {len(items)} problems x {a.n} on GPUs {','.join(g for g, *_ in jobs)} (logs in {tmp})", flush=True)
        failed = []
        for g, part, log, cmd, env, p in jobs:
            rc = watch(p, part, log)
            if rc == "hung":   # the shard finished its answers but its worker never handed them back: once more
                print(f"[prun] GPU {g}: answers done but the worker hung at shutdown; killed, running the shard again", flush=True)
                rc = watch(start(cmd, env, log, mode="a"), part, log)
            if rc != 0 or not part.exists():
                failed.append(g)
                print(f"[prun] GPU {g} failed:\n" + "".join(open(log).readlines()[-25:]), flush=True)
        if failed:
            raise SystemExit(f"generation failed on GPU(s) {failed}; nothing written to {out}")
        order = {r["id"]: i for i, r in enumerate(items)}
        rows = sorted((r for _, part, *_ in jobs for r in read_jsonl(part)), key=lambda r: (order[r["id"]], r["k"]))
        write_jsonl(out, rows)
    res = C._write_result(a.case, a.model, tag)
    if a.model == C.BASE:  # pair every edit already run on this set, as `wsp.case run base` does
        for p in (C.RUNS / a.case).glob(f"*__{tag}.jsonl"):
            if not p.name.startswith(C.BASE + "__"):
                C._write_result(a.case, p.name.split("__")[0], tag)
    print(json.dumps(res["paired"] or res["summary"], indent=1))


if __name__ == "__main__":
    main()
