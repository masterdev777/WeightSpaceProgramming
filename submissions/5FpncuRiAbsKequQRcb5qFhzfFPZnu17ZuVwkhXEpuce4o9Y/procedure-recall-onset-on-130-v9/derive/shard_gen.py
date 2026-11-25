"""Generate and grade a model's answers on practice problems, spread over several GPUs (one vLLM per GPU).

    python derive/shard_gen.py --model DIR --ids IDS.json --n 4 --salt s0 --out RUN.jsonl [--max-tokens 16384]
                               [--gpus 0,1,2,3]

Each shard is a `python -m wsp.gen` process on one GPU. Per-sample seeds depend only on (problem id, sample index,
salt), so the merged file is exactly the run one process would make, and two models' runs of the same ids, n and
salt are paired (wsp.stats.paired).
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROFILE = "qwen35-0.8b-think"
DONE = re.compile(r"\[gen\] \d+ answers, \d+ tokens in")   # wsp.gen's line once a shard's answers are generated
HANG_S = int(os.environ.get("ONSET_HANG_S", 300))   # a shard whose log says its answers are done but whose output is still missing this long after is hung


def start(cmd, env, log, mode="w"):
    """One shard in its own process group (so a hung vLLM worker can be killed with it). aldiendix's prun.py watchdog,
    credited."""
    return subprocess.Popen(cmd, env=env, stdout=open(log, mode), stderr=subprocess.STDOUT, start_new_session=True)


def watch(p, part, log):
    """Wait for a shard: its return code, or "hung" when it printed its '[gen] N answers' line but wrote no output
    within HANG_S seconds (seen when a vLLM worker does not exit after generating); the hung group is killed."""
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


def gpus_default():
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v:
        return [g for g in v.split(",") if g]
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU")]


def run(model, ids, n, salt, out, gpus, max_tokens=None, profile=PROFILE, gpu_mem=0.90, resume=True):
    """resume: a shard finished by an earlier attempt (same ids, n, salt, cap) is kept, not generated again."""
    out = Path(out)
    tmp = out.parent / f".{out.stem}.shards"
    tmp.mkdir(parents=True, exist_ok=True)
    shards = [ids[i::len(gpus)] for i in range(len(gpus))]
    procs, kept = [], []
    for g, part in zip(gpus, shards):
        if not part:
            continue
        idf, of = tmp / f"ids_{g}.json", tmp / f"run_{g}.jsonl"
        sig = {"ids": part, "n": n, "salt": salt, "max_tokens": max_tokens, "model": str(model)}
        sigf = tmp / f"sig_{g}.json"
        if resume and of.exists() and sigf.exists() and json.loads(sigf.read_text()) == sig:
            kept.append(of)
            continue
        of.unlink(missing_ok=True)
        sigf.write_text(json.dumps(sig))
        idf.write_text(json.dumps(part))
        cmd = [sys.executable, "-m", "wsp.gen", "--profile", profile, "--model", str(model), "--ids", str(idf),
               "--n", str(n), "--salt", salt, "--out", str(of), "--gpu-mem", str(gpu_mem)]
        if max_tokens:
            cmd += ["--max-tokens", str(max_tokens)]
        log = tmp / f"log_{g}.txt"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append((g, of, cmd, env, log, start(cmd, env, log)))
    t0 = time.time()
    bad = []
    for g, of, cmd, env, log, p in procs:
        rc = watch(p, of, log)
        if rc == "hung":   # the shard finished its answers but its worker never handed them back: once more
            print(f"[shard_gen] GPU {g}: answers done but the worker hung at shutdown; killed, running the shard again",
                  flush=True)
            rc = watch(start(cmd, env, log, mode="a"), of, log)
        if rc != 0 or not of.exists():
            bad.append(g)
    if bad:
        raise SystemExit(f"shards on GPUs {bad} failed: see {tmp}/log_<gpu>.txt")
    tmpf = out.with_name(f".{out.name}.tmp")  # atomic: a half-merged file never looks done
    with open(tmpf, "w") as f:
        for of in kept + [p[1] for p in procs]:
            f.write(of.read_text())
    os.replace(tmpf, out)
    rows = sum(1 for _ in open(out))
    print(f"[shard_gen] {rows} answers on {len(procs)} GPUs in {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="a model dir, or with --case a case model: base or an edit name")
    ap.add_argument("--case", help="resolve --model inside this case (base = the starting model, built on first use)")
    ap.add_argument("--ids", required=True, help="json list of practice problem ids")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--salt", default="s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--gpus", help="comma-separated GPU indices (default: all)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()
    ids = json.loads(Path(a.ids).read_text())
    model = a.model
    if a.case:
        from wsp import case as wcase  # noqa: PLC0415
        model = wcase.model_path(a.case, a.model)
    run(model, ids, a.n, a.salt, a.out, a.gpus.split(",") if a.gpus else gpus_default(), a.max_tokens,
        gpu_mem=a.gpu_mem)


if __name__ == "__main__":
    main()
