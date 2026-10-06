"""Run many jobs in parallel subprocesses.

python -m harness.run_many --split dev   # seeds 0-9
python -m harness.run_many --split test  # seeds 1000-1029, refuses to run on uncommitted code
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SPLITS = {"dev": list(range(0, 10)), "test": list(range(1000, 1030))}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--stages", default="return,moved")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", a.split)
    if a.split == "test":
        dirty = git("status", "--porcelain", "--", "learner", "harness", "PREREGISTRATION.md")
        if dirty:
            sys.exit(f"Refusing to run the sealed test on uncommitted code:\n{dirty}")
    os.makedirs(out, exist_ok=True)
    manifest = {"split": a.split, "seeds": SPLITS[a.split], "stages": a.stages.split(","),
                "git_commit": git("rev-parse", "HEAD"), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    jobs = [(s, st) for st in a.stages.split(",") for s in SPLITS[a.split]]

    def one(job):
        s, st = job
        path = os.path.join(out, f"{st}_seed{s}.json")
        if os.path.exists(path):
            return path, 0.0
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "harness.run_job", "--seed", str(s), "--stage", st,
                            "--out", out], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-3000:])
        print(f"done {st} seed {s} in {time.time() - t0:.0f}s", flush=True)
        return path, time.time() - t0

    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(one, jobs))
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
