"""Run many seeds, detached from the terminal, with on-disk status.

    python -m harness.run_many --split dev      # seeds 0-7
    python -m harness.run_many --split test     # seeds 2000-2029; refuses on uncommitted code
    python -m harness.status --split test       # inspect at any time, no Claude needed

By default this re-launches itself in a new session under `nohup` and returns immediately, so it survives the
terminal / UI closing. Files under results/<split>/:
    runner.pid      PID of the detached runner
    runner.log      runner stdout/stderr
    status.json     per-seed state (pending/running/done/failed), job PIDs, timings, errors (updated atomically)
    logs/seedN.log  each job's own output
    seedN.json      a completed seed's result (written atomically; completed seeds are never rerun)
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

SPLITS = {"dev": list(range(0, 8)), "test": list(range(2000, 2030))}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Status:
    def __init__(self, path, data):
        self.path, self.data, self.lock = path, data, threading.Lock()
        self.write()

    def set(self, seed, **kw):
        with self.lock:
            self.data["jobs"][str(seed)].update(kw)
            self.write()

    def write(self):
        self.data["updated"] = now()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)


def foreground(split, parallel, out, variants):
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    seeds = SPLITS[split]
    st = Status(os.path.join(out, "status.json"), {
        "split": split, "seeds": seeds, "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_paths": git("status", "--porcelain", "--", "learner", "harness", "SPEC.md"),
        "runner_pid": os.getpid(), "host": socket.gethostname(), "started": now(), "finished": None,
        "command": " ".join(sys.argv), "variants": variants,
        "jobs": {str(s): {"state": "done" if os.path.exists(os.path.join(out, f"seed{s}.json")) else "pending"}
                 for s in seeds}})

    def one(s):
        if st.data["jobs"][str(s)]["state"] == "done":
            return
        log = open(os.path.join(out, "logs", f"seed{s}.log"), "w")
        cmd = [sys.executable, "-m", "harness.run_job", "--seed", str(s), "--out", out]
        if variants:
            cmd += ["--variants", variants]
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        st.set(s, state="running", pid=p.pid, started=now())
        rc = p.wait()
        log.close()
        if rc == 0:
            st.set(s, state="done", finished=now(), seconds=round(time.time() - t0))
        else:
            tail = open(os.path.join(out, "logs", f"seed{s}.log")).read()[-2000:]
            st.set(s, state="failed", finished=now(), returncode=rc, error_tail=tail)

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, seeds))
    st.data["finished"] = now()
    st.write()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", default="")
    ap.add_argument("--foreground", action="store_true", help="run in this process (used by the detached launcher)")
    a = ap.parse_args()
    out = os.path.abspath(a.out or os.path.join(ROOT, "results", a.split))
    if a.split == "test":
        dirty = git("status", "--porcelain", "--", "learner", "harness", "SPEC.md")
        if dirty:
            sys.exit(f"Refusing to run the sealed test on uncommitted code:\n{dirty}")
    os.makedirs(out, exist_ok=True)
    if a.foreground:
        foreground(a.split, a.parallel, out, a.variants)
        return
    cmd = [sys.executable, "-m", "harness.run_many", "--split", a.split, "--parallel", str(a.parallel),
           "--out", out, "--foreground"] + (["--variants", a.variants] if a.variants else [])
    log = open(os.path.join(out, "runner.log"), "a")
    # start_new_session=True puts the runner in its own session (setsid); nohup execs in place, so
    # p.pid is the runner's PID and it survives the terminal / UI closing.
    p = subprocess.Popen(["nohup", *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    open(os.path.join(out, "runner.pid"), "w").write(str(p.pid))
    print(f"started detached runner PID {p.pid}\n  status:  python -m harness.status --split {a.split}"
          f"\n  log:     {os.path.join(out, 'runner.log')}")


if __name__ == "__main__":
    main()
