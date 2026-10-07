"""Inspect a run without Claude:  python -m harness.status --split test  [--out DIR]

Reads results/<split>/status.json, checks whether the runner and job PIDs are alive, and summarises.
Equivalent manual inspection:  cat results/<split>/status.json ; ps -fp $(cat results/<split>/runner.pid) ;
ls results/<split>/seed*.json | wc -l ; tail results/<split>/logs/seedN.log
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", a.split)
    path = os.path.join(out, "status.json")
    if not os.path.exists(path):
        print(f"no status file at {path}: run has not started")
        return
    s = json.load(open(path))
    jobs = s["jobs"]
    by = {}
    for seed, j in jobs.items():
        by.setdefault(j["state"], []).append(seed)
    runner_alive = alive(s.get("runner_pid"))
    print(f"split={s['split']}  commit={s['git_commit'][:10]}  host={s['host']}")
    print(f"runner PID {s['runner_pid']}: {'ALIVE' if runner_alive else 'not running'}   "
          f"started {s['started']}   last update {s['updated']}   finished {s['finished']}")
    print("  " + "   ".join(f"{k}: {len(v)}" for k, v in sorted(by.items())) + f"   (total {len(jobs)})")
    for seed in by.get("running", []):
        j = jobs[seed]
        print(f"  running seed {seed}: pid {j['pid']} {'alive' if alive(j['pid']) else 'DEAD'} since {j['started']}")
    for seed in by.get("failed", []):
        print(f"  FAILED seed {seed}: rc={jobs[seed].get('returncode')}\n    ..." +
              jobs[seed].get("error_tail", "")[-400:].replace("\n", "\n    "))
    done = [j["seconds"] for j in jobs.values() if j.get("seconds")]
    if done and not runner_alive and by.get("pending"):
        print("  runner is not alive but jobs are pending: rerun the same run_many command to resume")
    elif done and by.get("pending"):
        rem = len(by.get("pending", [])) + len(by.get("running", []))
        print(f"  mean {sum(done) / len(done):.0f}s per seed; ~{rem} seeds left")


if __name__ == "__main__":
    main()
