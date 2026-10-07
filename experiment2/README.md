# Experiment 2 — learned, local evidence discrimination

Can the learner learn, from its own experience, which evidence distinguishes which thing — and
revise that when the evidence stops being reliable? The spec and pre-registration are in
[SPEC.md](SPEC.md); results go in [RESULTS.md](RESULTS.md) once the sealed test has run.

Perception, the evidence extractors and the world renderer are copied from Experiment 1
(`../experiment1`, which is not modified).

New in this experiment:

- per-unit learned usefulness, with full change history;
- one shared decision rule, where the variants differ only in the weight source;
- a two-phase world in which location stops being reliable.

## Commands (run from this directory)

```
pip install numpy scipy pytest
python -m pytest tests -q

python -m harness.run_many --split dev        # detached; returns immediately
python -m harness.status   --split dev        # PID alive?, seeds done/running/failed
python -m harness.report   --split dev        # -> results/dev/REPORT.md, summary.json

python -m harness.run_many --split test       # sealed seeds 2000-2029; refuses on uncommitted code
python -m harness.inspect_unit --split test --seed 2000            # mapped units of a seed
python -m harness.inspect_unit --split test --seed 2000 --unit 7   # developmental history of one unit
```

## Checking a run without Claude

- `results/<split>/status.json` holds, for every seed: its state (pending, running, done or
  failed), the job PID, timings and an error tail. The runner's PID is in it and in `runner.pid`.
- Check a process with `ps -fp <pid>`.
- Each seed logs to `logs/seedN.log`.
- Completed seeds are `seedN.json`, written atomically. Re-running the same `run_many` command
  resumes and skips completed seeds.

## Layout

| Path | What |
|---|---|
| `learner/mechanisms/identity.py` | `windowed_auc` (samples, vote, learned usefulness) and `weighted_vote` (shared decision rule) |
| `learner/core.py` | samples with provenance, usefulness history, feedback audit |
| `learner/config.py` | variants A_learned, B_fixed, C_equal, D1_location_only, D2_always_new |
| `harness/scene.py` | six things with unequal evidence reliability; two-phase schedule |
| `harness/run_job.py` | one seed: events, sliding, scoring, oracle discriminability |
| `harness/report.py` | pre-registered metrics and criteria |
