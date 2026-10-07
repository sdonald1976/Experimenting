# Experiment 3 — long-span continuity as the teacher

Spec and pre-registration: [SPEC.md](SPEC.md). Results, once the sealed test has run:
[RESULTS.md](RESULTS.md).

The code is copied from `../experiment2`, which is not modified. The only learner change is the
`continuity_span` usefulness mechanism in `learner/mechanisms/identity.py`: which continuity
comparisons produce "same thing" samples.

The world adds 30 ticks of continuous change in view to each phase-2 observation window: sliding,
rotation, and the camera moving and approaching.

## Commands (from this directory)

```
python -m pytest tests -q
python -m harness.run_many --split dev     # detached; seeds 0-7
python -m harness.status   --split dev     # no Claude needed: PID alive?, seeds done/running/failed
python -m harness.report   --split dev
python -m harness.run_many --split test    # sealed seeds 3000-3029; refuses on uncommitted code
python -m harness.inspect_unit --split test --seed 3000 --variant B_long             # mapped units
python -m harness.inspect_unit --split test --seed 3000 --variant B_long --unit 3    # history of one unit
```

To check a run by hand:

- read `results/<split>/status.json`;
- check the runner with `ps -fp $(cat results/<split>/runner.pid)`;
- each seed logs to `logs/seedN.log`;
- completed seeds are `seedN.json`, written atomically and never rerun.
