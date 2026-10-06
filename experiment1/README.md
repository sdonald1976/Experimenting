# Experiment 1 — can persistent identity emerge from stereo experience?

Smallest executable version of the Experiment #1 specification.

A learner receives only two camera images per tick, a noisy sense of its own movement, and
its own sensor calibration. It is never given object labels, IDs, masks, depth or
positions. It has to:

1. discover distinct persistent things;
2. accumulate experiences of them;
3. after they have been out of view, recognise them again — or honestly say it doesn't know.

## Result

**Sealed test: fails the pre-registered criteria** — narrowly on specificity. Read
[RESULTS.md](RESULTS.md).

- Uncontaminated units: yes.
- Things that stayed put: 74% re-identified with 0.3% false claims, against a location-only
  shortcut's 72% correct and 15% false.
- New or replaced things wrongly linked: 5 of 177 (limit ≤ 5% at 95% confidence; bound 6.1%).

## Layout

| Path | Side | What |
|---|---|---|
| `harness/render.py`, `harness/scene.py` | harness (knows ground truth) | ray-cast stereo world, things, camera route, trial schedule |
| `harness/run_job.py` | harness | runs one seed: feeds learners through pipes, scores, audits |
| `harness/scoring.py`, `harness/report.py` | harness | outcomes, metrics, bootstrap CIs, pre-registered criteria |
| `learner/core.py` | learner | units, experience records, provenance, identity revision |
| `learner/stores.py` | learner | observation store (immutable) and inference store (provenance, epistemic state) |
| `learner/mechanisms/*` | learner | every EXPERIMENTAL mechanism, registered and replaceable |
| `learner/config.py` | learner | main learner, ablation, and trivial baselines |
| `ASSUMPTIONS.md` | — | every decision and assumption |
| `MECHANISMS.md` | — | generated list of mechanisms, parameters and assumptions |
| `PREREGISTRATION.md` | — | criteria, fixed before the sealed test |
| `results/` | — | raw per-seed JSON, `summary.json`, `REPORT.md` |

## Run

```
pip install numpy scipy pytest
python -m pytest tests -q
python -m harness.run_many --split dev      # seeds 0-9
python -m harness.report --split dev
python -m harness.run_many --split test     # seeds 1000-1029; refuses on uncommitted code
python -m harness.report --split test
```

To try an alternative mechanism:

1. Register a new class in `learner/mechanisms/` with the same slot.
2. Point a variant at it in `learner/config.py`.

Its name, version, parameters and assumptions are then logged with every result.
