# Experiment 2 — learned, local evidence discrimination

This file is the minimum specification and the pre-registration in one. It is committed before
any sealed result exists. The test runner refuses to run on uncommitted changes to `learner/`,
`harness/` or this file.

Experiment 1's sealed result stands as evidence. Its code, results and test seeds
(1000–1029) are not used here.

## Question

Can the learner acquire, from its own legitimate experience, *which evidence distinguishes
which persistent thing*, such that:

- its learned preferences match what its experience actually made discriminating;
- those preferences can differ per unit;
- they are revised when an evidence source stops being reliable?

## Learner variants (the only thing that differs between them is how evidence kinds are weighted)

All variants share the following, copied from Experiment 1 unchanged:

- perception: stereo, candidates, continuity, sampling;
- the four semantically neutral evidence extractors: spatial extent, chromaticity
  distribution, high-pass local patterns, location in the self-motion frame.

They also share the same per-kind **vote** and the same **decision rule** (both below).

| ID | Variant | Weight of evidence kind *k* when judging unit *u* |
|---|---|---|
| **A** | `learned` | **w(u,k) learned from u's own experience** (below); starts at 0 = uncommitted |
| **B** | `fixed` | predetermined, same for every unit, never changes: patterns 0.35, chroma 0.30, extent 0.20, location 0.15 — our prior guess, written down before any Experiment 2 data |
| **C** | `equal` | 1 for every kind |
| **D1** | `location_only` | shortcut: SAME as the nearest recorded location (exposes location leakage) |
| **D2** | `always_new` | shortcut: never links (exposes "do nothing" safety) |

### Shared vote (per unit, per kind) — EXPERIMENTAL

Each unit keeps two sample sets per evidence kind. Every sample is stored with its tick and the
record ids that produced it.

- **W (same):** the distance from a new continuity record to the unit's best-matching record
  ≥ 9 ticks older in the same uninterrupted continuity segment.
- **B (different):** the distance from a record of another unit observed on the same tick to
  this unit's best-matching earlier record.

A query at distance *d* gets vote

v = P̂(W ≥ d) − P̂(B ≤ d)

with pseudo-count 1. v lies in (−1, 1) and is exactly 0 with no samples. Votes and usefulness
use only the most recent 30 W and 60 B samples. This **recency window** is the assumption that
allows revision. The full history is kept.

### A's learned usefulness — EXPERIMENTAL

w(u,k) = shrink · max(0, 2·AUC − 1), where:

- AUC = P̂(w < b) between the windowed W samples and the hardest 25% of the windowed B samples
  ("can this kind separate u from the things most similar to it?");
- shrink = n/(n+10), with n = min(#W, #B).

Every change of w(u,k) is logged with tick, old value, new value, sample counts, and the sample
(type, distance, record ids) that caused it.

### Shared decision rule — EXPERIMENTAL; identical for A, B and C

1. S = Σ w·v / Σ w over the kinds available.
2. SAME if S ≥ 0.5 **and** S stays ≥ 0.5 with any one kind removed. NEW mirrors this at −0.5.
   Otherwise UNKNOWN.
3. For A, Σ w < 0.5 means there is no learned basis yet, so the result is UNKNOWN.

The leave-one-out condition is the architectural rule "no single kind defines identity" (Q6).
It applies to all variants identically. No rule says any kind is more or less important; in A,
that comes only from w(u,k).

Revision of young units uses the same rule: each record is scored, then the per-kind median is
taken (as in Experiment 1).

### Feedback path

**None.** W samples come only from continuity-linked records within one segment, and B samples
only from co-observation. Recognition- or revision-derived identity never creates a usefulness
sample; a recognition link starts a new segment. This is tested in `tests/`.

## World (harness only; the learner is never told anything about it)

- **Things:** 6 target things with deliberately unequal evidence reliability:
  - three share one colour;
  - two share another colour;
  - several share a size;
  - half are elongated boxes or cylinders, whose visible extent varies with viewpoint;
  - half are spheres or cubes, whose visible extent is nearly constant.
  - Every thing carries its own random marks.
- **Phase 1 (location reliable):** exploration arc, then 4 look-away/return cycles. Nothing moves.
- **Phase 2 (location becomes unreliable):** 6 cycles with 20-tick observation windows. The
  identity evaluation happens at observation tick 10, as in Experiment 1. Then 1–2 visible things
  **slide in plain view** (2.5 cm/tick for 10 ticks). While the camera is away:
  - 2 cycles swap two things' locations;
  - 1 cycle moves a thing;
  - 1 cycle introduces a novel thing;
  - 1 cycle replaces a thing with a different one in the same place;
  - 1 cycle changes nothing.
- **No event marker** reaches the learner. Its input is identical in kind to Experiment 1:
  stereo images, noisy self-motion, calibration.

## Seeds

- **Development:** 0–7.
- **Sealed test:** 2000–2029, run once.
- **Unit of replication:** the seed. All intervals are 95% cluster-bootstrap over seeds (4000
  resamples). Comparisons between variants are paired by seed.

## Hypotheses, metrics, falsification

Oracle discriminability is computed by the harness **after** the run, from ground truth applied
to the learner's own records; the learner never sees it. For a unit u mapped to physical thing g
(≥ 80% of its records on g, ≥ 10 records), it is the AUC with which kind-k distance separates:

- u's records vs. other records of g (≥ 9 ticks apart), from
- u's records vs. records of other target things.

| | Hypothesis | Metric | Falsified if |
|---|---|---|---|
| **H2a** | A's learned usefulness corresponds to what its experience made discriminating | per-seed Spearman ρ over (unit, kind) pairs between final w(u,k) and oracle AUC(u,k); mean over seeds | CI lower bound ≤ 0, **or** not above the same ρ computed with B's fixed weights (paired difference CI lower bound ≤ 0) |
| **H2a-local** | usefulness is local: differences between units in w(u,k) track differences in the oracle | per seed, per kind: Spearman ρ across units; averaged over kinds, then over seeds | CI lower bound ≤ 0 |
| **H2b** | A revises location usefulness when location stops predicting identity | Δ = w(u, location) at end − at start of phase 2. Difference-in-differences: mean Δ for units whose thing slid while observed − mean Δ for units whose thing never moved | CI upper bound ≥ 0 |
| **H2c** | learned weighting makes identity claims safer than equal weighting without giving up re-identification | (i) false-SAME rate over all phase-2 identity trials plus novel / same-place-different trials: A − C; (ii) correct-SAME rate over eligible reappearance trials (both phases): A − C | (i) CI upper bound ≥ 0, **or** (ii) CI lower bound < −0.10 |
| **H2c-B** | same comparison, A vs. fixed weights B | same, A − B | same |

Also reported, not pass/fail:

- all outcome rates per condition and variant;
- D1/D2 shortcut results;
- swap and moved trials separately;
- learned usefulness by kind;
- example developmental histories;
- integrity audits:
  - observations unaltered;
  - provenance present;
  - locality;
  - no canary or names in learner state;
  - no recognition-derived usefulness samples.

Outcome definitions, trial eligibility and the viewpoint tolerance are exactly Experiment 1's:

- **Outcomes:** correct SAME, false SAME, NEW, UNKNOWN, not detected. UNKNOWN is never an error.
- **Eligibility:** unobserved gap, ≥ 150 px visible, experienced viewpoint within 15°.

## Process

| Command | What it does |
|---|---|
| `python -m harness.run_many --split dev\|test` | starts work detached (`setsid nohup`) |
| `python -m harness.status --split test` | shows PID, alive or not, jobs done / running / failed; reads `results/<split>/status.json` and does not need Claude |

Each seed's result is written to its own file as it completes, and completed seeds are never
rerun.

## Changes after development

(Listed here before the sealed run. Empty means none.)
