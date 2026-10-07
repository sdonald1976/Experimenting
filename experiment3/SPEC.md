# Experiment 3 — does long-span continuity teach which evidence identifies a thing?

This file is the minimum specification and the pre-registration. It is committed before any sealed
result exists. The sealed runner refuses uncommitted changes to `learner/`, `harness/` or this file.

Experiments 1 and 2 are not modified; their results stand. Code was copied from Experiment 2.

## What changes from Experiment 2 (and nothing else in the learner)

**Only which continuity comparisons produce "same thing" (W) samples.** Everything else is identical
to Experiment 2:

- perception;
- the four evidence kinds;
- "different thing" (B) samples;
- windows (30 W / 60 B);
- vote;
- learned-usefulness formula;
- decision rule;
- revision.

| W sample source | Rule |
|---|---|
| **short** (Experiment 2) | best match among records of the same uninterrupted segment ≥ 9 ticks older |
| **long** (new, EXPERIMENTAL) | best match among records of the same uninterrupted segment **≥ 30 ticks older** |
| **both** (new) | each continuity record contributes its short and its long sample to one pooled W stream |

The thing was continuously observed between the two records, so **continuity, not recognition,**
establishes that they concern one persistent thing.

There is still **no recognition-feedback path**: a recognition link starts a new segment, and
segments never span an absence. This is audited in every run.

## Variants

| ID | Samples | Weights | Role |
|---|---|---|---|
| **A_short** | short | learned | Experiment 2's learner |
| **B_long** | long | learned | primary test |
| **C_both** | both | learned | combination, with no extra rule |
| E_long_equal | long | equal | attribution: separates the effect of long samples on *votes* from their effect on *weights* |
| D0_short_equal | short | equal | Experiment 2 baseline |
| D1_location_only | — | — | shortcut |
| D2_always_new | — | — | shortcut |

## World (harness only; the learner is told nothing)

Experiment 2's world (6 things with overlapping colour and size), with one change: phase-2
observation windows are 40 ticks long.

1. Ticks 1–10 are the identity evaluation look, unchanged from Experiment 2.
2. Then 30 ticks of **continuous change in plain view**:
   - 1–2 things slide 0.9 m (0.6 m if the floor is crowded);
   - 1–2 other things rotate in place by 120° (visible size and marks change);
   - the camera moves along its arc and approaches by 0.5 m (viewpoint and apparent size change).

While the camera is away, the same swaps, moves, novel things and replacements as in Experiment 2
happen. The learner gets no event markers.

What this world provides:

- **Locally stable but unreliable over a long span:** location of sliding things; visible size and
  marks of rotating elongated things.
- **Useful over long spans:** colour, which is constant, but shared within colour groups.
- **Different reliability per unit:** some things slide or rotate and others never move.

## Seeds

- **Development:** 0–7.
- **Sealed test:** 3000–3029, run once.
- **Unit of replication:** the seed. All intervals are 95% cluster-bootstrap over seeds (4000
  resamples), and comparisons between variants are paired by seed.

## Hypotheses and falsification

**Long-gap oracle** (harness only, computed after the run): for a unit mapped to physical thing g,
the AUC with which kind-k distance separates:

- the learner's own records of g that are **≥ 60 ticks apart** (spanning continuous change and
  absences), from
- records of other target things.

This is "what stays useful for this thing's identity across persistent change". The **short-gap
oracle** (≥ 9 ticks, as in Experiment 2) is reported alongside it.

| | Hypothesis | Metric | Supported only if |
|---|---|---|---|
| **H3a** | long-span learning yields evidence preferences that better reflect what stays discriminating across persistent change | per seed: Spearman ρ between final w(u,k) and the long-gap oracle over (unit, kind) pairs; ρ(B) − ρ(A) | paired CI lower bound > 0 |
| **H3a-local** | B's preferences differ between units in the right direction | B: across-unit Spearman with the long-gap oracle within each kind, averaged | CI lower bound > 0 |
| **H3b** | location dependence decreases after observed movement | per seed: mean final w(u, location) over units whose thing slid in view (B) − same for A | CI upper bound < 0 |
| **H3c** | false identity claims decrease while re-identification is preserved | (i) false-SAME over phase-2 reappearances, swaps, moves, novel and same-place-different things: B − A; (ii) correct-SAME over all eligible reappearances: B − A | (i) CI upper bound < 0, **and** (ii) CI lower bound ≥ −0.05 |

**Pre-stated fragmentation check (descriptive, not pass/fail).** Long-span learning is
*inoperative* in a seed if fewer than half of B's mapped units received ≥ 5 long W samples. The
number of such seeds is reported, together with:

- units and identity groups per thing;
- the fraction of visible thing-ticks tracked by a unit whose uninterrupted segment is ≥ 30 ticks
  old.

Also reported, not pass/fail:

- C_both and E_long_equal on every metric;
- ρ against the short-gap oracle (A is expected to fit it better);
- the window-level location Δ from Experiment 2;
- UNKNOWN rates per condition;
- D1 and D2;
- all audits.

**Answers to the two headline questions:**

- "Long-span continuity improved learned evidence" = H3a.
- "That improvement affected identity decisions" = H3c.

## Process (as in Experiment 2)

| Command | What it does |
|---|---|
| `python -m harness.run_many --split dev\|test` | starts work detached |
| `python -m harness.status --split test` | shows run state from `results/<split>/status.json`; no Claude needed |
| `python -m harness.report --split test` | writes `REPORT.md` and `summary.json` |

Completed seeds are written atomically and never rerun.

## Changes after development

(Listed here before the sealed run. Empty means none.)
