# Experiment 2 — sealed result

- **Seeds:** 2000–2029.
- **Code:** frozen commit `01d94d9`; no seed failed. The replacement event in seeds 2015 and 2017 was skipped because the floor was too crowded (logged as `skipped_events`; failure path in SPEC.md change 4), so there are 28 rather than 30 same-place-different trials.
- **Full tables:** [results/test/REPORT.md](results/test/REPORT.md).
- **Raw per-seed files:** `results/test/seed*.json`.
- **Per-unit developmental histories:** `results/test/histories/`.

## 1. Hypothesis

A learner that weights each evidence kind *per unit*, by how well that kind has separated:

- "same thing" (its own uninterrupted continuity), from
- "different thing" (things seen at the same moment),

will hold preferences that match what its experience actually made discriminating (**H2a**). Those
preferences will differ between units (**H2a-local**) and will drop for location when it watches a
thing move (**H2b**). It will make fewer false identity claims than equal weighting without losing
much re-identification (**H2c**).

## 2. What was actually tested

All variants share:

- Experiment 1's perception and its four neutral extractors: extent, chromaticity, high-pass
  local patterns, self-frame location;
- one vote per kind;
- one decision rule: weighted mean of votes ≥ 0.5, surviving removal of any one kind; else UNKNOWN.

Only the weights differ:

| Variant | Weights |
|---|---|
| **A** | learned w(u,k): AUC of W vs. the hardest 25% of B, over the last 30 W and 60 B samples, starting at 0 |
| **B** | fixed guess: patterns 0.35, colour 0.30, size 0.20, location 0.15 |
| **C** | equal |
| **D1** | location-only shortcut |
| **D2** | never links |

Further details:

- **No feedback path:** 80,484 W samples, all from continuity; 0 from recognition.
- **World:** 6 things with deliberately overlapping colour and size. Phase 1 is static. In phase 2,
  things slide 60 cm in view, and swaps, moves, novel things and replacements happen while the
  camera is away.
- **Changes made on development seeds, before freezing:** listed in [SPEC.md](SPEC.md), including
  a fix to the shared vote and a redefinition of H2b.

## 3. Sealed results

| | Result [95% CI] | Verdict |
|---|---|---|
| **H2a** learned usefulness vs. oracle discriminability | Spearman ρ = **0.25** [0.20, 0.29]; our fixed guess B: −0.10; A − B = 0.35 [0.26, 0.44] | **supported** |
| **H2a-local** differences between units, within a kind | ρ = **0.36** [0.33, 0.40] | **supported** |
| **H2b** location usefulness change over a slide, slid − not slid | **−0.033** [−0.062, −0.007] (189 tracked slides) | **supported, but small** |
| **H2c vs C** false-SAME, A − C | +0.001 [−0.015, 0.014] | **falsified** |
| **H2c vs B** false-SAME, A − B | +0.004 [−0.007, 0.013] | **falsified** |

Identity outcomes:

| Variant | Correct SAME (all reappearances) | False SAME (phase 2 + novel) | Swapped things: false SAME |
|---|---|---|---|
| A learned | 45.7% | 1.4% | 4.2% |
| B fixed | 47.1% | 1.0% | 0.8% |
| C equal | 49.4% | 1.3% | 2.5% |
| D1 location-only | 60.7% | **36.4%** | **73.9%** |

## 4. Failures and unexpected behaviour

- **Learning the weights did not make identity decisions any safer.** A's false-claim rate equals
  C's and B's, and A re-identifies 3.6 points less than C. Swapped things fooled A slightly more
  (4.2%) than B (0.8%).
  - A learned that location is its most useful kind (mean w = 0.54).
  - Its experience was mostly a static world, so that was a reasonable thing to learn — and
    exactly what swaps exploit.
- **The H2b effect is real but small, and the dev prediction was wrong.** On development seeds
  (47 slides) H2b was not supported, and before the sealed run I predicted it would fail.
  - With 189 slides it is significant, but the size is tiny: slid units' location usefulness still
    *rose* on average (+0.13), just less than for things that did not slide (+0.16).
  - Only 14% of slid units showed an actual decrease.
  - Cause: a W sample compares with the best match ≥ 9 ticks back, so the learner mostly measures
    short-term stability and only partly notices long movements.
- **Ranking across kinds is off.** Colour has the *highest* oracle discriminability (0.89) but the
  *lowest* learned usefulness (0.25).
  - The learner judges each kind against the hardest negatives it has seen. With three same-coloured
    things present, colour rarely separates those.
  - The oracle judges against all other things. Both views are defensible, and they disagree.
  - H2a holds mainly because of within-kind, between-unit differences.
- **Units fragment.** A thing seen again after an UNKNOWN continues in a new unit, so phase-1 units
  rarely kept learning through phase 2. The original long-range H2b metric had no data in any seed.
- **History caveat:** each usefulness change records one causing sample. When several samples arrive
  on the same tick, the others are in the sample store but not named in that history line.

## 5. Comparison with baselines

- **Location-only (D1)** re-identifies best (61%), but makes false claims on 36% of phase-2 and new
  things, and 74% of swaps.
- **A, B and C** all keep false claims at 1–1.4%, and all answer UNKNOWN on about 75% of swaps. That
  safety comes from the shared rule (leave-one-out plus UNKNOWN), not from the learned weights.
- **Never-link (D2)** is perfectly safe and useless.

## 6. What it demonstrated

From continuity and co-observation alone, with no labels and no feedback from its own recognitions,
the learner forms **per-unit** evidence preferences. They track which evidence actually distinguishes
each thing better than a reasonable fixed guess does, and they vary between units in the right
direction. When it watches a thing move, its confidence in location for that thing measurably
(slightly) drops. Its history of how and why each belief changed is preserved and inspectable, for
example:

```
python -m harness.inspect_unit --split test --seed 2006 --unit 3 --kind location_self_frame
```

This shows a red box's location usefulness going from 0.75 to 0.38 during a slide.

## 7. What it did NOT demonstrate

- That learned preferences improve identity decisions: H2c failed against both equal and fixed
  weights.
- Fast or substantial revision: the effect is small, and most slid units still gained confidence in
  location.
- Any revision driven by moves the learner did *not* see: swaps and unobserved moves are invisible to
  continuity by construction, and recognition feedback is deliberately excluded.
- Correct cross-kind ranking: colour is undervalued compared with the oracle.
- Anything beyond this synthetic world: uniformly coloured rigid things, no occlusion or lighting
  change.

## 8. Recommended next scientific question

**Which experience should define "same thing" for learning evidence reliability?**

Continuity over a few ticks teaches short-term stability, but recognition operates across absences.
The learner needs a legitimate signal about long-gap sameness. Candidates:

1. Continuity over long uninterrupted observations, compared end-to-start rather than best-match.
2. A deliberately separated, independently tested feedback path from its own *high-confidence*
   recognitions, measured for self-reinforcing error against a no-feedback control.

Test whether either one turns learned weights into safer identity decisions — the part of this
hypothesis that failed — without opening a self-confirming loop.
