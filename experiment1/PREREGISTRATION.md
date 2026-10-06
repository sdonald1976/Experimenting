# Experiment 1 — pre-registration

This file is committed to git **before** the sealed test seeds are run. The test runner
(`harness/run_many.py --split test`) refuses to run while `learner/`, `harness/` or this
file has uncommitted changes, and records the commit hash in `results/test/manifest.json`.

## Hypotheses

- **H1a (discovery):** the learner forms persistent units that are not contaminated, i.e. not
  mixing different physical things.
- **H1b (accumulation):** units accumulate several experience records through continuity.
  Reported descriptively, not pass/fail.
- **H1c (re-identification):** after a thing has been out of view, the learner's belief about
  it, seen again from an already-experienced viewpoint, links to the identity it had before.
- **H1d (specificity):** never-seen things — including a different thing put in the exact place
  of a known one — are not linked to an existing identity.

## Data

- **Development seeds 0–9:** used for all debugging and any parameter change. Every change
  made after seeing dev results is listed in "Changes after dev" below.
- **Sealed test seeds 1000–1029:** run once, after this file is committed.
- **Stages:** `return` (things stay put) and `moved` (in each plain cycle, one thing is
  relocated while unobserved).
- **Per run:** 7 cycles = 4 plain + 2 novel + 1 same-place-different-thing.
- **Unit of replication:** the seed. Confidence intervals are 95% cluster-bootstrap
  intervals over seeds (4000 resamples).

## Trial eligibility (decided before results)

Reappearance trials count toward H1c only if all of the following hold:

- the thing was invisible for at least one tick of the cycle;
- any world change happened while it was unobserved (checked from ground truth);
- at the evaluation tick it is visible with at least 150 px;
- its current viewpoint is within 15° of an object-frame azimuth experienced before the cycle
  (Q15; harness-only).

Trials from never-experienced viewpoints are reported separately and not scored against
H1c.

## Outcome of a trial

Measured at the last tick of the 10-tick observation window.

- **correct SAME:** the learner's claimed unit, and the identity group it believes it belongs
  to, contains a unit associated with this thing before the cycle began, and none associated
  with another thing.
- **false SAME:** the group contains units associated with another thing (structure
  included).
- **NEW:** none of the group's units were associated with anything before.
- **UNKNOWN:** the claimed unit is provisional (`possibly_same_as`). Not an error, not correct.
- **not detected:** no learner unit claims the thing.

All outcomes are reported for every variant.

## Criteria for the main learner

All values are EXPERIMENTAL choices, fixed here:

1. **H1a:** the upper 95% bound of the mean per-run contamination rate must be ≤ 0.10.
2. **H1d:** the false-SAME rate on novel + same-place-different trials must have an upper
   95% bound ≤ 0.05. The bound used is the larger of the bootstrap bound and the one-sided
   Clopper-Pearson bound. With 90 trials, zero errors gives a bound of about 0.033.
3. **H1c**, evaluated separately for condition A (`return` stage, unmoved things) and
   condition B (`moved` stage, the moved thing). It passes if:
   - the main learner's correct-SAME lower 95% bound is above the upper 95% bound of every
     baseline that itself satisfies H1d; and
   - H1d passes.

   Baselines that fail H1d (any "always match" strategy) cannot beat the learner on
   re-identification, because their specificity is unacceptable. This is the joint "beat
   every baseline on re-identification AND specificity" rule.

For **condition B**, a high UNKNOWN rate is the expected honest outcome if location was
learned to be highly distinguishing (contradiction C2 in the design discussion). This will
be reported, not hidden.

## Also reported (not pass/fail)

- Outcome rates for the `moved`-stage unmoved things.
- Unseen-viewpoint trials.
- Latency to first correct belief.
- Fragmentation (identity groups and units per thing).
- Coverage, spurious units.
- The UNKNOWN candidate lists.
- The no-location ablation.
- All audits:
  - observations unaltered;
  - provenance present;
  - locality violations;
  - canary / human-name scan of the learner's state.

## Changes after dev

The first dev run (`results/dev_v1/`, all 6 variants, code commit `39155db`) failed H1d (3/60
false SAME) and H1a narrowly (contamination upper bound 0.104). Diagnosis on dev seeds 1 and 2
(learner belief dumps) found two defects, fixed as NEW registered versions. The old versions
remain registered and replaceable.

1. **`metric_local_patterns` → `metric_local_patterns_highpass` (v2).** v1 patches were dominated
   by smooth shading and face edges shared by every sphere or box, so "local pattern" evidence
   acted as a second shape cue. v2 high-pass filters the grey image at the patch scale before
   sampling. New parameter: `highpass_sigma_frac = 0.35`.
2. **`tail_ratio` → `tail_ratio_matched` (v2), and `leave_one_out_evidence` →
   `leave_one_out_evidence_median` (v2).** Recognition compares a query with a unit's
   *best-matching* record, but v1's usefulness samples were *single-pair* distances, and
   revision took a minimum over many × many pairs. Both bias decisions toward SAME. v2 builds
   W/B samples with the same best-match statistic. With several query records, it scores each
   one separately and takes the per-kind median. New parameter: `w_min_gap_ticks = 9`.

**Threshold choice.** This rule was written down before looking at the sweep results. The
recognition threshold is chosen on dev from {3, 5, 8} as the smallest value whose dev
specificity bound is ≤ 0.05, i.e. the H1d criterion applied to dev. If none qualifies, the value
with the fewest dev false-SAME on novel + same-place-different trials is used, and the test is
expected to fail H1d — which will be reported as such.

No other parameters were changed. Baseline variants are unaffected by these changes: they do not
use the pattern extractor, usefulness or this recognition rule. Their dev numbers come from
`dev_v1`.

**Sweep result** (`results/dev_sweep/`): the dev false-SAME on novel + same-place-different
trials was 0/60 at every threshold (3, 5, 8), so by the rule above **threshold = 3** (unchanged).

## Frozen for the sealed test

- **Variants run:** `main` (threshold 3), `ablation_no_location`, `baseline_always_new`,
  `baseline_nearest_location`, `baseline_most_recent`, `baseline_random`.
  The sweep variants are not run on test.
- **Seeds:** 1000–1029, both stages, one run each.
- **Code:** whatever commit `results/test/manifest.json` records. The runner refuses uncommitted
  changes to `learner/`, `harness/` or this file.

## Post-freeze harness fix (during the sealed test)

The first test launch (commit `ec59516`) stopped when one job's world generator could not find
a free floor position for a new thing ("no free position"). The fix changes only that failure
path: it retries with another shape and size, or moves another thing. Jobs that did not hit the
failure consume the same random numbers and produce identical worlds, so the 21 completed test
files are kept unchanged. The learner, scoring and criteria were not touched. The remaining jobs
were run with the fixed commit recorded in the manifest.
