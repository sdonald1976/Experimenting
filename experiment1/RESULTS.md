# Experiment 1 — sealed test result

- **Seeds:** 1000–1029 (30 worlds never used during development).
- **Stages:** both (`return` and `moved`).
- **Variants:** 6.
- **Code:** commit `e44c088` (frozen code plus one harness-only fix, see PREREGISTRATION.md).
- **Full tables:** [results/test/REPORT.md](results/test/REPORT.md).
- **Raw per-seed data:** `results/test/*.json`.

## Verdict against the pre-registered criteria

| Criterion | Result | Pass |
|---|---|---|
| H1a — units do not mix physical things (contamination upper 95% ≤ 0.10) | 0.010 | **yes** |
| H1d — never-seen things not linked to an old identity (false-SAME upper 95% ≤ 0.05) | 5 / 177 = 2.8%, upper bound **0.061** | **no** |
| H1c-A — re-identification, things stayed put | 73.9% correct [68.6, 79.3]; beats every baseline that meets H1d | **no** (requires H1d) |
| H1c-B — re-identification, thing moved while unobserved | 39.5% correct [29.2, 50.0] | **no** (requires H1d) |

**As pre-registered, Experiment 1 fails.** The learner made 5 false identity claims in 177
never-seen trials. The bound on that rate (6.1%) is above the 5% limit, and H1c was defined to
require H1d.

## What actually happened

All values are on the main learner, sealed test, final belief after a 10-tick look.

| Situation | Correct SAME | False SAME | UNKNOWN | Not detected |
|---|---|---|---|---|
| Thing stayed put, seen again from an experienced side (n = 866) | **73.9%** | 0.3% | 16.6% | 9.1% |
| Thing moved while unobserved (n = 119) | 39.5% | 1.7% | **47.9%** | 10.9% |
| Never-seen thing (n = 117) | — | 1.7% | **85.5%** (NEW 0.9%) | 12.0% |
| Different thing put in a known thing's exact place (n = 60) | — | 5.0% | **85.0%** | 10.0% |
| Known thing seen from a side never experienced (n = 219) | 16.0% | 1.4% | **72.6%** | 9.6% |

### The useful contrasts

- **Against the location-only shortcut.** On things that stayed put, the location-only baseline
  is almost as accurate (71.8%). But it links 82–88% of never-seen and replaced things to an old
  identity, and its units have 41% contamination. The learner gets similar re-identification at
  about 1/30th of the false-claim rate. This is the core positive result.

- **Location evidence vs. moved things (predicted as contradiction C2).** Removing location
  evidence entirely raises moved-thing re-identification from 39.5% to **52.9%**, with 0 false
  SAME. In a mostly static world, continuity teaches that location is a highly distinguishing
  cue, so a moved thing becomes UNKNOWN. That is epistemically honest, but costly.

- **Recognition is earned, not assumed.** From never-experienced sides only 16% are
  re-identified; 73% stay UNKNOWN. This matches the architecture's claim that the system cannot
  recognise an appearance it has never experienced.

- **The learner almost never says NEW.** Never-seen things end as UNKNOWN (85.5%), almost never
  as a confident NEW (0.9%). The evidence rule requires every kind to agree before declaring
  "different", and a newcomer usually shares colour or size with something. "Not knowing" is a
  legitimate state, but the learner cannot yet *conclude* novelty.

- **When UNKNOWN, the right identity is often among the candidates.** In 202 of the 398
  eligible reappearance trials that ended UNKNOWN, the provisional `possibly_same_as` list
  contained the correct identity.

### Where the 5 false claims came from

| Trial | Never-seen thing | Linked to |
|---|---|---|
| same place | purple box | purple sphere that stood there |
| same place | red cylinder | purple sphere that stood there |
| same place | pink cylinder | green sphere that stood there |
| novel | green sphere | the old green sphere |
| novel | red cylinder | the old red box |

Four of the five were made by **identity revision** (merging a young unit into an older group),
not by the first recognition decision. In each case two or three agreeing kinds (location +
size, or colour + size) outvoted the kind that disagreed. The "no single kind is definitive"
rule permits exactly this. The extractors are weak at separating things that share a colour or
a footprint.

## Discovery, accumulation, integrity

- **Contamination:** 0.5% of pixels in units touching target things (mean per run).
- **Fragmentation is high.** A physical thing ends up in a median of **3** believed-identity
  groups; the ideal is 1. Many units are spurious: on average 94 of 112 units per run lived
  ≤ 2 ticks. Most fragments are unresolved provisional identities, not wrong ones.
- **Accumulation (H1b, descriptive):** 609 experience records per run, kept separately and
  never averaged.
- **Coverage:** in 85% of the ticks where a target thing is visible, most of its pixels are
  claimed by a unit associated with it.
- **Audits — all clean in every run and variant:**
  - stored observations byte-identical to what was sent;
  - 0 inferences without provenance;
  - 0 locality violations;
  - no canary string and no human thing name anywhere in the learner's belief state.

## What this does and does not show

**Shown, in this synthetic world:** with only stereo images, noisy self-motion and its own
calibration, and no labels, the learner forms uncontaminated persistent units. It re-identifies
things it has experienced far more selectively than shortcut strategies. It mostly says UNKNOWN
where its experience cannot justify an answer.

**Not shown:**
- that it meets the pre-registered specificity bar;
- confident novelty detection;
- low fragmentation;
- robustness to multi-coloured things, occlusion, moving things or lighting changes (not
  staged);
- anything about real cameras.

## Changes that would address the failures

These are untested. Each would be a new EXPERIMENTAL mechanism version, tested on dev, and then
a new pre-registration and a new sealed test. The test seeds used here must not be reused.

- **Make revision as conservative as first recognition.** 4/5 false claims came from merges.
- **Rethink when agreeing kinds may outvote a strongly disagreeing one.** This is an
  architectural question, not a tuning knob, because Q6 says location must never be definitive.
- **Stronger, still semantically neutral, local-pattern evidence** for separating same-coloured
  things.
