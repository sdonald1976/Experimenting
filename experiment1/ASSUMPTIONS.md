# Experiment 1 — every assumption, in one place

Nothing here is hidden in code. Mechanism-level assumptions and parameter values are
generated from the code into [MECHANISMS.md](MECHANISMS.md) and also written into every
result file.

## 1. Decisions that were still open, and what this build does

These were listed as blocking (D1–D6). They were not answered, so this build adopts the
recommendation that was on the table. Each can be changed without rewriting the system.

| # | Open decision | What this build does | Where |
|---|---|---|---|
| D1 | Can an inference ever be KNOWN? | **No.** Only raw observations are KNOWN. Every identity claim, including unit existence and continuity links, is BELIEVED or UNKNOWN. The inference store refuses KNOWN. | `learner/stores.py` |
| D2 | Does "no single kind is definitive" also apply to NEW? | **Yes, both directions.** A conclusion (SAME or NEW) must survive removing any one evidence kind. | `leave_one_out_evidence` |
| D3 | What breaks continuity? | **Any one-tick gap.** Re-appearance after any gap must go through recognition. | `overlap_continuity` |
| D4 | Self-motion signal | **Noisy body-frame motion estimate** every tick (3% relative + small absolute noise). Location is kept in the learner's own dead-reckoned frame, which drifts. | harness `SENSOR_PARAMS`, `dead_reckoning` |
| D5 | Scoring part-level / extra units | A unit lying almost entirely on one thing is fine; **contamination** (a unit whose pixels are < 80% one physical thing, among units with ≥ 20% target pixels) is penalised. Re-identification is correct when the learner's believed identity group for the region contains units associated with that thing *before it disappeared*, and none associated with any other thing. | `harness/scoring.py` |
| D6 | Scoring UNKNOWN | UNKNOWN is never counted as an error. It is an abstention: reported separately and **not** counted as correct. `possibly_same_as` = UNKNOWN. Whether the UNKNOWN's candidate list contained the right identity is also reported. Indistinguishable twins were not staged (deferred). | `harness/scoring.py`, `harness/report.py` |

## 2. Accepted exceptions to "knowledge must be earned"

These are knowledge the learner did NOT earn. They were authorised, and are listed so that
results are read correctly.

- **Sensor self-knowledge (Q1):** focal length, principal point, baseline, rectified images,
  camera mount pitch.
- **Hand-designed perception:** block-matching stereo, the candidate-region rule
  (depth jumps + chromaticity jumps), overlap continuity, the four evidence extractors.
  They encode spatial and photometric priors (e.g. "surfaces of one thing tend to share
  chromaticity"). They encode **no** object categories and nothing about this scene.
- **Treating simultaneously observed regions as different things.** This is the second
  usefulness signal (Q6b) and also excludes co-visible units from identity matches. It is an
  inference, and wrong when one thing is split into two regions.

## 3. What the learner receives — and only this

- Once: calibration `{width, height, fx, fy, cx, cy, baseline, mount_pitch}`.
- Every tick: `{tick, left (uint8 HxWx3), right (uint8 HxWx3), self_motion {d_forward, d_right, d_yaw}}`.

The learner runs in a separate OS process connected by a pipe. Both sides validate the
schema. The learner package never imports the harness (tested). After every run, the
harness checks that:

- the learner's stored observations are byte-identical to what was sent;
- every inference has provenance;
- no unit was modified without being involved in the tick (locality);
- the learner's entire belief state contains neither a canary string nor any human-readable
  thing name.

## 4. Harness-side choices (the world, not the learner)

All in `harness/scene.py` (`SCENE_PARAMS`), `harness/run_job.py` (`SENSOR_PARAMS`) and
`harness/scoring.py` (`SCORING_PARAMS`), and written into every result file.

- **Room and things:** an 8 × 8 m room, grey floor, four differently tinted walls. Four rigid
  things (box / cylinder / sphere) in the middle, deliberately overlapping in evidence:
  A and B share colour, B and C share shape and size. Each thing carries random
  intensity-only "markings" (dark spots) and fine texture. Novel things are random and may
  reuse existing colours and shapes.
- **Rendering:** no shadows, single white directional light plus ambient, pixel noise
  σ = 3/255. Chromaticity-preserving textures were chosen on purpose so stereo has texture to
  match. This couples the scene to the candidate rule; see limitations.
- **Camera route (Q3, passive):**
  1. A 1°/tick exploration arc of ±40° around the scene at 1.9 m.
  2. Then 7 cycles. In each cycle the camera turns 180° away (12°/tick), slides to a new arc
     position while facing away, turns back, and observes for 10 ticks.
- **Cycle types per run:**
  - 4 plain cycles. In the *moved* stage, one thing is relocated while unobserved (Q14 #3),
    with its yaw chosen so it is seen again from an already-experienced side.
  - 2 novel cycles: a new thing appears.
  - 1 replacement cycle: a thing is removed and a different one put in its exact place
    (control C2).
- **Not staged in this smallest version:** occlusion (Q14 #2), moving things, lighting
  changes, twins, background changes, monocular and shuffled-time controls.

## 5. Known limitations of this build (stated before seeing test results)

- The candidate rule splits multi-coloured things and merges touching same-coloured
  surfaces. The scene was designed so things are uniformly coloured. Results say nothing
  about textured or multi-coloured things.
- Things far from the cameras or at grazing angles may produce no valid stereo, and then no
  candidate at all ("not detected").
- Usefulness samples come only from this run's experience. Nothing is carried across seeds.
- The learner has no ceiling on memory (Q17). Run time grows with the number of units.
