# TinyBrain — C# solution

A readable C# version of the microscopic learning system as it stood in **Experiment 2**:

- a learner that discovers persistent things from stereo images;
- that learner learning, per thing, which evidence identifies it;
- a simulated world to test the learner in, which knows the ground truth.

## Get it and run it

1. Download the repository: on GitHub, switch to branch `claude/compassionate-wozniak-ih2izy`, then
   **Code → Download ZIP**. Or use `git clone`.
2. Install the **.NET 8 SDK** (Windows, macOS or Linux). No NuGet packages are needed.
3. In this `csharp/` folder:

```
dotnet build -c Release
dotnet run -c Release --project tests/TinyBrain.Checks                 # architecture checks
dotnet run -c Release --project src/TinyBrain.Runner -- --seed 0       # one world, all 5 learner variants (~6 min)
dotnet run -c Release --project src/TinyBrain.Runner -- --seed 0 --variants A_learned --frames frames --frame-every 20
dotnet run -c Release --project src/TinyBrain.Runner -- --seeds 0-3 --out results.json
dotnet run -c Release --project src/TinyBrain.Runner -- --describe     # every mechanism, its parameters and assumptions
```

Or open `TinyBrain.sln` in Visual Studio or Rider and run `TinyBrain.Runner`.

`--frames` writes BMP images with three panels: **left camera | right camera | what the learner
believes**. In the third panel, each colour is one persistent unit the learner discovered. Black
means no depth or no candidate.

## The one rule that shapes the code

**The learner may only receive what real sensors could give it.** That's two camera images, a noisy
sense of its own movement, and its own camera calibration. It never receives object names, IDs,
masks, depth or positions.

This is enforced structurally:

- **`TinyBrain.Learner`** holds the learner. It does **not** reference `TinyBrain.World`, so it
  cannot see the simulator even by accident. The first check in `TinyBrain.Checks` verifies this.
- **`TinyBrain.World`** holds the simulator and the scorer. It knows everything, including which
  thing is which, and uses it only to *score* the learner afterwards.

```
World (ground truth) --render--> Observation {Left, Right, SelfMotion} --> Learner.Step()
        ^                                                                         |
        +----------- VariantScorer compares ------ LearnerExport (pixel claims) <-+
```

## How the learner works, file by file (`src/TinyBrain.Learner`)

| File | What it is | Status |
|---|---|---|
| `Sensing.cs` | `Observation`, `Calibration`, `SelfMotion`: everything the learner is allowed to receive | required |
| `Stores.cs` | **Observation store** (what was perceived; copied, hashed, read-only) and **inference store** (every belief, with mandatory provenance). Only observations are KNOWN; beliefs are BELIEVED or UNKNOWN. | required |
| `Units.cs` | **Unit**: a persistent identity created on first encounter, with no meaning attached. **ExperienceRecord**: one encounter, never averaged away. **Sample** / **UsefulnessChange**: append-only learning history. | required |
| `Mechanism.cs` | Base class: every non-required rule declares its name, version, parameters and assumptions | — |
| `Perception.cs` | Dead reckoning; SAD stereo matching; candidate regions (depth + colour boundaries); continuity (region overlap from one tick to the next) | EXPERIMENTAL |
| `Evidence.cs` | Four neutral evidence kinds: visible **size**, **colour** distribution, high-pass **local patterns**, **location** in the learner's own frame | EXPERIMENTAL |
| `Identity.cs` | `WindowedAucUsefulness`: per-unit "same" (W) and "different" (B) samples, a vote, and a learned usefulness weight. `WeightedVote`: the shared SAME / NEW / UNKNOWN rule. Baselines. Identity revision. | EXPERIMENTAL / BASELINE |
| `Learner.cs` | The per-tick loop (below) | required structure |
| `LearnerConfig.cs` | Which mechanism fills each slot; the 5 variants of Experiment 2 | experimenter's choice |

**One tick of `Learner.Step`:**

1. **Store** the observation and update dead reckoning.
2. **Stereo** gives depth. **Candidates** are regions bounded by depth or colour jumps.
3. **Continuity:** a candidate overlapping a unit's region from the previous tick *is* that unit
   (uninterrupted experience).
4. **Recognition**, for candidates with no continuity: compare with every known unit's records. The
   result is SAME, NEW or **UNKNOWN**. UNKNOWN creates a *provisional* unit marked
   "possibly the same as X".
5. **Co-visibility:** regions seen at the same moment are believed to be different things.
6. **Usefulness learning, local per unit:**
   - continuity gives "same thing" distances (W);
   - co-observation gives "different thing" distances (B);
   - **recognition results never become training data** (checked by `FeedbackAudit`).
7. **Revision:** young provisional units are re-compared. A merge is a *new* inference, and nothing
   is erased.

## Variants (Experiment 2)

| Variant | Recognition weights |
|---|---|
| `A_learned` | each unit's own learned usefulness |
| `B_fixed` | a fixed human guess: patterns > colour > size > location |
| `C_equal` | all evidence equal |
| `D1_location_only` | shortcut: nearest location always wins |
| `D2_always_new` | shortcut: never links anything |

## What to expect in the output

For every variant, the runner prints:

- **Outcome rates** for things seen again after the camera looked away. **FALSE** (a wrong identity
  claim) is the serious error. **UNKNOWN** is allowed, and is the honest answer when the evidence
  doesn't justify a claim.
- **Audits:**
  - observations unaltered;
  - every inference has provenance;
  - no updates to uninvolved units;
  - no ground-truth names in the learner's state;
  - no usefulness samples from recognition.
- **Learned usefulness** of the longest-lived units. The thing name next to each unit is ground
  truth shown to *you*; the learner never had it.

## Relationship to the Python experiments

The Python code in `../experiment1` and `../experiment2` is the **reference implementation**. Every
recorded and pre-registered result comes from it, and those results are not reproduced here. This
port follows the same design, parameters and rules. Differences:

- **Different random number generators**, so worlds and numbers differ seed for seed. Expect similar
  behaviour, not identical values.
- **Isolation is assembly-level** (the learner cannot reference the world). Python ran each learner
  in a separate OS process.
- **Not included:** the statistics and pre-registered criteria (bootstrap intervals, the oracle
  comparisons of Experiment 2). The runner prints raw outcome rates.
