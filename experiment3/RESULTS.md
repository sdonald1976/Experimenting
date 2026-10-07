# Experiment 3 — sealed result

- **Seeds:** 3000–3029.
- **Code:** frozen commit `dfd6e69`; 30/30 seeds completed, no events skipped.
- **Full tables:** [results/test/REPORT.md](results/test/REPORT.md).
- **Raw per-seed files:** `results/test/seed*.json`.
- **Per-unit learning histories:** `results/test/histories/`.

## 1. Hypotheses and pre-registered criteria (B_long vs A_short)

| | Hypothesis | Supported only if | Result [95% CI] | Verdict |
|---|---|---|---|---|
| **H3a** | Long-span learning yields preferences that better match what stays discriminating across persistent change | ρ(learned, long-gap oracle) B − A has CI lower bound > 0 | **+0.040** [−0.002, +0.082] | **not supported** (narrowly) |
| **H3a-local** | B's preferences differ between units in the right direction | CI lower bound > 0 | +0.333 [0.291, 0.374] | supported |
| **H3b** | Location dependence drops after observed movement | location usefulness of units that watched their thing slide, B − A: CI upper bound < 0 | **−0.397** [−0.439, −0.354] | **supported** |
| **H3c** | Fewer false claims, re-identification preserved | false-SAME B − A: CI upper bound < 0, **and** correct-SAME B − A: CI lower bound ≥ −0.05 | false −0.003 [−0.012, +0.006]; correct **−0.119** [−0.153, −0.087] | **falsified** on both parts |

## 2. Exactly what changed from Experiment 2

**Learner:** one thing — which continuity comparisons produce "same thing" (W) samples.

| Variant | Rule for W samples |
|---|---|
| A (short, as in Experiment 2) | records in the same uninterrupted segment ≥ 9 ticks apart |
| B (long) | records in the same segment ≥ 30 ticks apart |
| C (both) | both sample types pooled |

Everything else is identical to Experiment 2: vote, learned-usefulness formula, decision rule,
revision, perception and evidence. There was still no recognition feedback: 0 of 143,383 W samples
came from anything but continuity.

**World:** each phase-2 look gained 30 ticks of continuous change in view, after the evaluation
tick:

- things sliding 0.6–0.9 m;
- things rotating 120°;
- the camera moving and approaching.

**Instrumentation:**

- history now records every sample behind each change;
- fragmentation tracking was added;
- a long-gap oracle (records ≥ 60 ticks apart, spanning change and absence) was added.

## 3. Sealed results and baselines

| Variant | Correct SAME | False SAME | ρ with long-gap oracle | Location usefulness of units that watched a slide |
|---|---|---|---|---|
| A_short (learned) | 40.9% | 2.1% | 0.26 | 0.55 |
| **B_long (learned)** | 29.0% | 1.8% | 0.30 | **0.15** |
| C_both (learned) | 39.2% | 2.2% | 0.28 | 0.44 |
| E_long_equal (long samples, equal weights) | 30.3% | **0.5%** | **0.35** | 0.17 |
| D0_short_equal (Experiment 2 baseline) | 44.4% | 1.0% | 0.27 | 0.56 |
| D1 location-only | 58.9% | **37.9%** | — | — |
| D2 always-new | 0% | 0% | — | — |

On swapped things specifically, false claims were: A 5.1%, B 2.5%, C 3.4%, E 0.8%, D0 2.5%,
location-only 66.9%.

## 4. Unexpected behaviour

- **Long-span learning made the learner less confident about everything, not just location.** All
  four evidence kinds got lower learned usefulness under B: location 0.28 vs 0.59, colour 0.14 vs
  0.24, size 0.08 vs 0.22, patterns 0.06 vs 0.25.
  - Over 30 ticks of real change, every kind varies more than it does over 9 ticks, so B's "same"
    and "different" distances overlap more for all of them.
  - The result was more UNKNOWN answers and fewer re-identifications. That is caution, not better
    discrimination.
- **The clearest gains appeared with equal weights, not learned ones.** E_long_equal vs D0_short_equal:
  - correspondence with the long-gap oracle: +0.080 [0.037, 0.122], significant;
  - false claims: 0.5% vs 1.0%, n.s.
  - This suggests long-span samples improve the per-kind *votes* more than they improve the learned
    *weights*.
- **Ranking across kinds is still off.** B still ranks location first and colour second; the long-gap
  oracle ranks colour (0.88) above size (0.81) above location (0.76).
- **More caution produced more fragments.** B splits a thing into a median of 10.5 identity groups,
  against 8 for A.

## 5. Did long-span continuity improve learned evidence?

**Partly.**

- **Location (yes, strongly):** units that watched their thing move learned to trust location far
  less (−0.40). Experiment 2's short window barely registered such movement (window Δ: A −0.01, B
  −0.21).
- **Overall correspondence with what stays discriminating (not significantly):** +0.04, CI touching
  0. The pre-registered H3a is not supported.
- **Per-unit differentiation:** kept (+0.33), but weaker than A's (+0.42).

## 6. Did that improvement affect identity decisions?

**No, not as hypothesised.**

- **False claims:** not reduced (1.8% vs 2.1%, n.s.). Swaps fooled B less (2.5% vs 5.1%), but the
  pooled reduction is not significant.
- **Re-identification:** fell by 12 points, failing the pre-registered non-inferiority margin.

The learner became more conservative, not more discerning.

## 7. Fragmentation findings

- **Continuity is short-lived.** A thing in view is tracked by a segment at least 30 ticks old only
  **21%** of the time.
- **This is identical for every variant.** Segment boundaries come from perception and the
  "any gap breaks continuity" rule (every look-away ends a segment), not from identity reasoning.
- **Long-span learning did operate.** 83% of B's mapped units received ≥ 5 long-span samples, with
  no inoperative seeds. In total it had 52,442 long samples, against 90,941 short ones for A.
- **What it learned stayed trapped in fragments.** A physical thing ends up spread over a median of
  8 (A) to 10.5 (B) identity groups. Each fragment learns alone, and nothing learned in one fragment
  reaches the next fragment of the same thing without a recognition, which is deliberately never
  used as training truth.
- **Fragmentation is now the binding constraint.** Long-span knowledge exists, but each unit's
  lifetime is too short, and its successors too disconnected, for it to accumulate.

## 8. What was demonstrated

- Continuity over substantially separated observations is a legitimate, operational teacher. With
  no labels and no recognition feedback, it taught units that watched their thing move to stop
  relying on location.
- With equal weights, long-span samples also made per-unit evidence correspond better to what
  remains discriminating across persistent change.
- The full learning provenance of every unit is inspectable, for example:

```
python -m harness.inspect_unit --split test --seed 3000 --variant B_long --unit <id> --kind location_self_frame
```

## 9. What was NOT demonstrated

- That long-span learning yields significantly better *learned weights* overall (H3a narrowly fails).
- Safer identity decisions, or preserved re-identification (H3c fails).
- Correct ranking across evidence kinds: colour is still undervalued.
- Any knowledge carried across an absence. By construction, continuity ends at every gap, and
  recognition is not a teacher.
- Anything beyond this synthetic world.

## 10. The most important next scientific question

**How can knowledge learned during one uninterrupted observation of a thing be carried into the
next observation of the same thing, when the link between the two is only a belief?**

Continuity now teaches well, but its lessons die with each fragment. The next experiment should
test whether provisional, belief-weighted pooling of experience across a recognised gap:

- accumulates long-span knowledge,
- without a self-reinforcing error loop.

That loop is exactly what Experiments 2 and 3 deliberately excluded. So the feedback path must be:

- isolated as its own arm;
- reversible when the belief is revised;
- audited against a no-feedback control.
