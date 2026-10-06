"""Aggregate a split's result files into statistics + a markdown report.

python -m harness.report --split dev
Criteria implemented here are those in PREREGISTRATION.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOT = 4000
FALSE_SAME_LIMIT = 0.05          # pre-registered (EXPERIMENTAL value)
CONTAMINATION_LIMIT = 0.10       # pre-registered (EXPERIMENTAL value)
OUTCOMES = ["correct_same", "false_same", "new", "unknown", "not_detected"]


def load(split):
    files = sorted(glob.glob(os.path.join(ROOT, "results", split, "*_seed*.json")))
    return [json.load(open(f)) for f in files]


def eligible_reappearance(t):
    return (t["type"] in ("moved", "unmoved") and t["experienced_viewpoint"] and t["was_unobserved"]
            and t["event_unobserved"] is not False)


def boot_rate(per_seed, rng):
    """per_seed: list of (k, n). Cluster bootstrap over seeds."""
    k = np.array([a for a, _ in per_seed], float)
    n = np.array([b for _, b in per_seed], float)
    if n.sum() == 0:
        return None, None, None
    idx = rng.integers(0, len(k), size=(BOOT, len(k)))
    ks, ns = k[idx].sum(1), n[idx].sum(1)
    rates = np.where(ns > 0, ks / np.maximum(ns, 1), np.nan)
    lo, hi = np.nanpercentile(rates, [2.5, 97.5])
    return k.sum() / n.sum(), float(lo), float(hi)


def cp_upper(k, n, conf=0.95):
    return 1.0 if n == 0 else (1.0 if k == n else float(stats.beta.ppf(conf, k + 1, n - k)))


def summarize(results):
    rng = np.random.default_rng(12345)
    variants = list(results[0]["variants"])
    out = {"n_files": len(results), "seeds": sorted({r["seed"] for r in results}), "variants": {}}
    for v in variants:
        V = {}
        conds = {
            "A_return_unmoved": lambda r, t: r["stage"] == "return" and eligible_reappearance(t) and t["type"] == "unmoved",
            "B_moved_moved": lambda r, t: r["stage"] == "moved" and eligible_reappearance(t) and t["type"] == "moved",
            "B_moved_unmoved": lambda r, t: r["stage"] == "moved" and eligible_reappearance(t) and t["type"] == "unmoved",
            "novel": lambda r, t: t["type"] == "novel",
            "same_place_different": lambda r, t: t["type"] == "replace_new",
            "unseen_viewpoint": lambda r, t: t["type"] in ("moved", "unmoved") and t["experienced_viewpoint"] is False,
        }
        for cname, pred in conds.items():
            per_seed = defaultdict(Counter)
            lat = []
            for r in results:
                for t in r["variants"][v]["trials"]:
                    if pred(r, t):
                        fin = t["seq"][-1]["outcome"]
                        per_seed[r["seed"]][fin] += 1
                        per_seed[r["seed"]]["_n"] += 1
                        if fin == "correct_same":
                            first = next(i for i, o in enumerate(t["seq"]) if o["outcome"] == "correct_same")
                            lat.append(first)
            seeds = sorted({r["seed"] for r in results})
            tot = Counter()
            for s in seeds:
                tot.update(per_seed[s])
            entry = {"n": tot["_n"], "counts": {o: tot[o] for o in OUTCOMES}}
            for o in ("correct_same", "false_same", "unknown", "new", "not_detected"):
                rate, lo, hi = boot_rate([(per_seed[s][o], per_seed[s]["_n"]) for s in seeds], rng)
                entry[o] = {"rate": rate, "ci95": [lo, hi]}
            entry["false_same"]["cp_upper95"] = cp_upper(tot["false_same"], tot["_n"])
            entry["correct_latency_ticks_mean"] = float(np.mean(lat)) if lat else None
            V[cname] = entry
        # specificity pooled over novel + same-place-different trials
        per_seed = defaultdict(lambda: [0, 0])
        for r in results:
            for t in r["variants"][v]["trials"]:
                if t["type"] in ("novel", "replace_new"):
                    per_seed[r["seed"]][1] += 1
                    per_seed[r["seed"]][0] += t["seq"][-1]["outcome"] == "false_same"
        ks = [per_seed[s] for s in sorted(per_seed)]
        rate, lo, hi = boot_rate(ks, rng)
        k_tot, n_tot = sum(a for a, _ in ks), sum(b for _, b in ks)
        V["H1d_false_same_novel_all"] = {"n": n_tot, "k": k_tot, "rate": rate, "ci95": [lo, hi],
                                         "upper95": max(hi or 0, cp_upper(k_tot, n_tot))}
        # discovery
        cont = [r["variants"][v]["discovery"]["contamination_px_rate"] for r in results
                if r["variants"][v]["discovery"]["contamination_px_rate"] is not None]
        rate, lo, hi = boot_rate([(c, 1) for c in cont], rng)
        frag = [n for r in results for n in r["variants"][v]["discovery"]["fragmentation_groups_per_thing"].values()]
        ufrag = [n for r in results for n in r["variants"][v]["discovery"]["fragmentation_units_per_thing"].values()]
        cov = [c for r in results for c in r["variants"][v]["discovery"]["coverage_per_thing"].values()]
        V["H1a"] = {"contamination_mean": rate, "contamination_ci95": [lo, hi],
                    "fragmentation_groups_mean": float(np.mean(frag)), "fragmentation_groups_median": float(np.median(frag)),
                    "fragmentation_units_mean": float(np.mean(ufrag)),
                    "coverage_mean": float(np.mean(cov)),
                    "spurious_units_mean": float(np.mean([r["variants"][v]["discovery"]["spurious_units"] for r in results])),
                    "units_mean": float(np.mean([r["variants"][v]["discovery"]["n_units"] for r in results]))}
        # audits
        aud = [r["variants"][v]["audit"] for r in results]
        V["audits"] = {
            "observations_unaltered_all": all(a["observations_unaltered"] for a in aud),
            "missing_provenance_total": sum(a["inference_missing_provenance"] for a in aud),
            "locality_violations_total": sum(a["locality_violations"] for a in aud),
            "leaks_found": sorted({x for a in aud for x in a["canary_or_names_in_learner_state"]}),
        }
        V["unknown_includes_correct"] = sum(
            1 for r in results for t in r["variants"][v]["trials"]
            if eligible_reappearance(t) and t["seq"][-1]["outcome"] == "unknown" and t["seq"][-1].get("unknown_includes_correct"))
        out["variants"][v] = V
    out["criteria"] = criteria(out)
    return out


def criteria(S):
    V = S["variants"]
    m = V["main"]
    res = {}
    res["H1a_contamination"] = {"value_upper": m["H1a"]["contamination_ci95"][1], "limit": CONTAMINATION_LIMIT,
                                "pass": (m["H1a"]["contamination_ci95"][1] or 0) <= CONTAMINATION_LIMIT}
    res["H1d_specificity"] = {"upper95": m["H1d_false_same_novel_all"]["upper95"], "limit": FALSE_SAME_LIMIT,
                              "pass": m["H1d_false_same_novel_all"]["upper95"] <= FALSE_SAME_LIMIT}
    for cond in ("A_return_unmoved", "B_moved_moved"):
        lo = m[cond]["correct_same"]["ci95"][0]
        comps = {}
        ok = lo is not None
        for v, d in V.items():
            if not v.startswith("baseline"):
                continue
            spec_ok = d["H1d_false_same_novel_all"]["upper95"] <= FALSE_SAME_LIMIT
            hi = d[cond]["correct_same"]["ci95"][1]
            beats = lo is not None and hi is not None and lo > hi
            comps[v] = {"baseline_correct_upper": hi, "baseline_meets_specificity": spec_ok, "main_beats": beats}
            if spec_ok and not beats:
                ok = False
        res[f"H1c_{cond}"] = {"main_correct_lower": lo, "vs_baselines": comps,
                              "pass": bool(ok and res["H1d_specificity"]["pass"])}
    res["H1b_note"] = "Accumulation is reported descriptively via records/units; see report."
    return res


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def markdown(S, split):
    L = [f"# Experiment 1 results — split `{split}`", "",
         f"Files: {S['n_files']}; seeds: {S['seeds'][0]}–{S['seeds'][-1]} ({len(S['seeds'])})", ""]
    C = S["criteria"]
    L += ["## Pre-registered criteria (main learner)", "", "| Criterion | Value | Limit / comparison | Pass |", "|---|---|---|---|"]
    L.append(f"| H1a contamination (upper 95%) | {fmt(C['H1a_contamination']['value_upper'])} | ≤ {CONTAMINATION_LIMIT} | {C['H1a_contamination']['pass']} |")
    L.append(f"| H1d false-SAME on novel + same-place-different (upper 95%) | {fmt(C['H1d_specificity']['upper95'])} | ≤ {FALSE_SAME_LIMIT} | {C['H1d_specificity']['pass']} |")
    for cond in ("A_return_unmoved", "B_moved_moved"):
        c = C[f"H1c_{cond}"]
        L.append(f"| H1c {cond}: correct-SAME lower 95% | {fmt(c['main_correct_lower'])} | above every baseline that meets H1d | {c['pass']} |")
    L += ["", "## Outcome rates per condition (final belief at end of observe window)", ""]
    for cond in ("A_return_unmoved", "B_moved_moved", "B_moved_unmoved", "novel", "same_place_different", "unseen_viewpoint"):
        L += [f"### {cond}", "", "| variant | n | correct SAME | false SAME | NEW | UNKNOWN | not detected |", "|---|---|---|---|---|---|---|"]
        for v, d in S["variants"].items():
            e = d[cond]
            cell = lambda o: f"{fmt(e[o]['rate'])} [{fmt(e[o]['ci95'][0])}, {fmt(e[o]['ci95'][1])}]"
            L.append(f"| {v} | {e['n']} | {cell('correct_same')} | {cell('false_same')} | {cell('new')} | {cell('unknown')} | {cell('not_detected')} |")
        L.append("")
    L += ["## Specificity pooled (novel + same-place-different)", "", "| variant | false SAME k/n | rate | upper 95% |", "|---|---|---|---|"]
    for v, d in S["variants"].items():
        h = d["H1d_false_same_novel_all"]
        L.append(f"| {v} | {h['k']}/{h['n']} | {fmt(h['rate'])} | {fmt(h['upper95'])} |")
    L += ["", "## Discovery (H1a) and audits", "", "| variant | contamination mean [95%] | groups/thing mean (median) | units/thing | coverage | units/run | spurious/run | audits |", "|---|---|---|---|---|---|---|---|"]
    for v, d in S["variants"].items():
        h, a = d["H1a"], d["audits"]
        aud = "ok" if (a["observations_unaltered_all"] and not a["missing_provenance_total"] and not a["locality_violations_total"] and not a["leaks_found"]) else json.dumps(a)
        L.append(f"| {v} | {fmt(h['contamination_mean'])} [{fmt(h['contamination_ci95'][0])}, {fmt(h['contamination_ci95'][1])}] | {h['fragmentation_groups_mean']:.2f} ({h['fragmentation_groups_median']:.0f}) | {h['fragmentation_units_mean']:.2f} | {h['coverage_mean']:.3f} | {h['units_mean']:.1f} | {h['spurious_units_mean']:.1f} | {aud} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    res = load(a.split)
    S = summarize(res)
    d = os.path.join(ROOT, "results", a.split)
    json.dump(S, open(os.path.join(d, "summary.json"), "w"), indent=1)
    open(os.path.join(d, "REPORT.md"), "w").write(markdown(S, a.split))
    print(markdown(S, a.split))
