"""Experiment 2 report: python -m harness.report --split dev|test
Implements exactly the metrics and falsification rules in SPEC.md."""
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
KINDS = ["spatial_extent", "chroma_distribution", "metric_local_patterns_highpass", "location_self_frame"]
FIXED = {"metric_local_patterns_highpass": 0.35, "chroma_distribution": 0.30, "spatial_extent": 0.20,
         "location_self_frame": 0.15}
NONINF_MARGIN = 0.10
OUTS = ["correct_same", "false_same", "new", "unknown", "not_detected"]


def eligible(t):
    return t["experienced_viewpoint"] and t["was_unobserved"] and t["event_unobserved"] is not False


CONDS = {
    "P1_reappear": lambda t: t["phase"] == 1 and t["type"] == "unmoved" and eligible(t),
    "P2_unmoved": lambda t: t["phase"] == 2 and t["type"] == "unmoved" and eligible(t),
    "P2_swapped": lambda t: t["type"] == "swapped" and eligible(t),
    "P2_moved": lambda t: t["type"] == "moved" and eligible(t),
    "novel": lambda t: t["type"] == "novel",
    "same_place_different": lambda t: t["type"] == "replace_new",
    "unseen_viewpoint": lambda t: t["type"] in ("unmoved", "swapped", "moved") and t["experienced_viewpoint"] is False,
}
FALSE_POOL = lambda t: (t["phase"] == 2 and t["type"] in ("unmoved", "swapped", "moved") and eligible(t)) or \
    t["type"] in ("novel", "replace_new")
CORRECT_POOL = lambda t: t["type"] in ("unmoved", "swapped", "moved") and eligible(t)


def load(split):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ROOT, "results", split, "seed*.json")))]


def ci(vals_by_seed, rng, stat=np.mean):
    v = np.array(vals_by_seed, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None, None, None
    idx = rng.integers(0, len(v), (BOOT, len(v)))
    b = stat(v[idx], axis=1)
    return float(stat(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def pooled_rate(res, v, pred, outcome):
    """Per seed (k, n) for a pooled rate."""
    out = []
    for r in res:
        ts = [t for t in r["variants"][v]["trials"] if pred(t)]
        out.append((sum(t["final"]["outcome"] == outcome for t in ts), len(ts)))
    return out


def boot_ratio(kn, rng):
    k = np.array([a for a, _ in kn], float)
    n = np.array([b for _, b in kn], float)
    if n.sum() == 0:
        return None, None, None
    idx = rng.integers(0, len(k), (BOOT, len(k)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(k.sum() / n.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_diff(kn_a, kn_b, rng):
    ka, na = (np.array(x, float) for x in zip(*kn_a))
    kb, nb = (np.array(x, float) for x in zip(*kn_b))
    idx = rng.integers(0, len(ka), (BOOT, len(ka)))
    d = ka[idx].sum(1) / np.maximum(na[idx].sum(1), 1) - kb[idx].sum(1) / np.maximum(nb[idx].sum(1), 1)
    return float(ka.sum() / na.sum() - kb.sum() / nb.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y)[0])


def correspondence(res):
    rho_a, rho_b, rho_local, deltas = [], [], [], []
    for r in res:
        units = [u for u in r["variants"]["A_learned"]["units"] if u["oracle"]]
        xa, xb, y = [], [], []
        for u in units:
            for k in KINDS:
                if k in u["oracle"]:
                    xa.append(u["learned_final"].get(k, 0.0))
                    xb.append(FIXED[k])
                    y.append(u["oracle"][k])
        rho_a.append(spearman(xa, y))
        rho_b.append(spearman(xb, y))
        loc = []
        for k in KINDS:
            us = [u for u in units if k in u["oracle"]]
            loc.append(spearman([u["learned_final"].get(k, 0.0) for u in us], [u["oracle"][k] for u in us]))
        loc = [x for x in loc if np.isfinite(x)]
        rho_local.append(np.mean(loc) if loc else np.nan)
        L = "location_self_frame"
        d = {c: [u["learned_final"].get(L, 0.0) - u["learned_at_phase2_start"].get(L, 0.0)
                 for u in units if u["existed_before_phase2"] and u["motion_class"] == c] for c in ("slid", "static")}
        deltas.append(np.mean(d["slid"]) - np.mean(d["static"]) if d["slid"] and d["static"] else np.nan)
    return rho_a, rho_b, rho_local, deltas


def window_did(res, v="A_learned"):
    """H2b: per seed, mean over slide windows of [mean delta(slid things' units) - mean delta(static ones)],
    delta = location usefulness at window end - at slide start, same unit tracking the thing throughout."""
    out = []
    for r in res:
        per = []
        wins = {}
        for w in r["variants"][v].get("slide_windows", []):
            if w.get("same_unit_throughout"):
                wins.setdefault(w["t0"], []).append(w)
        for ws in wins.values():
            ds = [w["loc_w_end"] - w["loc_w_t0"] for w in ws if w["slid"]]
            dn = [w["loc_w_end"] - w["loc_w_t0"] for w in ws if not w["slid"]]
            if ds and dn:
                per.append(np.mean(ds) - np.mean(dn))
        out.append(np.mean(per) if per else np.nan)
    return out


def summarize(res):
    rng = np.random.default_rng(777)
    variants = list(res[0]["variants"])
    S = {"n_seeds": len(res), "seeds": [r["seed"] for r in res], "rates": {}, "audits": {}, "usefulness": {}}
    for v in variants:
        S["rates"][v] = {}
        for c, pred in CONDS.items():
            e = {"n": sum(n for _, n in pooled_rate(res, v, pred, "x"))}
            for o in OUTS:
                e[o] = boot_ratio(pooled_rate(res, v, pred, o), rng)
            S["rates"][v][c] = e
        S["rates"][v]["H2c_false_pool"] = boot_ratio(pooled_rate(res, v, FALSE_POOL, "false_same"), rng)
        S["rates"][v]["H2c_correct_pool"] = boot_ratio(pooled_rate(res, v, CORRECT_POOL, "correct_same"), rng)
        aud = [r["variants"][v]["audit"] for r in res]
        S["audits"][v] = {
            "observations_unaltered_all": all(a["observations_unaltered"] for a in aud),
            "missing_provenance": sum(a["inference_missing_provenance"] for a in aud),
            "locality_violations": sum(a["locality_violations"] for a in aud),
            "leaks": sorted({x for a in aud for x in a["canary_or_names_in_learner_state"]}),
            "w_samples_not_from_continuity": sum(a["feedback"]["w_samples_not_from_continuity"] for a in aud),
            "w_samples": sum(a["feedback"]["w_samples"] for a in aud),
            "b_samples": sum(a["feedback"]["b_samples"] for a in aud),
        }
        if v in ("A_learned", "B_fixed", "C_equal"):
            units = [u for r in res for u in r["variants"][v]["units"]]
            S["usefulness"][v] = {
                "mapped_units": len(units),
                "learned_mean_by_kind": {k: float(np.mean([u["learned_final"].get(k, 0.0) for u in units])) if units else None for k in KINDS},
                "oracle_mean_by_kind": {k: float(np.mean([u["oracle"][k] for u in units if k in u["oracle"]])) if units else None for k in KINDS},
            }
    ra, rb, rl, dd = correspondence(res)
    diff = [a - b for a, b in zip(ra, rb)]
    S["H2a"] = {"rho_A": ci(ra, rng), "rho_B_fixed": ci(rb, rng), "rho_A_minus_B": ci(diff, rng),
                "seeds_with_value": int(np.isfinite(np.array(ra, float)).sum())}
    S["H2a_local"] = {"rho_local": ci(rl, rng), "seeds_with_value": int(np.isfinite(np.array(rl, float)).sum())}
    wd = window_did(res)
    S["H2b"] = {"did_location_delta": ci(wd, rng), "seeds_with_value": int(np.isfinite(np.array(wd, float)).sum()),
                "n_windows_slid_tracked": sum(1 for r in res for w in r["variants"]["A_learned"].get("slide_windows", [])
                                              if w.get("same_unit_throughout") and w["slid"])}
    S["H2b_long_descriptive"] = {"did_location_delta": ci(dd, rng),
                                 "seeds_with_value": int(np.isfinite(np.array(dd, float)).sum())}
    for other in ("C_equal", "B_fixed"):
        if other in variants:
            S[f"H2c_vs_{other}"] = {
                "false_diff": boot_diff(pooled_rate(res, "A_learned", FALSE_POOL, "false_same"),
                                        pooled_rate(res, other, FALSE_POOL, "false_same"), rng),
                "correct_diff": boot_diff(pooled_rate(res, "A_learned", CORRECT_POOL, "correct_same"),
                                          pooled_rate(res, other, CORRECT_POOL, "correct_same"), rng)}
    S["criteria"] = criteria(S)
    return S


def criteria(S):
    c = {}
    a, ab = S["H2a"]["rho_A"], S["H2a"]["rho_A_minus_B"]
    c["H2a"] = bool(a[1] is not None and a[1] > 0 and ab[1] is not None and ab[1] > 0)
    l = S["H2a_local"]["rho_local"]
    c["H2a_local"] = bool(l[1] is not None and l[1] > 0)
    d = S["H2b"]["did_location_delta"]
    c["H2b"] = bool(d[2] is not None and d[2] < 0)
    for other in ("C_equal", "B_fixed"):
        h = S.get(f"H2c_vs_{other}")
        if h:
            c[f"H2c_vs_{other}"] = bool(h["false_diff"][2] < 0 and h["correct_diff"][1] >= -NONINF_MARGIN)
    return c


def f(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def fci(t):
    return "n/a" if t is None or t[0] is None else f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"


def markdown(S, split):
    C = S["criteria"]
    L = [f"# Experiment 2 — `{split}` ({S['n_seeds']} seeds: {S['seeds'][0]}–{S['seeds'][-1]})", "",
         "## Pre-registered hypotheses", "", "| | Metric | Value [95% CI] | Supported |", "|---|---|---|---|",
         f"| H2a | Spearman(learned w, oracle) per seed | {fci(S['H2a']['rho_A'])} | {C['H2a']} |",
         f"| | same, B's fixed weights | {fci(S['H2a']['rho_B_fixed'])} | |",
         f"| | A − B | {fci(S['H2a']['rho_A_minus_B'])} | |",
         f"| H2a-local | across-unit Spearman within kind | {fci(S['H2a_local']['rho_local'])} | {C['H2a_local']} |",
         f"| H2b | Δ location usefulness over a slide window, slid − non-slid (same window) | {fci(S['H2b']['did_location_delta'])} (seeds: {S['H2b']['seeds_with_value']}, tracked slides: {S['H2b']['n_windows_slid_tracked']}) | {C['H2b']} |",
         f"| (descr.) | Δ across all of phase 2, pre-phase-2 units | {fci(S['H2b_long_descriptive']['did_location_delta'])} (seeds: {S['H2b_long_descriptive']['seeds_with_value']}) | |"]
    for o in ("C_equal", "B_fixed"):
        h = S.get(f"H2c_vs_{o}")
        if h:
            L.append(f"| H2c vs {o} | false-SAME A − {o} | {fci(h['false_diff'])} | {C[f'H2c_vs_{o}']} |")
            L.append(f"| | correct-SAME A − {o} (must be ≥ −{NONINF_MARGIN}) | {fci(h['correct_diff'])} | |")
    L += ["", "## Identity outcomes (final belief after a 10-tick look)", ""]
    for c in list(CONDS) + ["H2c_false_pool", "H2c_correct_pool"]:
        if c.startswith("H2c"):
            continue
        L += [f"### {c}", "", "| variant | n | correct SAME | false SAME | NEW | UNKNOWN | not detected |", "|---|---|---|---|---|---|---|"]
        for v, R in S["rates"].items():
            e = R[c]
            L.append(f"| {v} | {e['n']} | " + " | ".join(fci(e[o]) for o in OUTS) + " |")
        L.append("")
    L += ["### H2c pools", "", "| variant | false-SAME pool | correct-SAME pool |", "|---|---|---|"]
    for v, R in S["rates"].items():
        L.append(f"| {v} | {fci(R['H2c_false_pool'])} | {fci(R['H2c_correct_pool'])} |")
    L += ["", "## Learned usefulness vs oracle discriminability (mean over mapped units)", "",
          "| variant | mapped units | " + " | ".join(KINDS) + " |", "|---|---|" + "---|" * len(KINDS)]
    for v, U in S["usefulness"].items():
        L.append(f"| {v} learned | {U['mapped_units']} | " + " | ".join(f(U["learned_mean_by_kind"][k]) for k in KINDS) + " |")
        L.append(f"| {v} oracle | | " + " | ".join(f(U["oracle_mean_by_kind"][k]) for k in KINDS) + " |")
    L += ["", "## Audits", "", "| variant | obs unaltered | missing provenance | locality violations | leaks | W samples not from continuity / all W |", "|---|---|---|---|---|---|"]
    for v, a in S["audits"].items():
        L.append(f"| {v} | {a['observations_unaltered_all']} | {a['missing_provenance']} | {a['locality_violations']} | {a['leaks'] or 'none'} | {a['w_samples_not_from_continuity']} / {a['w_samples']} |")
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
