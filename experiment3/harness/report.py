"""Experiment 3 report: python -m harness.report --split dev|test
Implements exactly the metrics and criteria in SPEC.md."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOT = 4000
KINDS = ["spatial_extent", "chroma_distribution", "metric_local_patterns_highpass", "location_self_frame"]
LEARNED = ["A_short", "B_long", "C_both"]
NONINF_MARGIN = 0.05
OUTS = ["correct_same", "false_same", "new", "unknown", "not_detected"]
L = "location_self_frame"


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


def ci(vals, rng):
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [None, None, None]
    b = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def paired(a, b, rng):
    a, b = np.array(a, float), np.array(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return ci(a[ok] - b[ok], rng)


def kn(res, v, pred, outcome):
    out = []
    for r in res:
        ts = [t for t in r["variants"][v]["trials"] if pred(t)]
        out.append((sum(t["final"]["outcome"] == outcome for t in ts), len(ts)))
    return out


def boot_ratio(kns, rng):
    k = np.array([a for a, _ in kns], float)
    n = np.array([b for _, b in kns], float)
    if n.sum() == 0:
        return [None, None, None]
    idx = rng.integers(0, len(k), (BOOT, len(k)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return [float(k.sum() / n.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def boot_diff(a, b, rng):
    ka, na = (np.array(x, float) for x in zip(*a))
    kb, nb = (np.array(x, float) for x in zip(*b))
    idx = rng.integers(0, len(ka), (BOOT, len(ka)))
    d = ka[idx].sum(1) / np.maximum(na[idx].sum(1), 1) - kb[idx].sum(1) / np.maximum(nb[idx].sum(1), 1)
    return [float(ka.sum() / na.sum() - kb.sum() / nb.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y)[0])


def per_seed(res, v):
    """Per-seed correspondence, location and fragmentation values for one learned variant."""
    out = {"rho_long": [], "rho_short": [], "rho_local_long": [], "loc_slid": [], "window_did": [],
           "frac_units_long5": [], "tracked_long_frac": []}
    for r in res:
        units = [u for u in r["variants"][v]["units"] if u["oracle_long"]]
        for key, oname in (("rho_long", "oracle_long"), ("rho_short", "oracle")):
            x, y = [], []
            for u in units:
                for k in KINDS:
                    if k in u[oname]:
                        x.append(u["learned_final"].get(k, 0.0))
                        y.append(u[oname][k])
            out[key].append(spearman(x, y))
        loc = []
        for k in KINDS:
            us = [u for u in units if k in u["oracle_long"]]
            loc.append(spearman([u["learned_final"].get(k, 0.0) for u in us], [u["oracle_long"][k] for u in us]))
        loc = [x for x in loc if np.isfinite(x)]
        out["rho_local_long"].append(np.mean(loc) if loc else np.nan)
        slid = [u["learned_final"].get(L, 0.0) for u in r["variants"][v]["units"] if u["motion_class"] == "slid"]
        out["loc_slid"].append(np.mean(slid) if slid else np.nan)
        wins, per = {}, []
        for w in r["variants"][v].get("slide_windows", []):
            if w.get("same_unit_throughout"):
                wins.setdefault(w["t0"], []).append(w)
        for ws in wins.values():
            ds = [w["loc_w_end"] - w["loc_w_t0"] for w in ws if w["slid"]]
            dn = [w["loc_w_end"] - w["loc_w_t0"] for w in ws if not w["slid"]]
            if ds and dn:
                per.append(np.mean(ds) - np.mean(dn))
        out["window_did"].append(np.mean(per) if per else np.nan)
        mapped = r["variants"][v]["units"]
        out["frac_units_long5"].append(np.mean([u["long_w_samples"] >= 5 for u in mapped]) if mapped else np.nan)
        f = r["variants"][v]["fragmentation_tracking"]
        out["tracked_long_frac"].append(f["long_tracked_ticks"] / f["visible_ticks"] if f["visible_ticks"] else np.nan)
    return out


def summarize(res):
    rng = np.random.default_rng(31337)
    variants = list(res[0]["variants"])
    S = {"n_seeds": len(res), "seeds": [r["seed"] for r in res], "rates": {}, "audits": {}, "learned": {}, "frag": {}}
    for v in variants:
        S["rates"][v] = {c: {"n": sum(n for _, n in kn(res, v, p, "x")), **{o: boot_ratio(kn(res, v, p, o), rng) for o in OUTS}}
                         for c, p in CONDS.items()}
        S["rates"][v]["false_pool"] = boot_ratio(kn(res, v, FALSE_POOL, "false_same"), rng)
        S["rates"][v]["correct_pool"] = boot_ratio(kn(res, v, CORRECT_POOL, "correct_same"), rng)
        aud = [r["variants"][v]["audit"] for r in res]
        S["audits"][v] = {
            "observations_unaltered_all": all(a["observations_unaltered"] for a in aud),
            "missing_provenance": sum(a["inference_missing_provenance"] for a in aud),
            "locality_violations": sum(a["locality_violations"] for a in aud),
            "leaks": sorted({x for a in aud for x in a["canary_or_names_in_learner_state"]}),
            "w_not_from_continuity": sum(a["feedback"]["w_samples_not_from_continuity"] for a in aud),
            "w_samples": sum(a["feedback"]["w_samples"] for a in aud),
            "w_samples_long": sum(a["feedback"]["w_samples_long"] for a in aud),
        }
        disc = [r["variants"][v]["discovery"] for r in res]
        frag_g = [n for d in disc for n in d["fragmentation_groups_per_thing"].values()]
        frag_u = [n for d in disc for n in d["fragmentation_units_per_thing"].values()]
        S["frag"][v] = {"groups_per_thing_mean": float(np.mean(frag_g)), "groups_per_thing_median": float(np.median(frag_g)),
                        "units_per_thing_mean": float(np.mean(frag_u)),
                        "units_per_run": float(np.mean([r["variants"][v]["learner_stats"]["n_units"] for r in res]))}
    seeds = {v: per_seed(res, v) for v in variants if v in LEARNED or v == "E_long_equal" or v == "D0_short_equal"}
    for v, ps in seeds.items():
        units = [u for r in res for u in r["variants"][v]["units"]]
        S["learned"][v] = {
            "rho_long": ci(ps["rho_long"], rng), "rho_short": ci(ps["rho_short"], rng),
            "rho_local_long": ci(ps["rho_local_long"], rng), "loc_usefulness_slid_units": ci(ps["loc_slid"], rng),
            "window_did": ci(ps["window_did"], rng),
            "mapped_units": len(units),
            "mean_learned_by_kind": {k: float(np.mean([u["learned_final"].get(k, 0.0) for u in units])) for k in KINDS},
            "mean_oracle_long_by_kind": {k: float(np.mean([u["oracle_long"][k] for u in units if k in u["oracle_long"]])) for k in KINDS},
            "mean_oracle_short_by_kind": {k: float(np.mean([u["oracle"][k] for u in units if k in u["oracle"]])) for k in KINDS},
        }
        S["frag"][v].update({"frac_mapped_units_with_5_long_samples": ci(ps["frac_units_long5"], rng),
                             "tracked_by_long_segment_frac": ci(ps["tracked_long_frac"], rng),
                             "inoperative_seeds": int(np.sum(np.array(ps["frac_units_long5"], float) < 0.5))})
    S["compare"] = {}
    for x, y in (("B_long", "A_short"), ("C_both", "A_short"), ("E_long_equal", "D0_short_equal")):
        if x in seeds and y in seeds:
            S["compare"][f"{x}-{y}"] = {
                "rho_long": paired(seeds[x]["rho_long"], seeds[y]["rho_long"], rng),
                "loc_slid": paired(seeds[x]["loc_slid"], seeds[y]["loc_slid"], rng),
                "false_pool": boot_diff(kn(res, x, FALSE_POOL, "false_same"), kn(res, y, FALSE_POOL, "false_same"), rng),
                "correct_pool": boot_diff(kn(res, x, CORRECT_POOL, "correct_same"), kn(res, y, CORRECT_POOL, "correct_same"), rng),
            }
    BA = S["compare"]["B_long-A_short"]
    S["criteria"] = {
        "H3a": bool(BA["rho_long"][1] is not None and BA["rho_long"][1] > 0),
        "H3a_local": bool(S["learned"]["B_long"]["rho_local_long"][1] is not None and S["learned"]["B_long"]["rho_local_long"][1] > 0),
        "H3b": bool(BA["loc_slid"][2] is not None and BA["loc_slid"][2] < 0),
        "H3c": bool(BA["false_pool"][2] < 0 and BA["correct_pool"][1] >= -NONINF_MARGIN),
    }
    return S


def f(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def fci(t):
    return "n/a" if t is None or t[0] is None else f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"


def markdown(S, split):
    C, BA = S["criteria"], S["compare"]["B_long-A_short"]
    Ls = [f"# Experiment 3 — `{split}` ({S['n_seeds']} seeds: {S['seeds'][0]}–{S['seeds'][-1]})", "",
          "## Pre-registered hypotheses (B_long vs A_short)", "", "| | Metric | Value [95% CI] | Supported |", "|---|---|---|---|",
          f"| H3a | ρ(learned, long-gap oracle): B − A | {fci(BA['rho_long'])} | {C['H3a']} |",
          f"| H3a-local | B across-unit ρ with long-gap oracle | {fci(S['learned']['B_long']['rho_local_long'])} | {C['H3a_local']} |",
          f"| H3b | final location usefulness of slid-thing units: B − A | {fci(BA['loc_slid'])} | {C['H3b']} |",
          f"| H3c | false-SAME pool: B − A (must be < 0) | {fci(BA['false_pool'])} | {C['H3c']} |",
          f"| | correct-SAME pool: B − A (must be ≥ −{NONINF_MARGIN}) | {fci(BA['correct_pool'])} | |", "",
          "## Other comparisons", "", "| comparison | ρ long-gap | location usefulness (slid units) | false-SAME | correct-SAME |", "|---|---|---|---|---|"]
    for name, c in S["compare"].items():
        Ls.append(f"| {name} | {fci(c['rho_long'])} | {fci(c['loc_slid'])} | {fci(c['false_pool'])} | {fci(c['correct_pool'])} |")
    Ls += ["", "## Learned evidence vs oracles", "",
           "| variant | mapped units | ρ long-gap oracle | ρ short-gap oracle | local ρ (long) | loc. usefulness, slid units | window Δ (slid − not) |",
           "|---|---|---|---|---|---|---|"]
    for v, d in S["learned"].items():
        Ls.append(f"| {v} | {d['mapped_units']} | {fci(d['rho_long'])} | {fci(d['rho_short'])} | {fci(d['rho_local_long'])} | {fci(d['loc_usefulness_slid_units'])} | {fci(d['window_did'])} |")
    Ls += ["", "Mean learned usefulness / long-gap oracle / short-gap oracle by kind:", "",
           "| variant | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for v, d in S["learned"].items():
        Ls.append(f"| {v} learned | " + " | ".join(f(d["mean_learned_by_kind"][k]) for k in KINDS) + " |")
    v0 = "B_long"
    Ls.append("| oracle long (B's units) | " + " | ".join(f(S["learned"][v0]["mean_oracle_long_by_kind"][k]) for k in KINDS) + " |")
    Ls.append("| oracle short (B's units) | " + " | ".join(f(S["learned"][v0]["mean_oracle_short_by_kind"][k]) for k in KINDS) + " |")
    Ls += ["", "## Fragmentation", "",
           "| variant | units/run | units/thing | groups/thing mean (median) | thing-ticks tracked by ≥30-tick segment | mapped units with ≥5 long samples | inoperative seeds |",
           "|---|---|---|---|---|---|---|"]
    for v, d in S["frag"].items():
        Ls.append(f"| {v} | {d['units_per_run']:.0f} | {d['units_per_thing_mean']:.2f} | {d['groups_per_thing_mean']:.2f} ({d['groups_per_thing_median']:.0f}) | "
                  f"{fci(d.get('tracked_by_long_segment_frac'))} | {fci(d.get('frac_mapped_units_with_5_long_samples'))} | {d.get('inoperative_seeds', 'n/a')} |")
    Ls += ["", "## Identity outcomes (final belief after a 10-tick look)", ""]
    for c in CONDS:
        Ls += [f"### {c}", "", "| variant | n | correct SAME | false SAME | NEW | UNKNOWN | not detected |", "|---|---|---|---|---|---|---|"]
        for v, R in S["rates"].items():
            Ls.append(f"| {v} | {R[c]['n']} | " + " | ".join(fci(R[c][o]) for o in OUTS) + " |")
        Ls.append("")
    Ls += ["### Pools", "", "| variant | false-SAME pool | correct-SAME pool |", "|---|---|---|"]
    for v, R in S["rates"].items():
        Ls.append(f"| {v} | {fci(R['false_pool'])} | {fci(R['correct_pool'])} |")
    Ls += ["", "## Audits", "", "| variant | obs unaltered | missing provenance | locality | leaks | W not from continuity / all W | long W |", "|---|---|---|---|---|---|---|"]
    for v, a in S["audits"].items():
        Ls.append(f"| {v} | {a['observations_unaltered_all']} | {a['missing_provenance']} | {a['locality_violations']} | {a['leaks'] or 'none'} | {a['w_not_from_continuity']} / {a['w_samples']} | {a['w_samples_long']} |")
    return "\n".join(Ls) + "\n"


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
