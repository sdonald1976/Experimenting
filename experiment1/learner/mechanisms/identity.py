"""Identity mechanisms: usefulness learning, recognition, revision. ALL EXPERIMENTAL.

Recognition mechanisms all expose:
    decide(query_evs, groups, ctx) -> dict(decision, group, possible, scores)
where decision is "SAME" | "NEW" | "UNKNOWN" and groups is a list of
(group_id, [Unit, ...]) the query may be compared against (co-visible units
already excluded by the caller).
"""
from __future__ import annotations

import numpy as np

from ..registry import Mechanism, register

SAME, NEW, UNKNOWN_D = "SAME", "NEW", "UNKNOWN"


# --------------------------------------------------------------- usefulness
@register("usefulness")
class TailRatioUsefulness(Mechanism):
    NAME = "tail_ratio"
    VERSION = "1"
    DEFAULTS = {"pseudo_count": 1.0}
    ASSUMPTIONS = [
        "Usefulness is LOCAL to (unit, evidence kind) (Q6c/Q6e). It is stored as two sample sets:",
        "  W = distances between that unit's records that are linked by uninterrupted continuity "
        "(Q6b signal 1);",
        "  B = distances between that unit's records and records of OTHER units observed on the same "
        "tick (Q6b signal 2). Treating simultaneously observed regions as different things is itself "
        "an assumption (two regions could be parts of one thing).",
        "Recognition-derived links never add W or B samples (Q6b).",
        "Score of a distance d = log P_same(d) - log P_diff(d), with P_same = (#W >= d + a)/(|W| + 2a) "
        "and P_diff = (#B <= d + a)/(|B| + 2a). With no samples the score is exactly 0 (neutral, Q6d). "
        "Its magnitude is bounded by log(n + 2): confidence can only grow with experience.",
        "These are tail fractions, not calibrated likelihoods.",
    ]

    def score(self, W: np.ndarray, B: np.ndarray, d: float) -> float:
        a = self.p["pseudo_count"]
        nW, nB = len(W), len(B)
        ge = nW - np.searchsorted(W, d, side="left")      # W sorted
        le = np.searchsorted(B, d, side="right")          # B sorted
        return float(np.log((ge + a) / (nW + 2 * a)) - np.log((le + a) / (nB + 2 * a)))


# --------------------------------------------------------------- recognition
def _min_distances(query_evs, unit, extractors, kinds):
    out = {}
    for k in kinds:
        ex = extractors[k]
        best = None
        for q in query_evs:
            qa = q.get(k)
            if qa is None:
                continue
            for r in unit.records:
                ra = r.ev.get(k)
                if ra is None:
                    continue
                d = ex.distance(qa, ra)
                if best is None or d < best:
                    best = d
        if best is not None:
            out[k] = best
    return out


@register("recognition")
class LeaveOneOutEvidence(Mechanism):
    NAME = "leave_one_out_evidence"
    VERSION = "1"
    DEFAULTS = {"threshold": 3.0, "max_possible_listed": 5, "use_recognition_records": True}
    ASSUMPTIONS = [
        "Per comparison with a unit: for each evidence kind, d = the smallest distance between the "
        "query and any of the unit's experience records (best-matching past experience).",
        "Per-kind scores come from the unit's own usefulness samples and are SUMMED "
        "(treats kinds as independent - an assumption).",
        "No single kind is definitive (D2, both directions): SAME needs total >= threshold AND "
        "total >= threshold with ANY one kind removed; NEW needs the mirror image. Otherwise UNKNOWN. "
        "Fewer than two kinds available -> UNKNOWN.",
        "Across units (grouped by believed identity): exactly one SAME group -> SAME; several -> UNKNOWN; "
        "none SAME but some UNKNOWN -> UNKNOWN (provisional identity); all NEW -> NEW.",
        "threshold is an arbitrary EXPERIMENTAL value (log-odds-like units).",
    ]

    def unit_decision(self, query_evs, unit, ctx):
        kinds = ctx["kinds"]
        dists = _min_distances(query_evs, unit, ctx["extractors"], kinds)
        scores = {k: ctx["usefulness"].score(unit.W_sorted(k), unit.B_sorted(k), d) for k, d in dists.items()}
        t = self.p["threshold"]
        S = sum(scores.values())
        loo = [S - s for s in scores.values()]
        if len(scores) >= 2 and S >= t and min(loo) >= t:
            dec = SAME
        elif len(scores) >= 2 and S <= -t and max(loo) <= -t:
            dec = NEW
        else:
            dec = UNKNOWN_D
        return dec, S, scores, dists

    def decide(self, query_evs, groups, ctx):
        per = []
        for gid, units in groups:
            best = None
            for u in units:
                dec, S, sc, ds = self.unit_decision(query_evs, u, ctx)
                rank = {SAME: 2, UNKNOWN_D: 1, NEW: 0}[dec]
                if best is None or (rank, S) > (best[0], best[2]):
                    best = (rank, dec, S, u.id, sc, ds)
            per.append((gid, best))
        same = [(g, b) for g, b in per if b[1] == SAME]
        unk = [(g, b) for g, b in per if b[1] == UNKNOWN_D]
        detail = {g: {"decision": b[1], "score": round(b[2], 3), "via_unit": b[3],
                      "kind_scores": {k: round(v, 3) for k, v in b[4].items()}} for g, b in per}
        if len(same) == 1:
            g, b = same[0]
            return {"decision": SAME, "group": g, "unit": b[3], "possible": [], "detail": detail}
        if len(same) > 1:
            return {"decision": UNKNOWN_D, "group": None, "unit": None,
                    "possible": [g for g, _ in sorted(same, key=lambda x: -x[1][2])], "detail": detail}
        if unk:
            unk.sort(key=lambda x: -x[1][2])
            return {"decision": UNKNOWN_D, "group": None, "unit": None,
                    "possible": [g for g, _ in unk], "detail": detail}
        return {"decision": NEW, "group": None, "unit": None, "possible": [], "detail": detail}


# ---- baselines (trivial strategies; same interface, used for comparison only)
@register("recognition")
class AlwaysNew(Mechanism):
    NAME = "baseline_always_new"
    VERSION = "1"
    STATUS = "BASELINE"
    ASSUMPTIONS = ["Every non-continuous candidate is declared a new thing."]

    def decide(self, query_evs, groups, ctx):
        return {"decision": NEW, "group": None, "unit": None, "possible": [], "detail": {}}


@register("recognition")
class NearestLocation(Mechanism):
    NAME = "baseline_nearest_location"
    VERSION = "1"
    STATUS = "BASELINE"
    ASSUMPTIONS = ["Always SAME as the group whose recorded location is nearest (location-only shortcut)."]

    def decide(self, query_evs, groups, ctx):
        k = "location_self_frame"
        best = None
        for gid, units in groups:
            for u in units:
                d = _min_distances(query_evs, u, ctx["extractors"], [k]).get(k)
                if d is not None and (best is None or d < best[0]):
                    best = (d, gid, u.id)
        if best is None:
            return {"decision": NEW, "group": None, "unit": None, "possible": [], "detail": {}}
        return {"decision": SAME, "group": best[1], "unit": best[2], "possible": [], "detail": {}}


@register("recognition")
class MostRecent(Mechanism):
    NAME = "baseline_most_recent"
    VERSION = "1"
    STATUS = "BASELINE"
    ASSUMPTIONS = ["Always SAME as the most recently seen group."]

    def decide(self, query_evs, groups, ctx):
        best = None
        for gid, units in groups:
            for u in units:
                if best is None or u.last_seen > best[0]:
                    best = (u.last_seen, gid, u.id)
        if best is None:
            return {"decision": NEW, "group": None, "unit": None, "possible": [], "detail": {}}
        return {"decision": SAME, "group": best[1], "unit": best[2], "possible": [], "detail": {}}


@register("recognition")
class RandomExisting(Mechanism):
    NAME = "baseline_random_existing"
    VERSION = "1"
    STATUS = "BASELINE"
    DEFAULTS = {"seed": 0}
    ASSUMPTIONS = ["Always SAME as a uniformly random existing group."]

    def __init__(self, **p):
        super().__init__(**p)
        self.rng = np.random.default_rng(self.p["seed"])

    def decide(self, query_evs, groups, ctx):
        if not groups:
            return {"decision": NEW, "group": None, "unit": None, "possible": [], "detail": {}}
        gid, units = groups[self.rng.integers(len(groups))]
        return {"decision": SAME, "group": gid, "unit": units[0].id, "possible": [], "detail": {}}


# --------------------------------------------------------------- revision
@register("revision")
class YoungUnitReevaluation(Mechanism):
    NAME = "young_unit_reevaluation"
    VERSION = "1"
    DEFAULTS = {"max_records": 12}
    ASSUMPTIONS = [
        "Only revision exercised (Q9 kept minimal): a unit that was created as NEW or PROVISIONAL and "
        "has not yet been linked to an older identity is re-compared, using ALL its records so far, "
        "against older groups each time it gains a continuity record, until it has max_records records.",
        "Outcome SAME -> a new same_identity inference (nothing is erased; both units remain). "
        "Outcome changes of 'possibly_same_as' -> a new inference superseding the previous one.",
        "Groups ever observed on the same tick as the unit are excluded (believed distinct).",
        "No split detection, no artefact classification.",
    ]


@register("revision")
class NoRevision(Mechanism):
    NAME = "no_revision"
    VERSION = "1"
    STATUS = "BASELINE"
    ASSUMPTIONS = ["Identity decisions are never revisited."]
