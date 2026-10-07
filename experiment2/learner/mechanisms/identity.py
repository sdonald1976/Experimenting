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


class GroupDecision(Mechanism):
    """Base: aggregates per-unit decisions across believed-identity groups (as in Experiment 1)."""

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


# ============================================================ Experiment 2
@register("usefulness")
class WindowedAUCUsefulness(Mechanism):
    """Experiment 2. Same sample construction as tail_ratio_matched (best-match statistic), but:
    votes and usefulness use a RECENCY WINDOW of samples (full history kept), and an explicit,
    logged usefulness w(u,k) is learned per unit and evidence kind."""
    NAME = "windowed_auc"
    VERSION = "2"
    DEFAULTS = {"w_min_gap_ticks": 9, "w_window": 30, "b_window": 60, "pseudo_count": 1.0,
                "hard_quantile": 0.25, "shrink_n0": 10.0, "vote": "nearest"}
    ASSUMPTIONS = [
        "Samples per (unit, kind), each stored with tick and producing record ids:",
        "  W = best-match distance from a new CONTINUITY record to the unit's earlier records in the same "
        "uninterrupted segment that are >= w_min_gap_ticks older (Q6b signal 1).",
        "  B = best-match distance from a record of ANOTHER unit observed on the same tick to this unit's "
        "earlier records (Q6b signal 2). Simultaneously observed regions are assumed to be different things.",
        "No recognition- or revision-derived identity ever creates a sample (no feedback path).",
        "Only the most recent w_window W and b_window B samples are used (recency window - the assumption "
        "that lets beliefs be revised). All older samples remain stored.",
        "Vote for distance d ('nearest', v2 after DEV diagnosis): v = (2*P(|d-w| < |d-b|) - 1) * n/(n + pseudo_count), "
        "over windowed samples w in W, b in B (ties count 1/2), n = min(#W, #B); 0 with no samples. "
        "'Is d closer to what same-thing distances have been, or to what different-thing distances have been?' "
        "v1 ('tails': P(W >= d) - P(B <= d)) returned ~0 whenever d fell between the two distributions, e.g. "
        "revisit location drift larger than within-look jitter but far smaller than distances to other things.",
        "Usefulness w(u,k) = shrink * max(0, 2*AUC - 1), AUC = P(w < b) between windowed W and the hardest "
        "hard_quantile of windowed B; shrink = n/(n + shrink_n0), n = min(#W, #B). Starts at 0 (uncommitted).",
    ]

    def within(self, unit, rec, extractors):
        gap = self.p["w_min_gap_ticks"]
        prior = [x for x in unit.records[:-1] if x.segment == rec.segment and rec.tick - x.tick >= gap]
        return _best_with_id(rec, prior, extractors)

    def between(self, unit, other_rec, extractors, tick):
        prior = [x for x in unit.records if x.tick < tick]
        return _best_with_id(other_rec, prior, extractors)

    def _win(self, unit, k):
        W = np.array([s[0] for s in unit.W.get(k, [])[-self.p["w_window"]:]], float)
        B = np.array([s[0] for s in unit.B.get(k, [])[-self.p["b_window"]:]], float)
        return W, B

    def vote(self, unit, k, d):
        W, B = self._win(unit, k)
        a = self.p["pseudo_count"]
        if self.p["vote"] == "tails":
            return float((np.sum(W >= d) + a) / (len(W) + 2 * a) - (np.sum(B <= d) + a) / (len(B) + 2 * a))
        if len(W) == 0 or len(B) == 0:
            return 0.0
        dw, db = np.abs(d - W)[:, None], np.abs(d - B)[None, :]
        p = ((dw < db).sum() + 0.5 * (dw == db).sum()) / (dw.size * db.size)
        n = min(len(W), len(B))
        return float((2 * p - 1) * n / (n + a))

    def usefulness(self, unit, k):
        W, B = self._win(unit, k)
        if len(W) == 0 or len(B) == 0:
            return 0.0
        hard = np.sort(B)[: max(1, int(np.ceil(self.p["hard_quantile"] * len(B))))]
        auc = (np.sum(W[:, None] < hard[None, :]) + 0.5 * np.sum(W[:, None] == hard[None, :])) / (len(W) * len(hard))
        n = min(len(W), len(B))
        return float(n / (n + self.p["shrink_n0"]) * max(0.0, 2 * auc - 1))


def _best_with_id(rec, prior, extractors):
    out = {}
    for k, ex in extractors.items():
        a = rec.ev.get(k)
        if a is None:
            continue
        best = None
        for x in prior:
            if x.ev.get(k) is None:
                continue
            d = ex.distance(a, x.ev[k])
            if best is None or d < best[0]:
                best = (d, x.id)
        if best is not None:
            out[k] = best
    return out


@register("recognition")
class WeightedVote(GroupDecision):
    """Experiment 2 shared decision rule. Variants differ ONLY in the weight source."""
    NAME = "weighted_vote"
    VERSION = "1"
    DEFAULTS = {"threshold": 0.5, "weights": "learned", "min_mass": 0.5,
                "fixed_weights": {"metric_local_patterns_highpass": 0.35, "chroma_distribution": 0.30,
                                  "spatial_extent": 0.20, "location_self_frame": 0.15},
                "max_possible_listed": 5}
    ASSUMPTIONS = [
        "Per kind: d = best-match distance between the query and the unit's records; vote v = usefulness.vote(d).",
        "Weights: 'learned' = the unit's own learned usefulness w(u,k); 'fixed' = fixed_weights (same for all "
        "units, never changes); 'equal' = 1.",
        "S = sum(w v)/sum(w). SAME iff S >= threshold and S >= threshold with ANY one kind removed (Q6: no single "
        "kind defines identity); NEW mirrors at -threshold; else UNKNOWN. Same rule for every variant.",
        "'learned' only: total weight < min_mass -> UNKNOWN (no learned basis yet).",
        "Several query records (revision): each scored separately; per-kind median vote.",
        "Across units: exactly one SAME group -> SAME; several -> UNKNOWN; none SAME but some UNKNOWN -> UNKNOWN; "
        "all NEW -> NEW (as Experiment 1).",
    ]

    def _weights(self, unit, kinds, U):
        mode = self.p["weights"]
        if mode == "learned":
            return {k: U.usefulness(unit, k) for k in kinds}
        if mode == "fixed":
            return {k: float(self.p["fixed_weights"][k]) for k in kinds}
        return {k: 1.0 for k in kinds}

    def unit_decision(self, query_evs, unit, ctx):
        U = ctx["usefulness"]
        per_votes: dict[str, list] = {}
        dists: dict[str, list] = {}
        for q in query_evs:
            ds = _min_distances([q], unit, ctx["extractors"], ctx["kinds"])
            for k, d in ds.items():
                per_votes.setdefault(k, []).append(U.vote(unit, k, d))
                dists.setdefault(k, []).append(d)
        votes = {k: float(np.median(v)) for k, v in per_votes.items()}
        w = self._weights(unit, list(votes), U)
        t = self.p["threshold"]
        mass = sum(w.values())

        def S_of(ks):
            m = sum(w[k] for k in ks)
            return sum(w[k] * votes[k] for k in ks) / m if m > 0 else 0.0

        ks = list(votes)
        S = S_of(ks)
        loo = [S_of([x for x in ks if x != k]) for k in ks]
        if len(ks) < 2 or (self.p["weights"] == "learned" and mass < self.p["min_mass"]):
            dec = UNKNOWN_D
        elif S >= t and min(loo) >= t:
            dec = SAME
        elif S <= -t and max(loo) <= -t:
            dec = NEW
        else:
            dec = UNKNOWN_D
        scores = {k: round(w[k] * votes[k], 4) for k in ks}
        return dec, S, scores, {k: float(np.median(v)) for k, v in dists.items()}
