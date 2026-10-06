"""Scoring. HARNESS-SIDE ONLY: compares learner exports with ground truth.
Nothing computed here is ever sent to the learner.

All thresholds below are pre-registered harness parameters (see PREREGISTRATION.md).
"""
from __future__ import annotations

import numpy as np

SCORING_PARAMS = {
    "claim_min_overlap_px": 30,         # unit must cover >= this many px of the thing
    "claim_min_purity": 0.5,            # >= this share of the unit's px (that tick) on the thing
    "visible_min_px": 150,              # thing counts as visible
    "assoc_min_total_px": 200,          # unit needs this much history to count as associated
    "assoc_min_purity": 0.5,            # majority share for association
    "viewpoint_tolerance_deg": 15.0,    # Q15
    "contamination_purity": 0.8,
    "touches_targets_share": 0.2,
    "coverage_claim_share": 0.5,
    "spurious_max_ticks": 2,
}
GT_OFFSET = 1  # gt ids start at -1 (none)


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def members(self, root, universe):
        return [u for u in universe if self.find(u) == root]


class VariantScorer:
    def __init__(self, thing_ids: set[int]):
        self.thing_ids = thing_ids          # target things (non-structure), grows as novel ones appear
        self.counts: dict[int, dict[int, int]] = {}
        self.ticks_seen: dict[int, int] = {}
        self.uf = UF()
        self.last_claims: dict = {}
        self.per_tick_cov = []              # (tick, gt, gt_px, {uid: px})
        self.trials = []

    def update(self, export, gt):
        lab = export["labels"].astype(np.int64)
        for ev in export["identity_events"]:
            if ev[0] == "same":
                self.uf.union(ev[1], ev[2])
        key = lab * 256 + (gt.astype(np.int64) + GT_OFFSET)
        k, n = np.unique(key[lab > 0], return_counts=True)
        for kk, nn in zip(k, n):
            u, g = int(kk // 256), int(kk % 256) - GT_OFFSET
            d = self.counts.setdefault(u, {})
            d[g] = d.get(g, 0) + int(nn)
        for u in export["claims"]:
            self.ticks_seen[u] = self.ticks_seen.get(u, 0) + 1
        self.last_claims = export["claims"]
        for g in self.thing_ids:
            m = gt == g
            gp = int(m.sum())
            if gp >= SCORING_PARAMS["visible_min_px"]:
                us, cs = np.unique(lab[m & (lab > 0)], return_counts=True)
                self.per_tick_cov.append((export["tick"], g, gp, dict(zip(us.tolist(), cs.tolist()))))

    def association_snapshot(self):
        P = SCORING_PARAMS
        snap = {}
        for u, d in self.counts.items():
            tot = sum(d.values())
            g, n = max(d.items(), key=lambda kv: kv[1])
            if tot >= P["assoc_min_total_px"] and n / tot >= P["assoc_min_purity"]:
                snap[u] = g
        return snap

    def outcome(self, export, gt, g, snapshot):
        """Classify the learner's belief about thing g at this tick."""
        P = SCORING_PARAMS
        lab = export["labels"]
        m = gt == g
        us, ov = np.unique(lab[m & (lab > 0)], return_counts=True)
        if len(us) == 0:
            return {"outcome": "not_detected"}
        tot = {u: int((lab == u).sum()) for u in us}
        cands = [(int(o), int(u)) for u, o in zip(us, ov)
                 if o >= P["claim_min_overlap_px"] and o / tot[u] >= P["claim_min_purity"]]
        if not cands:
            return {"outcome": "not_detected"}
        _, p = max(cands)
        claim = export["claims"][p]
        universe = list(self.counts.keys())
        root = self.uf.find(p)
        members = self.uf.members(root, universe) or [p]
        prior_g = [u for u in members if snapshot.get(u) == g]
        prior_other = [u for u in members if u in snapshot and snapshot[u] != g]
        res = {"unit": p, "group_size": len(members), "link": claim.get("link")}
        if claim.get("epistemic") == "UNKNOWN":
            inc = False
            for q in claim.get("possible", []):
                r = self.uf.find(q)
                if any(snapshot.get(u) == g for u in self.uf.members(r, universe) + [q]):
                    inc = True
            res.update(outcome="unknown", unknown_includes_correct=inc)
        elif prior_other:
            res.update(outcome="false_same", contaminated=bool(prior_g),
                       wrong_thing=int(snapshot[prior_other[0]]))
        elif prior_g:
            res["outcome"] = "correct_same"
        else:
            res["outcome"] = "new"
        return res

    def discovery(self, structure_ids: set[int], final_canonical: dict):
        P = SCORING_PARAMS
        units = {}
        for u, d in self.counts.items():
            tot = sum(d.values())
            g, n = max(d.items(), key=lambda kv: kv[1])
            tgt = sum(v for k, v in d.items() if k in self.thing_ids)
            units[u] = {"total": tot, "major": g, "purity": n / tot, "target_share": tgt / tot}
        touching = {u: v for u, v in units.items() if v["target_share"] >= P["touches_targets_share"]}
        tp = sum(v["total"] for v in touching.values())
        cp = sum(v["total"] for v in touching.values() if v["purity"] < P["contamination_purity"])
        # fragmentation: believed-identity groups (final) whose pixels are mostly on thing g
        uf = UF()
        for u, r in final_canonical.items():
            uf.union(int(u), int(r))
        grp: dict[int, dict[int, int]] = {}
        for u, d in self.counts.items():
            r = uf.find(u)
            gd = grp.setdefault(r, {})
            for k, v in d.items():
                gd[k] = gd.get(k, 0) + v
        frag, ufrag = {}, {}
        for r, d in grp.items():
            g, n = max(d.items(), key=lambda kv: kv[1])
            if g in self.thing_ids and n / sum(d.values()) >= P["assoc_min_purity"]:
                frag[g] = frag.get(g, 0) + 1
        for u, v in units.items():
            if v["major"] in self.thing_ids and v["purity"] >= P["assoc_min_purity"]:
                ufrag[v["major"]] = ufrag.get(v["major"], 0) + 1
        # coverage with end-of-run association
        assoc = {u: v["major"] for u, v in units.items() if v["purity"] >= P["assoc_min_purity"]}
        cov_hits, cov_n = {}, {}
        for _, g, gp, d in self.per_tick_cov:
            claimed = sum(px for u, px in d.items() if assoc.get(u) == g)
            cov_n[g] = cov_n.get(g, 0) + 1
            cov_hits[g] = cov_hits.get(g, 0) + (claimed / gp >= P["coverage_claim_share"])
        return {
            "n_units": len(units),
            "n_units_touching_targets": len(touching),
            "contamination_px_rate": cp / tp if tp else None,
            "fragmentation_groups_per_thing": frag,
            "fragmentation_units_per_thing": ufrag,
            "coverage_per_thing": {g: cov_hits[g] / cov_n[g] for g in cov_n},
            "spurious_units": sum(1 for u in units if self.ticks_seen.get(u, 0) <= P["spurious_max_ticks"]),
            "units_major_structure": sum(1 for v in units.values() if v["major"] in structure_ids),
            "units_major_target": sum(1 for v in units.values() if v["major"] in self.thing_ids),
        }
