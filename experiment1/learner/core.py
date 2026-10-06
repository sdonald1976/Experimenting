"""The learner. Receives ONLY: calibration (once) and per-tick observations
{tick, left, right, self_motion}. Emits per-tick exports. Never imports or
receives anything from the harness.

Required architecture implemented here (not experimental):
  - observation store separate from inference store, provenance on every inference
  - persistent units created on first distinguishable encounter (Q4)
  - experience records kept separately per unit, addressable (Q7, Q21)
  - provisional identities with possibly_same_as (Q4a), UNKNOWN as a state (Q8)
  - revisions as new inferences, history never erased (Q9, Q11); no deletion (Q10)
  - local updates with a mutation audit (Q6e)
Everything else is delegated to registered EXPERIMENTAL mechanisms.
"""
from __future__ import annotations

import json

import numpy as np

from . import registry
from .mechanisms import evidence as _ev  # noqa: F401  (registers classes)
from .mechanisms import identity as _id  # noqa: F401
from .mechanisms import perception as _pc  # noqa: F401
from .mechanisms.identity import NEW, SAME, UNKNOWN_D
from .mechanisms.perception import cam_to_world, points_from_disparity
from .stores import BELIEVED, UNKNOWN, InferenceStore, ObservationStore

OBS_KEYS = {"tick", "left", "right", "self_motion"}
CALIB_KEYS = {"width", "height", "fx", "fy", "cx", "cy", "baseline", "mount_pitch"}

ALLOWED_TOUCH = {"creation", "continuity", "recognition_link", "co_visible", "usefulness",
                 "identity_revision"}


class Record:
    __slots__ = ("id", "unit", "tick", "obs", "link", "segment", "ev", "inf", "npix")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class Unit:
    def __init__(self, uid, tick, decision, inf_id):
        self.id = uid
        self.created_tick = tick
        self.decision = decision          # "NEW" | "PROVISIONAL"
        self.creation_inf = inf_id
        self.records: list[Record] = []
        self.segment = 0
        self.W: dict[str, list] = {}
        self.B: dict[str, list] = {}
        self._ws: dict = {}
        self._bs: dict = {}
        self.co_visible: set[int] = set()
        self.possible: list[int] = []     # current possibly_same_as (group ids)
        self.open = decision == "PROVISIONAL"
        self.linked_to_older = False
        self.last_seen = tick
        self.last_pix = None

    def add_w(self, k, d):
        self.W.setdefault(k, []).append(d)
        self._ws.pop(k, None)

    def add_b(self, k, d):
        self.B.setdefault(k, []).append(d)
        self._bs.pop(k, None)

    def W_sorted(self, k):
        if k not in self._ws:
            self._ws[k] = np.sort(np.asarray(self.W.get(k, []), float))
        return self._ws[k]

    def B_sorted(self, k):
        if k not in self._bs:
            self._bs[k] = np.sort(np.asarray(self.B.get(k, []), float))
        return self._bs[k]


class Learner:
    def __init__(self, calib: dict, config: dict):
        if set(calib) != CALIB_KEYS:
            raise ValueError(f"calibration keys {set(calib)} != {CALIB_KEYS}")
        self.calib = dict(calib)
        self.config = config
        self.mech = {s: registry.build(s, config[s]) for s in
                     ("self_motion", "stereo", "candidates", "continuity", "sampling",
                      "usefulness", "recognition", "revision")}
        self.extractors = {e["name"]: registry.build("evidence", e) for e in config["evidence"]}
        self.obs = ObservationStore()
        self.inf = InferenceStore()
        self.units: dict[int, Unit] = {}
        self.parent: dict[int, int] = {}
        self.next_uid = 1
        self.active: dict[int, int] = {}   # unit id -> label this tick
        self.touch_log: dict[str, int] = {}
        self.locality_violations = 0
        self.decision_counts = {SAME: 0, NEW: 0, UNKNOWN_D: 0}

    # ------------------------------------------------------------ helpers
    def describe(self):
        d = {s: m.describe() for s, m in self.mech.items()}
        d["evidence"] = [e.describe() for e in self.extractors.values()]
        return d

    def find(self, u):
        while self.parent.get(u, u) != u:
            u = self.parent[u]
        return u

    def _touch(self, unit, reason):
        if reason not in ALLOWED_TOUCH:
            raise ValueError(reason)
        if unit.id not in self._involved:
            self.locality_violations += 1
        self.touch_log[reason] = self.touch_log.get(reason, 0) + 1

    def _new_unit(self, tick, decision, inf_id):
        u = Unit(self.next_uid, tick, decision, inf_id)
        self.next_uid += 1
        self.units[u.id] = u
        self.parent[u.id] = u.id
        self._involved.add(u.id)
        self._touch(u, "creation")
        return u

    def _groups(self, exclude: set[int], older_than=None):
        g: dict[int, list] = {}
        for u in self.units.values():
            if u.id in exclude or (older_than is not None and u.created_tick >= older_than):
                continue
            if not u.records:
                continue
            g.setdefault(self.find(u.id), []).append(u)
        bad = {self.find(x) for x in exclude}
        return [(gid, us) for gid, us in g.items() if gid not in bad]

    def _record(self, unit, tick, oid, link, ev, region_inf, link_inf, npix):
        rec_inf = self.inf.add(tick, "experience_record", "core:record", [oid, region_inf, link_inf],
                               BELIEVED, unit=unit.id, link=link)
        r = Record(id=f"exp:{unit.id}:{len(unit.records)}", unit=unit.id, tick=tick, obs=oid, link=link,
                   segment=unit.segment, ev=ev, inf=rec_inf.id, npix=npix)
        unit.records.append(r)
        return r

    # ------------------------------------------------------------ main step
    def step(self, obs: dict) -> dict:
        if set(obs) != OBS_KEYS:
            raise ValueError(f"observation keys {set(obs)} != {OBS_KEYS}")
        t = int(obs["tick"])
        self._involved: set[int] = set()
        oid = self.obs.add(obs)
        o = self.obs.get(oid)
        M = self.mech
        cal = self.calib

        pose = M["self_motion"].step(o["self_motion"])
        pose_inf = self.inf.add(t, "pose_estimate", M["self_motion"].tag, [oid], BELIEVED, **pose)
        disp, valid = M["stereo"].compute(o["left"], o["right"])
        disp_inf = self.inf.add(t, "disparity_map", M["stereo"].tag, [oid], BELIEVED,
                                stored=False, recomputable_from=oid, n_valid=int(valid.sum()))
        labels, chroma, bright = M["candidates"].extract(o["left"], disp, valid)
        n_reg = int(labels.max())
        cam_pts = points_from_disparity(np.where(valid, disp, np.nan), cal)
        ctx = {"world_points": cam_to_world(cam_pts, cal, pose), "cam_points": cam_pts,
               "chroma": chroma, "bright": bright, "gray": o["left"].astype(np.float32).mean(axis=2),
               "fx": cal["fx"]}
        flat = labels.ravel()
        order = np.argsort(flat, kind="stable")
        bounds = np.searchsorted(flat[order], np.arange(n_reg + 2))
        region_pix = {c: order[bounds[c]:bounds[c + 1]] for c in range(1, n_reg + 1)}
        region_inf = {c: self.inf.add(t, "candidate_region", M["candidates"].tag, [disp_inf.id, oid],
                                      BELIEVED, npix=int(len(region_pix[c]))).id
                      for c in range(1, n_reg + 1)}

        # ---- continuity
        prev = [(uid, self.units[uid].last_pix) for uid in self.active]
        links = M["continuity"].link(prev, labels, n_reg, o["self_motion"]["d_yaw"], cal["fx"])
        new_active: dict[int, int] = {}
        claims: dict[int, dict] = {}
        due = M["sampling"].due(t)
        made: list[Record] = []
        ev_cache: dict[int, dict] = {}

        def evidence(c):
            if c not in ev_cache:
                ev_cache[c] = {k: ex.extract(ctx, region_pix[c]) for k, ex in self.extractors.items()}
            return ev_cache[c]

        for c, (uid, iou) in links.items():
            u = self.units[uid]
            self._involved.add(uid)
            self._touch(u, "continuity")
            link_inf = self.inf.add(t, "continuity_link", M["continuity"].tag,
                                    [region_inf[c], u.records[-1].inf if u.records else u.creation_inf],
                                    BELIEVED, unit=uid, iou=round(iou, 3))
            new_active[uid] = c
            u.last_seen, u.last_pix = t, region_pix[c]
            claims[uid] = {"link": "continuity"}
            if due:
                made.append(self._record(u, t, oid, "continuity", evidence(c), region_inf[c],
                                         link_inf.id, len(region_pix[c])))

        # ---- unlinked candidates: recognition (largest first)
        unlinked = sorted((c for c in range(1, n_reg + 1) if c not in links),
                          key=lambda c: -len(region_pix[c]))
        rctx = {"extractors": self.extractors, "usefulness": M["usefulness"],
                "kinds": list(self.extractors)}
        for c in unlinked:
            ev = evidence(c)
            groups = self._groups(exclude=set(new_active))
            res = M["recognition"].decide([ev], groups, rctx)
            self.decision_counts[res["decision"]] += 1
            ep = UNKNOWN if res["decision"] == UNKNOWN_D else BELIEVED
            dec_inf = self.inf.add(t, "recognition_decision", M["recognition"].tag, [region_inf[c]],
                                   ep, decision=res["decision"], group=res["group"],
                                   possible=res["possible"][:20], detail=_compact(res["detail"]))
            if res["decision"] == SAME:
                u = self.units[res["unit"]]
                if u.id in new_active:  # cannot be two regions at once
                    res = {"decision": UNKNOWN_D, "possible": [self.find(u.id)]}
                else:
                    self._involved.add(u.id)
                    self._touch(u, "recognition_link")
                    u.segment += 1  # recognition never extends continuity authority
                    u.last_seen, u.last_pix = t, region_pix[c]
                    new_active[u.id] = c
                    claims[u.id] = {"link": "recognition"}
                    made.append(self._record(u, t, oid, "recognition", ev, region_inf[c], dec_inf.id,
                                             len(region_pix[c])))
                    continue
            decision = "NEW" if res["decision"] == NEW else "PROVISIONAL"
            u = self._new_unit(t, decision, dec_inf.id)
            if decision == "PROVISIONAL":
                u.possible = list(res["possible"])
                self.inf.add(t, "possibly_same_as", M["recognition"].tag, [dec_inf.id], UNKNOWN,
                             unit=u.id, possible=u.possible[:20])
            u.last_pix = region_pix[c]
            new_active[u.id] = c
            claims[u.id] = {"link": "creation"}
            made.append(self._record(u, t, oid, "creation", ev, region_inf[c], dec_inf.id,
                                     len(region_pix[c])))

        # ---- co-visibility (simultaneously observed regions are believed distinct)
        act = list(new_active)
        for uid in act:
            u = self.units[uid]
            before = len(u.co_visible)
            u.co_visible.update(x for x in act if x != uid)
            if len(u.co_visible) != before:
                self._touch(u, "co_visible")

        # ---- usefulness samples (continuity + simultaneous distinction only)
        by_unit = {r.unit: r for r in made}
        for r in made:
            u = self.units[r.unit]
            if r.link == "continuity":
                same_seg = [x for x in u.records[:-1] if x.segment == r.segment][-5:]
                for x in same_seg:
                    for k, ex in self.extractors.items():
                        if r.ev.get(k) is not None and x.ev.get(k) is not None:
                            u.add_w(k, ex.distance(r.ev[k], x.ev[k]))
                if same_seg:
                    self._touch(u, "usefulness")
        ids = sorted(by_unit)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ra, rb = by_unit[a], by_unit[b]
                for k, ex in self.extractors.items():
                    if ra.ev.get(k) is not None and rb.ev.get(k) is not None:
                        d = ex.distance(ra.ev[k], rb.ev[k])
                        self.units[a].add_b(k, d)
                        self.units[b].add_b(k, d)
                self._touch(self.units[a], "usefulness")
                self._touch(self.units[b], "usefulness")

        # ---- revision of young identities
        identity_events = []
        if M["revision"].NAME == "young_unit_reevaluation":
            for r in made:
                u = self.units[r.unit]
                if r.link != "continuity" or u.linked_to_older or u.decision not in ("NEW", "PROVISIONAL"):
                    continue
                if len(u.records) > M["revision"].p["max_records"]:
                    continue
                groups = self._groups(exclude=u.co_visible | {u.id}, older_than=u.created_tick)
                if not groups:
                    continue
                res = M["recognition"].decide([x.ev for x in u.records], groups, rctx)
                if res["decision"] == SAME:
                    other = self.units[res["unit"]]
                    self._involved.update({u.id, other.id})
                    inf = self.inf.add(t, "same_identity", M["revision"].tag,
                                       [u.creation_inf] + [x.inf for x in u.records] + [other.records[-1].inf],
                                       BELIEVED, units=[u.id, other.id], detail=_compact(res["detail"]))
                    self.parent[self.find(u.id)] = self.find(other.id)
                    u.linked_to_older, u.open = True, False
                    self._touch(u, "identity_revision")
                    self._touch(other, "identity_revision")
                    identity_events.append(("same", u.id, other.id, inf.id))
                elif res["decision"] == UNKNOWN_D and res["possible"] != u.possible:
                    self._involved.add(u.id)
                    inf = self.inf.add(t, "possibly_same_as", M["revision"].tag,
                                       [u.creation_inf] + [x.inf for x in u.records], UNKNOWN,
                                       unit=u.id, possible=res["possible"][:20], supersedes_previous=True)
                    u.possible, u.open = list(res["possible"]), True
                    self._touch(u, "identity_revision")
                    identity_events.append(("possibly", u.id, u.possible[:5], inf.id))
                elif res["decision"] == NEW and u.open:
                    self._involved.add(u.id)
                    inf = self.inf.add(t, "resolved_distinct", M["revision"].tag,
                                       [u.creation_inf] + [x.inf for x in u.records], BELIEVED, unit=u.id)
                    u.possible, u.open = [], False
                    self._touch(u, "identity_revision")
                    identity_events.append(("resolved_distinct", u.id, None, inf.id))

        self.active = new_active
        out_labels = np.zeros(labels.shape, np.uint16)
        for uid, c in new_active.items():
            out_labels.ravel()[region_pix[c]] = uid
            claims[uid]["epistemic"] = UNKNOWN if self.units[uid].open else BELIEVED
            claims[uid]["group"] = self.find(uid)
            if self.units[uid].open:
                claims[uid]["possible"] = self.units[uid].possible[:5]
        return {"tick": t, "labels": out_labels, "claims": claims,
                "new_units": [u for u in claims if self.units[u].created_tick == t],
                "identity_events": identity_events,
                "canonical": {uid: self.find(uid) for uid in claims}}

    # ------------------------------------------------------------ end of run
    def final_report(self) -> dict:
        units = self.units.values()
        return {
            "mechanisms": self.describe(),
            "observation_hashes": self.obs.hashes,
            "inference_audit": self.inf.audit(),
            "touch_log": self.touch_log,
            "locality_violations": self.locality_violations,
            "decision_counts": self.decision_counts,
            "n_units": len(self.units),
            "n_units_provisional_open": sum(u.open for u in units),
            "n_records": sum(len(u.records) for u in units),
            "canonical": {u.id: self.find(u.id) for u in units},
            "unit_created": {u.id: [u.created_tick, u.decision] for u in units},
        }

    def state_dump(self) -> str:
        """Everything the learner believes, as text, for the harness canary scan."""
        recs = [{"id": r.id, "tick": r.tick, "kind": r.kind, "mech": r.mechanism,
                 "inputs": r.inputs, "ep": r.epistemic, "payload": r.payload} for r in self.inf.records]
        us = [{"id": u.id, "created": u.created_tick, "decision": u.decision,
               "records": [x.id for x in u.records], "possible": u.possible} for u in self.units.values()]
        return json.dumps({"inferences": recs, "units": us, "config": self.config}, default=str)


def _compact(detail):
    if not detail:
        return {}
    items = sorted(detail.items(), key=lambda kv: -kv[1].get("score", 0))[:5]
    return {str(k): v for k, v in items}
