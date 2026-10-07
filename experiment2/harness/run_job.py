"""Run one Experiment 2 seed: render, feed every learner variant (own process, pipe), score,
analyse learned usefulness against oracle discriminability, audit, save JSON.

python -m harness.run_job --seed 0 --out results/dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np

from learner import core as _learner_core  # noqa: F401  (registers mechanisms for analysis distances)
from learner import registry
from learner.config import VARIANTS
from learner.process import learner_main

from .render import Renderer
from .scene import (CANARY, PALETTE, SCENE_PARAMS, World, _random_size, arc_pos, build_schedule,
                    footprint, object_azimuth)
from .scoring import SCORING_PARAMS, VariantScorer

SENSOR_PARAMS = {
    "width": 256, "height": 192, "hfov_deg": 60.0, "baseline_m": 0.12,
    "mount_pitch_deg": -28.0, "cam_height_m": 1.0, "pixel_noise_sigma": 3.0,
    "self_motion_noise_rel": 0.03, "self_motion_noise_abs_m": 0.003, "self_motion_noise_abs_rad": 0.002,
}
ANALYSIS_PARAMS = {"map_min_records": 10, "map_min_share": 0.8, "record_min_purity": 0.5,
                   "oracle_gap_ticks": 9, "oracle_max_queries": 20, "oracle_max_pool": 60, "eval_observe_index": 9}
KINDS = ["spatial_extent", "chroma_distribution", "metric_local_patterns_highpass", "location_self_frame"]


def obs_hash(obs):
    h = hashlib.sha256()
    h.update(obs["left"].tobytes())
    h.update(obs["right"].tobytes())
    h.update(repr(sorted(obs["self_motion"].items())).encode())
    return h.hexdigest()


def validate_obs(obs, calib):
    assert set(obs) == {"tick", "left", "right", "self_motion"}, set(obs)
    for k in ("left", "right"):
        assert obs[k].dtype == np.uint8 and obs[k].shape == (calib["height"], calib["width"], 3)
    assert set(obs["self_motion"]) == {"d_forward", "d_right", "d_yaw"}


def true_delta(prev, cur):
    psi0 = prev["psi"]
    r = np.array([np.cos(psi0), -np.sin(psi0)])
    f = np.array([np.sin(psi0), np.cos(psi0)])
    dp = cur["pos"] - prev["pos"]
    return float(dp @ f), float(dp @ r), float(cur["psi"] - psi0)


def _face_experienced(t, ret_cam, rng, experienced_az):
    az = np.radians(float(rng.choice(experienced_az.get(t.gt_id) or [0.0])))
    v = ret_cam - t.pos
    t.yaw = float(np.arctan2(v[0], v[1]) - az)


def apply_event(world: World, ev, rng, experienced_az, cycle):
    kind = ev["kind"]
    info = {"kind": kind, "moved": [], "swapped": [], "added": None, "removed": None}
    ret_cam = arc_pos(ev["return_alpha"])
    if kind == "swap":
        a, b = rng.choice(len(world.things), size=2, replace=False)
        ta, tb = world.things[a], world.things[b]
        ta.pos, tb.pos = tb.pos.copy(), ta.pos.copy()
        for t in (ta, tb):
            _face_experienced(t, ret_cam, rng, experienced_az)
        info["swapped"] = [ta.gt_id, tb.gt_id]
    elif kind == "move":
        for t in [world.things[i] for i in rng.permutation(len(world.things))]:
            try:
                t.pos = world.free_position(t.footprint_radius(), exclude=(t,), min_from=t.pos.copy())
            except RuntimeError:
                continue
            _face_experienced(t, ret_cam, rng, experienced_az)
            info["moved"] = [t.gt_id]
            break
    elif kind in ("novel", "replace"):
        pos, avoid = None, None
        if kind == "replace":
            victim = world.things[rng.integers(len(world.things))]
            world.things.remove(victim)
            pos, avoid = victim.pos.copy(), (victim.shape, victim.name.split("_")[-2])
            info["removed"] = victim.gt_id
        placed = None
        for _ in range(400):
            shape = str(rng.choice(["box", "cylinder", "sphere"]))
            col = str(rng.choice(list(PALETTE)))
            if avoid and (shape, col) == avoid:
                continue
            size = _random_size(rng, shape)
            fp = footprint(shape, size)
            if pos is None:
                try:
                    placed = world.free_position(fp)
                except RuntimeError:
                    continue
                break
            if world.is_free(pos, fp):
                placed = pos
                break
        if placed is None:
            raise RuntimeError("could not place new thing")
        t = world.add_thing(shape, col, size, placed, rng.uniform(0, 2 * np.pi), f"NOVEL{cycle}")
        info["added"] = t.gt_id
    return info


def start_slides(world, rng, last_gt, tick):
    """Pick 1-2 currently visible things and slide them in plain view. Harness-only."""
    P = SCENE_PARAMS
    n_ticks = SCENE_PARAMS["observe_ticks_phase2"] - P["slide_start_observe_tick"]
    dist = P["slide_speed_m_per_tick"] * n_ticks
    vis = [t for t in world.things if (last_gt == t.gt_id).sum() >= SCORING_PARAMS["visible_min_px"]]
    rng.shuffle(vis)
    n = int(rng.integers(P["slide_things_per_window"][0], P["slide_things_per_window"][1] + 1))
    slides = []
    for t in vis:
        if len(slides) >= n:
            break
        for _ in range(40):
            ang = rng.uniform(0, 2 * np.pi)
            d = np.array([np.sin(ang), np.cos(ang)])
            end = t.pos + d * dist
            path_ok = np.linalg.norm(end) <= P["placement_radius_m"] + 0.15 and all(
                world.is_free(t.pos + d * dist * f, t.footprint_radius(), exclude=(t,)) for f in (0.33, 0.66, 1.0))
            if path_ok:
                slides.append({"thing": t, "vel": d * P["slide_speed_m_per_tick"], "left": n_ticks, "start": tick})
                break
    return slides


def run(seed, out_dir, variants, dump_state=None):
    t_start = time.time()
    world = World(seed)
    ticks, kinds = build_schedule(world)
    S = SENSOR_PARAMS
    rend = Renderer(S["width"], S["height"], S["hfov_deg"], S["baseline_m"],
                    np.radians(S["mount_pitch_deg"]), S["cam_height_m"], noise_sigma=S["pixel_noise_sigma"])
    calib = rend.calibration()
    noise_rng = np.random.default_rng(30_000 + seed)
    event_rng = np.random.default_rng(40_000 + seed)
    structure_ids = {s.gt_id for s in world.structure}
    T2 = next(i for i, tk in enumerate(ticks) if tk["phase"] == 2)

    ctx = mp.get_context("spawn")
    procs = {}
    for v in variants:
        a, b = ctx.Pipe()
        p = ctx.Process(target=learner_main, args=(b, VARIANTS[v]), daemon=True)
        p.start()
        procs[v] = (p, a)
    for v, (p, a) in procs.items():
        a.send(("calib", calib))
        kind, d = a.recv()
        assert kind == "ready", d

    scorers = {v: VariantScorer({t.gt_id for t in world.things}) for v in variants}
    record_gt = {v: {} for v in variants}
    experienced_az: dict[int, list] = {}
    sent_hashes, cycles = {}, []
    cur, prev_tick, last_gt = None, None, None
    invisible = {}
    slides, slid_log = [], {}           # gt -> list of slide start ticks (phase 2, in view)
    moved_p2 = set()                    # things displaced while unobserved in phase 2
    obs_index = 0
    for i, tk in enumerate(ticks):
        if tk["event"] is not None:
            snap = {v: scorers[v].association_snapshot() for v in variants}
            info = apply_event(world, tk["event"], event_rng, experienced_az, tk["cycle"])
            if tk["phase"] == 2:
                moved_p2.update(info["moved"] + info["swapped"])
            for sc in scorers.values():
                sc.thing_ids |= {t.gt_id for t in world.things}
            cur = {"cycle": tk["cycle"], "phase": tk["phase"], "kind": info["kind"], "info": info, "snap": snap,
                   "unobserved_ok": None, "eval": {}, "az_prior": {g: list(v) for g, v in experienced_az.items()}}
            cycles.append(cur)
            invisible = {}
        if tk["slide_start"]:
            slides = start_slides(world, event_rng, last_gt, i)
            for s_ in slides:
                slid_log.setdefault(s_["thing"].gt_id, []).append(i)
        for s_ in slides:
            if s_["left"] > 0 and s_["thing"] in world.things:
                s_["thing"].pos = s_["thing"].pos + s_["vel"]
                s_["left"] -= 1
        L, R, gt, _ = rend.render_stereo(world.surfaces(), tk["pos"], tk["psi"], noise_rng)
        last_gt = gt
        if tk["event"] is not None:
            touched = info["moved"] + info["swapped"] + [x for x in (info["added"], info["removed"]) if x]
            cur["unobserved_ok"] = bool(all((gt == g).sum() == 0 for g in touched))
        if tk["part"] == "away":
            for t in world.things:
                if (gt == t.gt_id).sum() == 0:
                    invisible[t.gt_id] = True
        for t in world.things:
            if (gt == t.gt_id).sum() >= SCORING_PARAMS["visible_min_px"]:
                experienced_az.setdefault(t.gt_id, []).append(object_azimuth(t, tk["pos"]))
        if prev_tick is None:
            sm = {"d_forward": 0.0, "d_right": 0.0, "d_yaw": 0.0}
        else:
            df, dr, dy = true_delta(prev_tick, tk)
            nz = lambda x, a: float(x + noise_rng.normal(0, S["self_motion_noise_rel"] * abs(x) + a))
            sm = {"d_forward": nz(df, S["self_motion_noise_abs_m"]), "d_right": nz(dr, S["self_motion_noise_abs_m"]),
                  "d_yaw": nz(dy, S["self_motion_noise_abs_rad"])}
        prev_tick = tk
        obs = {"tick": i, "left": L, "right": R, "self_motion": sm}
        validate_obs(obs, calib)
        sent_hashes[f"obs:{i}"] = obs_hash(obs)
        for v, (p, a) in procs.items():
            a.send(("obs", obs))
        exports = {}
        for v, (p, a) in procs.items():
            kind, ex = a.recv()
            if kind != "export":
                raise RuntimeError(f"{v}: {ex}")
            exports[v] = ex
            scorers[v].update(ex, gt)
            for uid, rid in ex["records_made"]:
                vals = gt[ex["labels"] == uid]
                if len(vals):
                    g, n = np.unique(vals, return_counts=True)
                    j = int(np.argmax(n))
                    record_gt[v][rid] = (int(g[j]), float(n[j] / len(vals)))
        obs_index = obs_index + 1 if tk["part"] == "observe" else 0
        if tk["part"] == "observe" and cur is not None and obs_index - 1 == ANALYSIS_PARAMS["eval_observe_index"]:
            for t in world.things:
                g = t.gt_id
                if (gt == g).sum() < SCORING_PARAMS["visible_min_px"]:
                    continue
                inf = cur["info"]
                if g == inf["added"]:
                    ttype = "replace_new" if cur["kind"] == "replace" else "novel"
                elif g in inf["swapped"]:
                    ttype = "swapped"
                elif g in inf["moved"]:
                    ttype = "moved"
                else:
                    ttype = "unmoved"
                seen_vp = None
                if ttype in ("swapped", "moved", "unmoved"):
                    az = object_azimuth(t, tk["pos"])
                    tol = SCORING_PARAMS["viewpoint_tolerance_deg"]
                    seen_vp = bool(any(abs((az - h + 180) % 360 - 180) <= tol for h in cur["az_prior"].get(g, [])))
                cur["eval"][g] = {"type": ttype, "phase": cur["phase"], "experienced_viewpoint": seen_vp,
                                  "was_unobserved": bool(invisible.get(g, False)),
                                  "slid_before": bool(slid_log.get(g)),
                                  "outcome": {v: scorers[v].outcome(exports[v], gt, g, cur["snap"][v]) for v in variants}}

    finals = {}
    for v, (p, a) in procs.items():
        a.send(("finish", None))
        kind, fin = a.recv()
        if kind != "final":
            raise RuntimeError(fin)
        finals[v] = fin
        p.join(timeout=30)

    names = dict(world.names)
    target_ids = {g for g in names if g >= 10}
    result = {"seed": seed, "cycle_kinds": kinds, "n_ticks": len(ticks), "phase2_start_tick": T2,
              "scene_params": SCENE_PARAMS, "sensor_params": SENSOR_PARAMS, "scoring_params": SCORING_PARAMS,
              "analysis_params": ANALYSIS_PARAMS, "thing_names": names,
              "slid_in_view": {str(g): ts for g, ts in slid_log.items()},
              "displaced_unobserved_phase2": sorted(moved_p2), "variants": {}}
    hist_dir = os.path.join(out_dir, "histories")
    os.makedirs(hist_dir, exist_ok=True)
    for v in variants:
        rep, dump = finals[v]["report"], finals[v]["state_dump"]
        if dump_state:
            os.makedirs(dump_state, exist_ok=True)
            open(os.path.join(dump_state, f"seed{seed}_{v}_state.json"), "w").write(dump)
        leak_terms = [CANARY] + [n for g, n in names.items() if g >= 10]
        audit = {
            "observations_unaltered": rep["observation_hashes"] == sent_hashes,
            "inference_missing_provenance": len(rep["inference_audit"]["missing_provenance"]),
            "locality_violations": rep["locality_violations"],
            "canary_or_names_in_learner_state": [x for x in leak_terms if x in dump],
            "feedback": rep["feedback_audit"],
        }
        trials = [{"cycle": c["cycle"], "phase": e["phase"], "cycle_kind": c["kind"], "thing": g,
                   "name": names.get(g), "type": e["type"], "experienced_viewpoint": e["experienced_viewpoint"],
                   "was_unobserved": e["was_unobserved"], "event_unobserved": c["unobserved_ok"],
                   "slid_before": e["slid_before"], "final": e["outcome"][v]}
                  for c in cycles for g, e in c["eval"].items()]
        analysis = analyse_units(rep, record_gt[v], target_ids, T2, slid_log, moved_p2, seed)
        result["variants"][v] = {
            "mechanisms": rep["mechanisms"], "audit": audit, "trials": trials,
            "discovery": scorers[v].discovery(structure_ids, rep["canonical"]),
            "learner_stats": {k: rep[k] for k in ("decision_counts", "n_units", "n_units_provisional_open",
                                                   "n_records", "touch_log", "cpu_seconds")},
            "units": analysis,
        }
        mapped = {u["unit"] for u in analysis}
        with open(os.path.join(hist_dir, f"seed{seed}_{v}.json"), "w") as f:
            json.dump({"seed": seed, "variant": v, "phase2_start_tick": T2, "thing_names": names,
                       "units": {str(u["unit"]): {"thing": names.get(u["gt"]),
                                                  "history": rep["usefulness_history"].get(u["unit"], [])}
                                 for u in analysis}}, f, default=_json_default)
    result["wall_seconds"] = time.time() - t_start
    path = os.path.join(out_dir, f"seed{seed}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, default=_json_default)
    os.replace(tmp, path)
    return path


def analyse_units(rep, rgt, target_ids, T2, slid_log, moved_p2, seed):
    """Map learner units to physical things (harness-only) and compare learned usefulness with
    oracle discriminability computed on the learner's OWN records."""
    A = ANALYSIS_PARAMS
    recs = rep["records"]
    ex = {e["name"]: registry.build("evidence", e) for e in rep["mechanisms"]["evidence"]}
    lab = {rid: g for rid, (g, pur) in rgt.items() if pur >= A["record_min_purity"]}
    by_unit = {}
    for rid, r in recs.items():
        by_unit.setdefault(r["unit"], []).append(rid)
    rng = np.random.default_rng(50_000 + seed)
    out = []
    for u, rids in by_unit.items():
        gs = [lab.get(r) for r in rids]
        if len(rids) < A["map_min_records"]:
            continue
        vals, cnt = np.unique([g for g in gs if g is not None] or [-99], return_counts=True)
        g = int(vals[np.argmax(cnt)])
        if g not in target_ids or cnt.max() / len(rids) < A["map_min_share"]:
            continue
        qs = [r for r in rids if lab.get(r) == g]
        qs = [qs[i] for i in np.linspace(0, len(qs) - 1, min(len(qs), A["oracle_max_queries"])).astype(int)]
        oracle = {}
        for k in KINDS:
            same, diff = [], []
            for q in qs:
                qa, tq = recs[q]["ev"].get(k), recs[q]["tick"]
                if qa is None:
                    continue
                sp = [r for r, gg in lab.items() if gg == g and abs(recs[r]["tick"] - tq) >= A["oracle_gap_ticks"]
                      and recs[r]["ev"].get(k) is not None]
                dp = [r for r, gg in lab.items() if gg in target_ids and gg != g and recs[r]["ev"].get(k) is not None]
                for pool, acc in ((sp, same), (dp, diff)):
                    if len(pool) > A["oracle_max_pool"]:
                        pool = list(rng.choice(pool, A["oracle_max_pool"], replace=False))
                    acc.extend(ex[k].distance(qa, recs[r]["ev"][k]) for r in pool)
            if same and diff:
                s_, d_ = np.array(same)[:, None], np.array(diff)[None, :]
                oracle[k] = float(((s_ < d_).sum() + 0.5 * (s_ == d_).sum()) / (s_.size * d_.size))
        hist = rep["usefulness_history"].get(u, [])
        at_p2 = {}
        for (tick, k, old, new, nW, nB, cause) in hist:
            if tick < T2:
                at_p2[k] = new
        loc_w_after = {g_: any(h[1] == "location_self_frame" and h[6][0] == "W" and h[0] >= min(ts)
                               for h in hist) for g_, ts in slid_log.items()}
        if g in slid_log and loc_w_after.get(g):
            mclass = "slid"
        elif g not in slid_log and g not in moved_p2:
            mclass = "static"
        else:
            mclass = "other"
        out.append({"unit": u, "gt": g, "n_records": len(rids), "created_tick": rep["unit_created"][u][0],
                    "oracle": oracle, "learned_final": rep["usefulness_final"].get(u, {}),
                    "learned_at_phase2_start": at_p2, "existed_before_phase2": rep["unit_created"][u][0] < T2,
                    "motion_class": mclass})
    return out


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--dump-state", default=None)
    a = ap.parse_args()
    print(run(a.seed, a.out, a.variants.split(","), a.dump_state))
