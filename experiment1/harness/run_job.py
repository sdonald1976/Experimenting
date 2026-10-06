"""Run one (seed, stage) job: render the world, feed every learner variant
through a pipe to its own process, score, audit, save JSON.

Usage: python -m harness.run_job --seed 0 --stage return --out results/dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np

from learner.config import VARIANTS
from learner.process import learner_main

from .render import Renderer
from .scene import CANARY, PALETTE, SCENE_PARAMS, World, arc_pos, build_schedule, object_azimuth
from .scoring import SCORING_PARAMS, VariantScorer

SENSOR_PARAMS = {
    "width": 256, "height": 192, "hfov_deg": 60.0, "baseline_m": 0.12,
    "mount_pitch_deg": -28.0, "cam_height_m": 1.0, "pixel_noise_sigma": 3.0,
    "self_motion_noise_rel": 0.03, "self_motion_noise_abs_m": 0.003, "self_motion_noise_abs_rad": 0.002,
}


def obs_hash(obs):
    h = hashlib.sha256()
    h.update(obs["left"].tobytes())
    h.update(obs["right"].tobytes())
    h.update(repr(sorted(obs["self_motion"].items())).encode())
    return h.hexdigest()


def validate_obs(obs, calib):
    assert set(obs) == {"tick", "left", "right", "self_motion"}, set(obs)
    for k in ("left", "right"):
        a = obs[k]
        assert a.dtype == np.uint8 and a.shape == (calib["height"], calib["width"], 3)
    assert set(obs["self_motion"]) == {"d_forward", "d_right", "d_yaw"}
    assert all(isinstance(v, float) for v in obs["self_motion"].values())


def true_delta(prev, cur):
    """Body-frame motion between two scheduled poses (ground truth, harness only)."""
    psi0 = prev["psi"]
    r = np.array([np.cos(psi0), -np.sin(psi0)])
    f = np.array([np.sin(psi0), np.cos(psi0)])
    dp = cur["pos"] - prev["pos"]
    return float(dp @ f), float(dp @ r), float(cur["psi"] - psi0)


def apply_event(world: World, ev, rng, experienced_az, cycle):
    """Modify the world while unobserved. Returns trial bookkeeping."""
    kind = ev["kind"]
    info = {"kind": kind, "moved": None, "added": None, "removed": None}
    ret_cam = arc_pos(ev["return_alpha"])
    if kind == "move":
        t = world.things[rng.integers(len(world.things))]
        old = t.pos.copy()
        try:
            t.pos = world.free_position(t.footprint_radius(), exclude=t, min_from=old)
        except RuntimeError:  # crowded: move another thing instead (failure path only)
            for t in world.things:
                old = t.pos.copy()
                try:
                    t.pos = world.free_position(t.footprint_radius(), exclude=t, min_from=old)
                    break
                except RuntimeError:
                    continue
            else:
                raise
        az_options = experienced_az.get(t.gt_id) or [0.0]
        az = np.radians(float(rng.choice(az_options)))
        v = ret_cam - t.pos
        t.yaw = float(np.arctan2(v[0], v[1]) - az)
        info["moved"] = t.gt_id
    elif kind in ("novel", "replace"):
        from .scene import _random_size
        pos, avoid = None, None
        if kind == "replace":
            victim = world.things[rng.integers(len(world.things))]
            world.things.remove(victim)
            pos, avoid = victim.pos.copy(), (victim.shape, victim.name.split("_")[-2])
            info["removed"] = victim.gt_id
        for _ in range(200):
            shape = str(rng.choice(["box", "cylinder", "sphere"]))
            col = str(rng.choice(list(PALETTE)))
            if avoid and (shape, col) == avoid:
                continue
            size = _random_size(rng, shape)
            fp = World._fp(shape, size)
            if pos is None:
                try:
                    p = world.free_position(fp)
                except RuntimeError:  # crowded: try another shape/size (failure path only)
                    continue
                break
            ok = all(np.linalg.norm(o.pos - pos) >= o.footprint_radius() + fp + 0.05 for o in world.things)
            if ok:
                p = pos
                break
        else:
            raise RuntimeError("could not place replacement")
        t = world.add_thing(shape, col, size, p, rng.uniform(0, 2 * np.pi), f"NOVEL{cycle}")
        info["added"] = t.gt_id
    return info


def run(seed: int, stage: str, out_dir: str, variants: list[str], dump_state: str | None = None):
    t_start = time.time()
    world = World(seed)
    ticks, kinds = build_schedule(world, stage)
    S = SENSOR_PARAMS
    rend = Renderer(S["width"], S["height"], S["hfov_deg"], S["baseline_m"],
                    np.radians(S["mount_pitch_deg"]), S["cam_height_m"], noise_sigma=S["pixel_noise_sigma"])
    calib = rend.calibration()
    noise_rng = np.random.default_rng(10_000 + seed * 7 + (stage == "moved"))
    event_rng = np.random.default_rng(20_000 + seed * 7 + (stage == "moved"))
    structure_ids = {s.gt_id for s in world.structure}

    ctx = mp.get_context("spawn")
    procs = {}
    for v in variants:
        a, b = ctx.Pipe()
        p = ctx.Process(target=learner_main, args=(b, VARIANTS[v]), daemon=True)
        p.start()
        procs[v] = (p, a)
    described = {}
    for v, (p, a) in procs.items():
        a.send(("calib", calib))
        kind, d = a.recv()
        assert kind == "ready", d
        described[v] = d

    scorers = {v: VariantScorer({t.gt_id for t in world.things}) for v in variants}
    experienced_az: dict[int, list] = {}
    sent_hashes = {}
    cycles = []          # bookkeeping per cycle
    cur = None
    invisible_since = {}  # gt -> bool seen invisible during this cycle
    prev_tick = None
    for i, tk in enumerate(ticks):
        # --- world modification while unobserved (event at first away tick)
        if tk["event"] is not None:
            snap = {v: scorers[v].association_snapshot() for v in variants}
            info = apply_event(world, tk["event"], event_rng, experienced_az, tk["cycle"])
            for sc in scorers.values():
                sc.thing_ids |= {t.gt_id for t in world.things}
            cur = {"cycle": tk["cycle"], "kind": info["kind"], "info": info, "snap": snap,
                   "event_tick": i, "unobserved_ok": None, "eval": {},
                   "az_prior": {g: list(v) for g, v in experienced_az.items()}}
            cycles.append(cur)
        L, R, gt, _depth = rend.render_stereo(world.surfaces(), tk["pos"], tk["psi"], noise_rng)
        present = {t.gt_id for t in world.things}
        if tk["event"] is not None:
            touched = [x for x in (info["moved"], info["added"], info["removed"]) if x is not None]
            cur["unobserved_ok"] = bool(all((gt == g).sum() == 0 for g in touched))
        if tk["phase"] in ("away",):
            for g in present:
                if (gt == g).sum() == 0:
                    invisible_since[g] = True
        # --- experienced viewpoints (harness bookkeeping for Q15)
        for t in world.things:
            if (gt == t.gt_id).sum() >= SCORING_PARAMS["visible_min_px"]:
                experienced_az.setdefault(t.gt_id, []).append(object_azimuth(t, tk["pos"]))
        # --- self-motion estimate (noisy, body frame)
        if prev_tick is None:
            sm = {"d_forward": 0.0, "d_right": 0.0, "d_yaw": 0.0}
        else:
            df, dr, dy = true_delta(prev_tick, tk)
            n = lambda x, a: float(x + noise_rng.normal(0, S["self_motion_noise_rel"] * abs(x) + a))
            sm = {"d_forward": n(df, S["self_motion_noise_abs_m"]), "d_right": n(dr, S["self_motion_noise_abs_m"]),
                  "d_yaw": n(dy, S["self_motion_noise_abs_rad"])}
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
        # --- trial evaluation during the observe window
        if tk["phase"] == "observe" and cur is not None:
            last = i + 1 == len(ticks) or ticks[i + 1]["phase"] != "observe"
            for t in world.things:
                g = t.gt_id
                if (gt == g).sum() < SCORING_PARAMS["visible_min_px"]:
                    continue
                if g == cur["info"]["added"]:
                    ttype = "replace_new" if cur["kind"] == "replace" else "novel"
                elif g == cur["info"]["moved"]:
                    ttype = "moved"
                else:
                    ttype = "unmoved"
                az = object_azimuth(t, tk["pos"])
                seen_vp = None
                if ttype in ("moved", "unmoved"):
                    hist = cur["az_prior"].get(g, [])
                    tol = SCORING_PARAMS["viewpoint_tolerance_deg"]
                    seen_vp = bool(any(abs((az - h + 180) % 360 - 180) <= tol for h in hist))
                e = cur["eval"].setdefault(g, {"type": ttype, "seq": {v: [] for v in variants},
                                               "experienced_viewpoint": seen_vp,
                                               "was_unobserved": bool(invisible_since.get(g, False))})
                for v in variants:
                    o = scorers[v].outcome(exports[v], gt, g, cur["snap"][v])
                    e["seq"][v].append(o)
            if last:
                invisible_since = {}

    finals = {}
    for v, (p, a) in procs.items():
        a.send(("finish", None))
        kind, fin = a.recv()
        if kind != "final":
            raise RuntimeError(fin)
        finals[v] = fin
        p.join(timeout=30)

    names = dict(world.names)
    result = {"seed": seed, "stage": stage, "cycle_kinds": kinds, "n_ticks": len(ticks),
              "scene_params": SCENE_PARAMS, "sensor_params": SENSOR_PARAMS, "scoring_params": SCORING_PARAMS,
              "thing_names": names, "wall_seconds": time.time() - t_start, "variants": {}}
    for v in variants:
        rep = finals[v]["report"]
        dump = finals[v]["state_dump"]
        if dump_state:
            os.makedirs(dump_state, exist_ok=True)
            with open(os.path.join(dump_state, f"{stage}_seed{seed}_{v}_state.json"), "w") as f:
                f.write(dump)
        leak_terms = [CANARY] + [n for g, n in names.items() if g >= 10]
        audit = {
            "observations_unaltered": rep["observation_hashes"] == sent_hashes,
            "inference_missing_provenance": len(rep["inference_audit"]["missing_provenance"]),
            "locality_violations": rep["locality_violations"],
            "canary_or_names_in_learner_state": [x for x in leak_terms if x in dump],
        }
        trials = []
        for c in cycles:
            for g, e in c["eval"].items():
                trials.append({"cycle": c["cycle"], "cycle_kind": c["kind"], "thing": g, "name": names.get(g),
                               "type": e["type"], "experienced_viewpoint": e["experienced_viewpoint"],
                               "was_unobserved": e["was_unobserved"], "event_unobserved": c["unobserved_ok"],
                               "seq": e["seq"][v]})
        result["variants"][v] = {
            "mechanisms": rep["mechanisms"], "audit": audit, "trials": trials,
            "discovery": scorers[v].discovery(structure_ids, rep["canonical"]),
            "learner_stats": {k: rep[k] for k in ("decision_counts", "n_units", "n_units_provisional_open",
                                                   "n_records", "touch_log", "cpu_seconds")},
            "inference_counts": rep["inference_audit"]["by_kind"],
        }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stage}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", choices=["return", "moved"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(v for v in VARIANTS if not v.startswith("main_threshold")))
    ap.add_argument("--dump-state", default=None, help="debug: write learner belief state JSON here")
    a = ap.parse_args()
    print(run(a.seed, a.stage, a.out, a.variants.split(","), a.dump_state))
