"""Experiment 2 world + scripted camera route + phase schedule.

HARNESS-SIDE ONLY. Contains ground truth and human-readable names.
"""
from __future__ import annotations

import numpy as np

from .render import Plane, Thing, make_texture

SCENE_PARAMS = {
    "room_half_size_m": 4.0,
    "placement_radius_m": 1.15,
    "min_gap_m": 0.2,
    "arc_radius_m": 2.3,
    "explore_arc_deg": 40.0,
    "explore_step_deg": 1.0,
    "turn_step_deg": 12.0,
    "away_move_ticks": 8,
    "observe_ticks_phase1": 10,
    "observe_ticks_phase2": 40,
    "return_arc_deg": 35.0,
    "min_move_distance_m": 0.45,
    "phase1_cycles": 4,
    "phase2_cycles": {"swap": 2, "move": 1, "novel": 1, "replace": 1, "plain": 1},
    "slide_speed_m_per_tick": 0.03,
    "slide_start_observe_tick": 10,   # continuous change starts after the 10-tick evaluation look
    "slide_things_per_window": [1, 2],
    "slide_distances_m": [0.9, 0.6],  # try the long slide first, then a shorter one if the floor is crowded
    "rotate_things_per_window": [1, 2],
    "rotate_deg_per_tick": 4.0,
    "camera_arc_deg_per_tick": 0.5,   # camera keeps moving along its arc during the change ...
    "camera_approach_m": 0.5,         # ... and approaches the scene (apparent size changes)
}

PALETTE = {
    "red": (0.80, 0.22, 0.17), "green": (0.22, 0.68, 0.27), "blue": (0.20, 0.32, 0.85),
    "yellow": (0.85, 0.74, 0.16), "purple": (0.58, 0.26, 0.76), "orange": (0.90, 0.50, 0.12),
    "cyan": (0.15, 0.68, 0.74), "pink": (0.90, 0.42, 0.62),
}
FLOOR = (0.50, 0.50, 0.50)
WALLS = [(0.78, 0.66, 0.50), (0.52, 0.62, 0.78), (0.62, 0.76, 0.56), (0.80, 0.58, 0.66)]

CANARY = "CANARY-GT-LEAK-e2-51b7d0aa"  # harness-only string; must never appear in learner state


def _random_size(rng, shape):
    if shape == "box":
        return rng.uniform(0.10, 0.20, size=3)
    if shape == "cylinder":
        return np.array([rng.uniform(0.09, 0.17), rng.uniform(0.20, 0.45), 0.0])
    return np.array([rng.uniform(0.10, 0.18), 0.0, 0.0])


def _surface_point(rng, shape, size):
    if shape == "sphere":
        u = rng.normal(size=3)
        u[1] = abs(u[1]) * 0.6 + 0.1
        return size[0] * u / np.linalg.norm(u)
    if shape == "box":
        u = rng.normal(size=3)
        u[1] = abs(u[1])
        return u / np.max(np.abs(u) / size)
    r, h = size[0], size[1]
    if rng.random() < 0.25:
        a, rr = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.random())
        return np.array([rr * np.cos(a), h / 2, rr * np.sin(a)])
    a = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(a), rng.uniform(-h / 2, h / 2), r * np.sin(a)])


def make_thing(rng, gt_id, shape, colour, size, pos, yaw, label):
    spots = [(_surface_point(rng, shape, size), rng.uniform(0.022, 0.045))
             for _ in range(rng.integers(3, 7))]
    return Thing(gt_id, shape, size, pos, yaw, PALETTE[colour], make_texture(rng), spots,
                 f"{label}_{colour}_{shape}")


def footprint(shape, size):
    return float(np.hypot(size[0], size[2])) if shape == "box" else float(size[0])


class World:
    """Six targets with deliberately unequal evidence reliability:
    - things 0,1,2 share colour c1; things 3,4 share colour c2; thing 5 has colour c3
    - things 0 and 4 are spheres of identical radius (shared size, viewpoint-stable extent)
    - things 1 and 5 are identical ELONGATED boxes (shared size, viewpoint-dependent extent)
    - thing 2 is a near-cube, thing 3 a tall thin cylinder (viewpoint-dependent extent)
    - every thing has its own random marks and fine texture."""

    def __init__(self, seed: int):
        self.rng = rng = np.random.default_rng(seed)
        self.seed = seed
        W = SCENE_PARAMS["room_half_size_m"]
        self.structure = [
            Plane(1, (0, 0, 0), (0, 1, 0), FLOOR, make_texture(rng)),
            Plane(2, (W, 0, 0), (-1, 0, 0), WALLS[0], make_texture(rng)),
            Plane(3, (-W, 0, 0), (1, 0, 0), WALLS[1], make_texture(rng)),
            Plane(4, (0, 0, W), (0, 0, -1), WALLS[2], make_texture(rng)),
            Plane(5, (0, 0, -W), (0, 0, 1), WALLS[3], make_texture(rng)),
        ]
        self.names = {1: "FLOOR", 2: "WALL_E", 3: "WALL_W", 4: "WALL_N", 5: "WALL_S"}
        self.things: list[Thing] = []
        self.next_id = 10
        c1, c2, c3 = (str(c) for c in rng.choice(list(PALETTE), size=3, replace=False))
        r_sph = rng.uniform(0.11, 0.15)
        elong = np.array([rng.uniform(0.20, 0.24), rng.uniform(0.07, 0.10), rng.uniform(0.06, 0.08)])
        cube = np.full(3, rng.uniform(0.10, 0.13))
        tall = np.array([rng.uniform(0.06, 0.08), rng.uniform(0.38, 0.46), 0.0])
        specs = [("sphere", c1, np.array([r_sph, 0, 0])), ("box", c1, elong), ("box", c1, cube),
                 ("cylinder", c2, tall), ("sphere", c2, np.array([r_sph, 0, 0])), ("box", c3, elong)]
        for i, (shape, col, size) in enumerate(specs):
            pos = self.free_position(footprint(shape, size))
            self.add_thing(shape, col, size.copy(), pos, rng.uniform(0, 2 * np.pi), f"THING_{'ABCDEF'[i]}")

    def add_thing(self, shape, col, size, pos, yaw, label):
        t = make_thing(self.rng, self.next_id, shape, col, size, pos, yaw, label)
        self.next_id += 1
        self.things.append(t)
        self.names[t.gt_id] = t.name
        return t

    def free_position(self, size_r, exclude=(), min_from=None, tries=800):
        P = SCENE_PARAMS
        for _ in range(tries):
            r = P["placement_radius_m"] * np.sqrt(self.rng.random())
            a = self.rng.uniform(0, 2 * np.pi)
            p = np.array([r * np.sin(a), r * np.cos(a)])
            if self.is_free(p, size_r, exclude) and (min_from is None or
                                                    np.linalg.norm(p - min_from) >= P["min_move_distance_m"]):
                return p
        raise RuntimeError("no free position")

    def is_free(self, p, size_r, exclude=()):
        return all(np.linalg.norm(t.pos - p) >= t.footprint_radius() + size_r + SCENE_PARAMS["min_gap_m"]
                   for t in self.things if t not in exclude)

    def surfaces(self):
        return self.structure + self.things


def arc_pos(alpha, R=None):
    R = SCENE_PARAMS["arc_radius_m"] if R is None else R
    return np.array([R * np.sin(alpha), R * np.cos(alpha)])


def object_azimuth(thing, cam_xz):
    v = cam_xz - thing.pos
    world_bearing = np.arctan2(v[0], v[1])
    return float(np.degrees((world_bearing - thing.yaw + np.pi) % (2 * np.pi) - np.pi))


def build_schedule(world: World):
    """Tick list: {pos, psi, alpha, R, phase, part, cycle, event, change_start}. Harness-only.
    Phase-2 observation windows: 10-tick evaluation look, then 30 ticks of continuous change in view
    (things slide / rotate, camera moves along the arc and approaches)."""
    P = SCENE_PARAMS
    rng = world.rng
    deg = np.radians
    R0 = P["arc_radius_m"]
    ticks = []

    def add(alpha, psi, phase, part, R=R0, cycle=None, event=None, change_start=False):
        ticks.append({"pos": arc_pos(alpha, R), "psi": psi, "alpha": alpha, "R": R, "phase": phase, "part": part,
                      "cycle": cycle, "event": event, "change_start": change_start})

    a = 0.0
    E, step = deg(P["explore_arc_deg"]), deg(P["explore_step_deg"])
    add(a, a + np.pi, 1, "explore")
    while a > -E + 1e-9:
        a -= step
        add(a, a + np.pi, 1, "explore")
    while a < E - 1e-9:
        a += step
        add(a, a + np.pi, 1, "explore")

    kinds2 = [k for k, n in P["phase2_cycles"].items() for _ in range(n)]
    rng.shuffle(kinds2)
    plan = [(1, "plain")] * P["phase1_cycles"] + [(2, k) for k in kinds2]
    psi = a + np.pi
    R = R0
    turn = deg(P["turn_step_deg"])
    n_turn = int(round(np.pi / turn))
    for ci, (phase, kind) in enumerate(plan):
        for _ in range(n_turn):
            psi += turn
            add(a, psi, phase, "turn_away", R, ci)
        target_a = a
        while abs(target_a - a) < deg(10):
            target_a = rng.uniform(-deg(P["return_arc_deg"]), deg(P["return_arc_deg"]))
        a0, R_from = a, R
        for k in range(1, P["away_move_ticks"] + 1):
            a = a0 + (target_a - a0) * k / P["away_move_ticks"]
            R = R_from + (R0 - R_from) * k / P["away_move_ticks"]          # back to the standard radius while away
            ev = {"kind": kind, "return_alpha": target_a} if k == 1 else None
            add(a, a + (psi - a0), phase, "away", R, ci, ev)
        psi = a + (psi - a0)
        for _ in range(n_turn):
            psi += turn
            add(a, psi, phase, "turn_back", R, ci)
        n_obs = P["observe_ticks_phase1"] if phase == 1 else P["observe_ticks_phase2"]
        n_change = n_obs - P["slide_start_observe_tick"]
        for k in range(n_obs):
            if phase == 2 and k >= P["slide_start_observe_tick"]:
                da = deg(P["camera_arc_deg_per_tick"]) * (1 if a < 0 else -1)   # move toward the arc centre
                a += da
                psi += da
                R -= P["camera_approach_m"] / n_change
            add(a, psi, phase, "observe", R, ci, change_start=(phase == 2 and k == P["slide_start_observe_tick"]))
    return ticks, [k for _, k in plan]
