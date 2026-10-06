"""Synthetic world + scripted camera route + trial schedule.

HARNESS-SIDE ONLY. Contains ground truth and human-readable names.
"""
from __future__ import annotations

import numpy as np

from .render import Plane, Thing, make_texture

# Harness parameters (scene design). Logged with every run.
SCENE_PARAMS = {
    "room_half_size_m": 4.0,
    "placement_radius_m": 0.95,
    "min_gap_m": 0.25,
    "arc_radius_m": 1.9,
    "explore_arc_deg": 40.0,
    "explore_step_deg": 1.0,
    "turn_step_deg": 12.0,
    "away_move_ticks": 8,
    "observe_ticks": 10,
    "return_arc_deg": 35.0,
    "min_move_distance_m": 0.45,
    "n_targets": 4,
    "cycles": {"plain": 4, "novel": 2, "replace": 1},
}

PALETTE = {
    "red": (0.80, 0.22, 0.17), "green": (0.22, 0.68, 0.27), "blue": (0.20, 0.32, 0.85),
    "yellow": (0.85, 0.74, 0.16), "purple": (0.58, 0.26, 0.76), "orange": (0.90, 0.50, 0.12),
    "cyan": (0.15, 0.68, 0.74), "pink": (0.90, 0.42, 0.62),
}
FLOOR = (0.50, 0.50, 0.50)
WALLS = [(0.78, 0.66, 0.50), (0.52, 0.62, 0.78), (0.62, 0.76, 0.56), (0.80, 0.58, 0.66)]

CANARY = "CANARY-GT-LEAK-7f3a91c2"  # harness-only string; must never appear in learner state


def _random_size(rng, shape):
    if shape == "box":
        return rng.uniform(0.10, 0.20, size=3)
    if shape == "cylinder":
        return np.array([rng.uniform(0.09, 0.17), rng.uniform(0.20, 0.45), 0.0])
    return np.array([rng.uniform(0.10, 0.18), 0.0, 0.0])


def _surface_point(rng, shape, size):
    """A random point on the thing's surface in local coordinates."""
    if shape == "sphere":
        u = rng.normal(size=3)
        u[1] = abs(u[1]) * 0.6 + 0.1  # keep spots on the visible upper part
        return size[0] * u / np.linalg.norm(u)
    if shape == "box":
        u = rng.normal(size=3)
        u[1] = abs(u[1])
        u = u / np.max(np.abs(u) / size)
        return u
    r, h = size[0], size[1]
    if rng.random() < 0.25:
        a, rr = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.random())
        return np.array([rr * np.cos(a), h / 2, rr * np.sin(a)])
    a = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(a), rng.uniform(-h / 2, h / 2), r * np.sin(a)])


def make_thing(rng, gt_id, shape, colour, size, pos, yaw, label):
    spots = [(_surface_point(rng, shape, size), rng.uniform(0.022, 0.045))
             for _ in range(rng.integers(3, 7))]
    name = f"{label}_{colour}_{shape}"
    return Thing(gt_id, shape, size, pos, yaw, PALETTE[colour], make_texture(rng), spots, name)


class World:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        P = SCENE_PARAMS
        W = P["room_half_size_m"]
        rng = self.rng
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
        # Deliberate evidence overlap: A,B share colour; B,C share shape+size.
        cols = list(rng.choice(list(PALETTE), size=3, replace=False))
        shared_shape = str(rng.choice(["box", "cylinder", "sphere"]))
        shared_size = _random_size(rng, shared_shape)
        specs = [
            (str(rng.choice(["box", "cylinder", "sphere"])), cols[0], None),
            (shared_shape, cols[0], shared_size),
            (shared_shape, cols[1], shared_size),
            (str(rng.choice(["box", "cylinder", "sphere"])), cols[2], None),
        ]
        for i, (shape, col, size) in enumerate(specs[: P["n_targets"]]):
            size = _random_size(rng, shape) if size is None else size.copy()
            pos = self.free_position(size_r=self._fp(shape, size))
            self.add_thing(shape, col, size, pos, rng.uniform(0, 2 * np.pi), "THING_" + "ABCD"[i])

    @staticmethod
    def _fp(shape, size):
        return float(np.hypot(size[0], size[2])) if shape == "box" else float(size[0])

    def add_thing(self, shape, col, size, pos, yaw, label):
        t = make_thing(self.rng, self.next_id, shape, col, size, pos, yaw, label)
        self.next_id += 1
        self.things.append(t)
        self.names[t.gt_id] = t.name
        return t

    def free_position(self, size_r, avoid=None, exclude=None, min_from=None, tries=500):
        P = SCENE_PARAMS
        for _ in range(tries):
            r = P["placement_radius_m"] * np.sqrt(self.rng.random())
            a = self.rng.uniform(0, 2 * np.pi)
            p = np.array([r * np.sin(a), r * np.cos(a)])
            ok = True
            for t in self.things:
                if exclude is not None and t is exclude:
                    continue
                if np.linalg.norm(t.pos - p) < t.footprint_radius() + size_r + P["min_gap_m"]:
                    ok = False
                    break
            if ok and min_from is not None and np.linalg.norm(p - min_from) < P["min_move_distance_m"]:
                ok = False
            if ok:
                return p
        raise RuntimeError("no free position")

    def surfaces(self):
        return self.structure + self.things


def arc_pos(alpha):
    R = SCENE_PARAMS["arc_radius_m"]
    return np.array([R * np.sin(alpha), R * np.cos(alpha)])


def object_azimuth(thing, cam_xz):
    """Viewing direction (deg) of the camera in the thing's own frame. Harness-only."""
    v = cam_xz - thing.pos
    world_bearing = np.arctan2(v[0], v[1])
    return float(np.degrees((world_bearing - thing.yaw + np.pi) % (2 * np.pi) - np.pi))


def build_schedule(world: World, stage: str):
    """Returns list of tick dicts: {pos, psi, phase, cycle, event(optional)}.
    Events are applied by the run loop at that tick (world modification while unobserved)."""
    P = SCENE_PARAMS
    rng = world.rng
    deg = np.radians
    ticks = []

    def add(alpha, psi, phase, cycle=None, event=None):
        ticks.append({"pos": arc_pos(alpha), "psi": psi, "alpha": alpha,
                      "phase": phase, "cycle": cycle, "event": event})

    a = 0.0
    E = deg(P["explore_arc_deg"])
    step = deg(P["explore_step_deg"])
    add(a, a + np.pi, "explore")
    while a > -E + 1e-9:
        a -= step
        add(a, a + np.pi, "explore")
    while a < E - 1e-9:
        a += step
        add(a, a + np.pi, "explore")

    kinds = (["plain"] * P["cycles"]["plain"] + ["novel"] * P["cycles"]["novel"]
             + ["replace"] * P["cycles"]["replace"])
    rng.shuffle(kinds)
    psi = a + np.pi
    turn = deg(P["turn_step_deg"])
    for ci, kind in enumerate(kinds):
        n_turn = int(round(np.pi / turn))
        for k in range(n_turn):
            psi += turn
            add(a, psi, "turn_away", ci)
        target_a = a
        while abs(target_a - a) < deg(10):
            target_a = rng.uniform(-deg(P["return_arc_deg"]), deg(P["return_arc_deg"]))
        n_mv = P["away_move_ticks"]
        a0 = a
        for k in range(1, n_mv + 1):
            a = a0 + (target_a - a0) * k / n_mv
            psi_k = a + (psi - a0)  # keep facing outward while moving
            ev = None
            if k == 1:
                ev = {"kind": "move" if (kind == "plain" and stage == "moved") else kind,
                      "return_alpha": target_a}
            add(a, psi_k, "away", ci, ev)
        psi = a + (psi - a0)
        for k in range(n_turn):
            psi += turn
            add(a, psi, "turn_back", ci)
        for k in range(P["observe_ticks"]):
            add(a, psi, "observe", ci)
    return ticks, kinds
