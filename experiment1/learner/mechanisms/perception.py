"""Perception mechanisms. ALL EXPERIMENTAL (hand-designed, no pretrained parts)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..registry import Mechanism, register


# --------------------------------------------------------------- self-motion
@register("self_motion")
class DeadReckoning(Mechanism):
    NAME = "dead_reckoning"
    VERSION = "1"
    DEFAULTS = {}
    ASSUMPTIONS = [
        "Integrates the noisy body-frame self-motion estimates exactly as given; "
        "no visual correction, so the estimated pose drifts.",
        "The learner's world frame is its own starting pose (it never learns true world coordinates).",
    ]

    def __init__(self, **p):
        super().__init__(**p)
        self.x = self.z = self.psi = 0.0

    def step(self, sm: dict):
        r = (np.cos(self.psi), -np.sin(self.psi))
        f = (np.sin(self.psi), np.cos(self.psi))
        self.x += sm["d_right"] * r[0] + sm["d_forward"] * f[0]
        self.z += sm["d_right"] * r[1] + sm["d_forward"] * f[1]
        self.psi += sm["d_yaw"]
        return {"x": self.x, "z": self.z, "psi": self.psi}


# --------------------------------------------------------------- stereo
@register("stereo")
class SADBlockMatching(Mechanism):
    NAME = "sad_block_matching"
    VERSION = "1"
    DEFAULTS = {"window": 7, "max_disp": 48, "min_disp": 1.0, "uniqueness": 0.92, "lr_tolerance": 1.0}
    ASSUMPTIONS = [
        "Images are rectified (uses the allowed sensor calibration, Q1).",
        "Sum-of-absolute-differences on grey levels over a square window; winner-takes-all.",
        "A disparity is kept only if it is unique (best/second-best cost ratio) and "
        "left-right consistent; everything else is 'no depth' (not guessed).",
    ]

    def compute(self, left, right):
        p = self.p
        L = left.astype(np.float32).mean(axis=2)
        R = right.astype(np.float32).mean(axis=2)
        H, W = L.shape
        D = p["max_disp"]
        big = 1e9
        cost = np.full((D, H, W), big, np.float32)
        for d in range(D):
            diff = np.abs(L[:, d:] - R[:, : W - d])
            cost[d, :, d:] = ndimage.uniform_filter(diff, size=p["window"], mode="nearest")
        dl = np.argmin(cost, axis=0)
        best = np.take_along_axis(cost, dl[None], 0)[0]
        # second best outside +-1 of the winner
        masked = cost.copy()
        for o in (-1, 0, 1):
            idx = np.clip(dl + o, 0, D - 1)
            np.put_along_axis(masked, idx[None], big, 0)
        second = masked.min(axis=0)
        unique = best < p["uniqueness"] * second
        # right-view disparities from the same cost volume
        cost_r = np.full_like(cost, big)
        for d in range(D):
            cost_r[d, :, : W - d] = cost[d, :, d:]
        dr = np.argmin(cost_r, axis=0)
        xs = np.arange(W)[None, :] - dl
        xs_c = np.clip(xs, 0, W - 1)
        lr_ok = (xs >= 0) & (np.abs(dl - np.take_along_axis(dr, xs_c, 1)) <= p["lr_tolerance"])
        # sub-pixel parabola
        dm = np.clip(dl - 1, 0, D - 1)
        dp = np.clip(dl + 1, 0, D - 1)
        c0 = np.take_along_axis(cost, dm[None], 0)[0]
        c2 = np.take_along_axis(cost, dp[None], 0)[0]
        denom = c0 - 2 * best + c2
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where((denom > 1e-6) & (dl > 0) & (dl < D - 1), 0.5 * (c0 - c2) / denom, 0.0)
        disp = dl + np.clip(off, -0.5, 0.5)
        valid = unique & lr_ok & (disp >= p["min_disp"]) & (best < big / 2)
        return disp.astype(np.float32), valid


def points_from_disparity(disp, calib):
    H, W = disp.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    with np.errstate(divide="ignore"):
        Z = calib["fx"] * calib["baseline"] / disp
    X = (u - calib["cx"]) * Z / calib["fx"]
    Y = (v - calib["cy"]) * Z / calib["fy"]
    return np.stack([X, Y, Z], axis=-1)


def cam_to_world(pts, calib, pose):
    """Camera-frame points -> the learner's own self-motion frame (y up)."""
    ph = calib["mount_pitch"]
    X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
    bx = X
    by = -np.cos(ph) * Y + np.sin(ph) * Z
    bz = np.sin(ph) * Y + np.cos(ph) * Z
    c, s = np.cos(pose["psi"]), np.sin(pose["psi"])
    wx = pose["x"] + bx * c + bz * s
    wz = pose["z"] - bx * s + bz * c
    return np.stack([wx, by, wz], axis=-1)


# --------------------------------------------------------------- candidates
@register("candidates")
class DepthChromaBoundaries(Mechanism):
    NAME = "depth_chroma_boundaries"
    VERSION = "1"
    DEFAULTS = {"disp_jump_px": 1.0, "chroma_jump": 0.05, "chroma_min_brightness": 30.0,
                "blur": 3, "min_pixels": 150}
    ASSUMPTIONS = [
        "A 'distinguishable candidate' (Q4) = a 4-connected set of pixels with valid depth, "
        "not separated by a disparity jump or a chromaticity jump, with at least min_pixels.",
        "Chromaticity (r/(r+g+b), g/(r+g+b)) is used instead of brightness so that shading "
        "and intensity patterns do not split a surface. This is a hand-designed prior.",
        "Pixels with no valid depth never belong to a candidate.",
        "No notion of object, background or ground: floor and wall pieces are candidates too (Q5).",
    ]

    def extract(self, img, disp, valid):
        p = self.p
        im = ndimage.uniform_filter(img.astype(np.float32), size=(p["blur"], p["blur"], 1))
        s = im.sum(axis=2) + 1e-6
        chroma = im[..., :2] / s[..., None]
        bright = s / 3

        def cut(a, b, sl_a, sl_b):
            va, vb = valid[sl_a], valid[sl_b]
            dj = np.abs(disp[sl_a] - disp[sl_b]) > p["disp_jump_px"]
            cj = np.linalg.norm(chroma[sl_a] - chroma[sl_b], axis=-1) > p["chroma_jump"]
            cj &= (bright[sl_a] > p["chroma_min_brightness"]) & (bright[sl_b] > p["chroma_min_brightness"])
            return ~va | ~vb | dj | cj

        H, W = disp.shape
        boundary = ~valid.copy()
        cr = cut(None, None, (slice(None), slice(0, W - 1)), (slice(None), slice(1, W)))
        cd = cut(None, None, (slice(0, H - 1), slice(None)), (slice(1, H), slice(None)))
        boundary[:, :-1] |= cr
        boundary[:, 1:] |= cr
        boundary[:-1, :] |= cd
        boundary[1:, :] |= cd
        lab, n = ndimage.label(valid & ~boundary)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        keep = np.nonzero(sizes >= p["min_pixels"])[0]
        keep = keep[keep > 0]
        remap = np.zeros(n + 1, np.int32)
        remap[keep] = np.arange(1, len(keep) + 1)
        return remap[lab], chroma, bright


# --------------------------------------------------------------- continuity
@register("continuity")
class OverlapContinuity(Mechanism):
    NAME = "overlap_continuity"
    VERSION = "1"
    DEFAULTS = {"min_iou": 0.3, "compensate_rotation": True}
    ASSUMPTIONS = [
        "Uninterrupted continuity = the candidate overlaps (IoU >= min_iou) the region the unit "
        "occupied on the immediately preceding tick. ANY gap of one tick breaks continuity (D3).",
        "Rotation compensation: the previous region is shifted horizontally by -fx * d_yaw "
        "(small-angle, pitch ignored) using the noisy self-motion estimate.",
        "One-to-one greedy assignment by IoU.",
    ]

    def link(self, prev: list, cur_labels, n_cur, d_yaw, fx):
        """prev: list of (unit_id, flat pixel indices). Returns {cur_label: (unit_id, iou)}."""
        H, W = cur_labels.shape
        cur_sizes = np.bincount(cur_labels.ravel(), minlength=n_cur + 1)
        shift = int(round(-fx * d_yaw)) if self.p["compensate_rotation"] else 0
        pairs = []
        for uid, pix in prev:
            ys, xs = np.divmod(pix, W)
            xs = xs + shift
            ok = (xs >= 0) & (xs < W)
            ys, xs = ys[ok], xs[ok]
            if len(xs) == 0:
                continue
            labs = cur_labels[ys, xs]
            inter = np.bincount(labs, minlength=n_cur + 1)
            for c in np.nonzero(inter[1:])[0] + 1:
                iou = inter[c] / (len(pix) + cur_sizes[c] - inter[c])
                if iou >= self.p["min_iou"]:
                    pairs.append((iou, uid, c))
        pairs.sort(reverse=True)
        used_u, out = set(), {}
        for iou, uid, c in pairs:
            if uid in used_u or c in out:
                continue
            used_u.add(uid)
            out[c] = (uid, float(iou))
        return out


# --------------------------------------------------------------- sampling
@register("sampling")
class FixedIntervalSampling(Mechanism):
    NAME = "fixed_interval"
    VERSION = "1"
    DEFAULTS = {"interval": 3}
    ASSUMPTIONS = [
        "An experience record is made for every active unit on ticks where tick % interval == 0, "
        "and immediately whenever a unit is created or re-linked by recognition (Q7).",
        "Records are never merged or averaged.",
    ]

    def due(self, tick):
        return tick % self.p["interval"] == 0
