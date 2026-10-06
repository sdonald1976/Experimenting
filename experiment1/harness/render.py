"""Ray-cast stereo renderer for the synthetic world.

HARNESS-SIDE ONLY. This module knows ground truth (geometry, identities,
depth). Nothing from here except rendered pixel arrays may reach the learner.

Conventions
-----------
World: y is up, floor is y = 0.
Body pose: position (x, z) on the floor plane, yaw psi.
    forward(psi) = (sin psi, 0, cos psi), right(psi) = (cos psi, 0, -sin psi)
Camera frame: x right, y down, z forward, pitched by `pitch` (negative = down)
relative to the body. Two rectified pinhole cameras; the right camera sits
`baseline` metres along the body's right vector.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- geometry


def body_axes(psi: float):
    fwd = np.array([np.sin(psi), 0.0, np.cos(psi)])
    right = np.array([np.cos(psi), 0.0, -np.sin(psi)])
    up = np.array([0.0, 1.0, 0.0])
    return right, up, fwd


def camera_axes(psi: float, pitch: float):
    """World-space (right, down, forward) unit vectors of a camera."""
    right, up, fwd = body_axes(psi)
    cam_fwd = np.sin(pitch) * up + np.cos(pitch) * fwd
    cam_down = -np.cos(pitch) * up + np.sin(pitch) * fwd
    return right, cam_down, cam_fwd


def yaw_rotate(v: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate world vectors (..., 3) about +y by `yaw` (same sense as body yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)


# ---------------------------------------------------------------- surfaces


class Surface:
    """A rendered primitive belonging to one physical thing (gt_id)."""

    gt_id: int
    albedo: np.ndarray  # rgb in [0,1]

    def intersect(self, o, d):  # -> t (N,), inf where no hit
        raise NotImplementedError

    def normal_and_local(self, p):  # -> normals (N,3), local coords (N,3)
        raise NotImplementedError


class Plane(Surface):
    def __init__(self, gt_id, point, normal, albedo, texture):
        self.gt_id = gt_id
        self.point = np.asarray(point, float)
        self.normal = np.asarray(normal, float)
        self.albedo = np.asarray(albedo, float)
        self.texture = texture
        self.spots = []

    def intersect(self, o, d):
        denom = d @ self.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.point - o) @ self.normal) / denom
        t = np.where((denom < -1e-9) & (t > 1e-6), t, np.inf)
        return t

    def normal_and_local(self, p):
        return np.broadcast_to(self.normal, p.shape), p


class Thing(Surface):
    """Rigid thing resting on the floor: box, cylinder or sphere."""

    def __init__(self, gt_id, shape, size, pos, yaw, albedo, texture, spots, name):
        self.gt_id = gt_id
        self.shape = shape          # "box" | "cylinder" | "sphere"
        self.size = np.asarray(size, float)  # box: (hx,hy,hz); cyl: (r,h,_); sphere: (r,_,_)
        self.pos = np.asarray(pos, float)    # (x, z) on floor
        self.yaw = float(yaw)
        self.albedo = np.asarray(albedo, float)
        self.texture = texture
        self.spots = spots          # list of (local_center(3,), radius)
        self.name = name            # human-readable, harness/log only

    @property
    def center(self):
        if self.shape == "box":
            cy = self.size[1]
        elif self.shape == "cylinder":
            cy = self.size[1] / 2
        else:
            cy = self.size[0]
        return np.array([self.pos[0], cy, self.pos[1]])

    def footprint_radius(self):
        if self.shape == "box":
            return float(np.hypot(self.size[0], self.size[2]))
        return float(self.size[0])

    def to_local(self, p):
        return yaw_rotate(p - self.center, -self.yaw)

    def intersect(self, o, d):
        c = self.center
        if self.shape == "sphere":
            r = self.size[0]
            oc = o - c
            b = d @ oc if oc.ndim == 1 else np.einsum("ij,ij->i", d, oc)
            cc = oc @ oc - r * r
            disc = b * b - cc
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(disc)
            return np.where((disc > 0) & (t > 1e-6), t, np.inf)
        # transform ray into local frame (yaw only)
        ol = yaw_rotate((o - c)[None, :], -self.yaw)[0]
        dl = yaw_rotate(d, -self.yaw)
        if self.shape == "box":
            h = self.size
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / dl
                t1 = (-h - ol) * inv
                t2 = (h - ol) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmin > 1e-6)
            return np.where(hit, tmin, np.inf)
        # cylinder, axis along local y, y in [-h/2, h/2]
        r, hgt = self.size[0], self.size[1]
        a = dl[:, 0] ** 2 + dl[:, 2] ** 2
        b = 2 * (ol[0] * dl[:, 0] + ol[2] * dl[:, 2])
        cc = ol[0] ** 2 + ol[2] ** 2 - r * r
        disc = b * b - 4 * a * cc
        with np.errstate(invalid="ignore", divide="ignore"):
            ts = (-b - np.sqrt(disc)) / (2 * a)
        ys = ol[1] + ts * dl[:, 1]
        side = np.where((disc > 0) & (ts > 1e-6) & (np.abs(ys) <= hgt / 2), ts, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            tc = (hgt / 2 - ol[1]) / dl[:, 1]
        px = ol[0] + tc * dl[:, 0]
        pz = ol[2] + tc * dl[:, 2]
        cap = np.where((tc > 1e-6) & (px * px + pz * pz <= r * r), tc, np.inf)
        return np.minimum(side, cap)

    def normal_and_local(self, p):
        loc = self.to_local(p)
        if self.shape == "sphere":
            nl = loc / np.linalg.norm(loc, axis=1, keepdims=True)
        elif self.shape == "box":
            q = np.abs(loc) / self.size
            k = np.argmax(q, axis=1)
            nl = np.zeros_like(loc)
            nl[np.arange(len(loc)), k] = np.sign(loc[np.arange(len(loc)), k])
        else:
            r, hgt = self.size[0], self.size[1]
            on_cap = loc[:, 1] > hgt / 2 - 1e-4
            nl = np.stack([loc[:, 0] / r, np.zeros(len(loc)), loc[:, 2] / r], axis=1)
            nl[on_cap] = [0, 1, 0]
        return yaw_rotate(nl, self.yaw), loc


# ---------------------------------------------------------------- texture


def make_texture(rng, amp=0.22, n=5, kmin=50.0, kmax=170.0):
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ks = rng.uniform(kmin, kmax, size=n)
    phases = rng.uniform(0, 2 * np.pi, size=n)
    return (dirs * ks[:, None], phases, amp)


def eval_texture(tex, local):
    kv, ph, amp = tex
    return 1.0 + amp * np.mean(np.sin(local @ kv.T + ph), axis=1)


# ---------------------------------------------------------------- renderer


class Renderer:
    def __init__(self, width, height, hfov_deg, baseline, pitch, cam_height,
                 light_dir=(0.35, 1.0, 0.25), ambient=0.42, noise_sigma=3.0):
        self.w, self.h = width, height
        self.f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2
        self.baseline = baseline
        self.pitch = pitch
        self.cam_height = cam_height
        L = np.asarray(light_dir, float)
        self.light = L / np.linalg.norm(L)
        self.ambient = ambient
        self.noise_sigma = noise_sigma
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        self.dirs_cam = np.stack([(u - self.cx) / self.f, (v - self.cy) / self.f,
                                  np.ones_like(u, float)], axis=-1).reshape(-1, 3)

    def calibration(self):
        """Sensor self-knowledge that a real system could legitimately have (Q1)."""
        return {"width": self.w, "height": self.h, "fx": float(self.f), "fy": float(self.f),
                "cx": float(self.cx), "cy": float(self.cy), "baseline": float(self.baseline),
                "mount_pitch": float(self.pitch)}

    def render_view(self, surfaces, origin, psi):
        r, dn, fw = camera_axes(psi, self.pitch)
        dirs = self.dirs_cam[:, 0:1] * r + self.dirs_cam[:, 1:2] * dn + self.dirs_cam[:, 2:3] * fw
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        n = len(dirs)
        best_t = np.full(n, np.inf)
        best_s = np.full(n, -1)
        for i, s in enumerate(surfaces):
            t = s.intersect(origin, dirs)
            closer = t < best_t
            best_t[closer] = t[closer]
            best_s[closer] = i
        img = np.zeros((n, 3))
        gt = np.full(n, -1, dtype=np.int32)
        depth = np.full(n, np.inf)
        for i, s in enumerate(surfaces):
            m = best_s == i
            if not m.any():
                continue
            p = origin + dirs[m] * best_t[m, None]
            nrm, loc = s.normal_and_local(p)
            shade = self.ambient + (1 - self.ambient) * np.clip(nrm @ self.light, 0, None)
            tex = eval_texture(s.texture, loc)
            for (sc, sr) in s.spots:
                tex = np.where(np.linalg.norm(loc - sc, axis=1) < sr, tex * 0.42, tex)
            img[m] = s.albedo[None, :] * (tex * shade)[:, None]
            gt[m] = s.gt_id
            depth[m] = (p - origin) @ fw
        return (img.reshape(self.h, self.w, 3), gt.reshape(self.h, self.w),
                depth.reshape(self.h, self.w))

    def render_stereo(self, surfaces, pos_xz, psi, rng):
        """Returns (left uint8, right uint8, gt_ids_left, depth_left).
        Only the two uint8 images may be sent to the learner."""
        right_vec, _, _ = body_axes(psi)
        o_left = np.array([pos_xz[0], self.cam_height, pos_xz[1]])
        o_right = o_left + self.baseline * right_vec
        L, gt, depth = self.render_view(surfaces, o_left, psi)
        R, _, _ = self.render_view(surfaces, o_right, psi)
        out = []
        for im in (L, R):
            im = im * 230.0 + rng.normal(0, self.noise_sigma, im.shape)
            out.append(np.clip(np.round(im), 0, 255).astype(np.uint8))
        return out[0], out[1], gt, depth
