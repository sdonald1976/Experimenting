"""Semantically neutral evidence extractors (Q6a). ALL EXPERIMENTAL.

Each extractor turns one candidate region into a value and defines a distance
between two values. Names are for humans; the learner treats them only as
"evidence kind 0..n". No extractor detects any human category.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..registry import Mechanism, register


class Evidence(Mechanism):
    def extract(self, ctx: dict, pix: np.ndarray):
        raise NotImplementedError

    def distance(self, a, b) -> float:
        raise NotImplementedError


@register("evidence")
class SpatialExtent(Evidence):
    NAME = "spatial_extent"
    VERSION = "1"
    DEFAULTS = {"lo_pct": 5.0, "hi_pct": 95.0, "min_points": 30}
    ASSUMPTIONS = [
        "Metric size of the VISIBLE surface: sorted spreads of the region's 3-D points along their "
        "principal axes (rotation-invariant). Needs the allowed calibration (Q1).",
        "Distance = mean |log ratio| of the three sorted spreads.",
    ]

    def extract(self, ctx, pix):
        P = ctx["world_points"].reshape(-1, 3)[pix]
        P = P[np.isfinite(P).all(axis=1)]
        if len(P) < self.p["min_points"]:
            return None
        Pc = P - np.median(P, axis=0)
        _, _, vt = np.linalg.svd(Pc - Pc.mean(0), full_matrices=False)
        proj = Pc @ vt.T
        lo, hi = np.percentile(proj, [self.p["lo_pct"], self.p["hi_pct"]], axis=0)
        return np.sort(hi - lo)[::-1] + 1e-3

    def distance(self, a, b):
        return float(np.mean(np.abs(np.log(a / b))))


@register("evidence")
class ChromaDistribution(Evidence):
    NAME = "chroma_distribution"
    VERSION = "1"
    DEFAULTS = {"bins": 16, "min_brightness": 30.0, "min_pixels": 30}
    ASSUMPTIONS = [
        "2-D histogram of pixel chromaticity (r, g fractions) over the region; brightness-invariant.",
        "Distance = Hellinger distance between histograms.",
    ]

    def extract(self, ctx, pix):
        c = ctx["chroma"].reshape(-1, 2)[pix]
        b = ctx["bright"].ravel()[pix]
        c = c[b > self.p["min_brightness"]]
        if len(c) < self.p["min_pixels"]:
            return None
        n = self.p["bins"]
        h, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=n, range=[[0, 1], [0, 1]])
        return np.sqrt(h.ravel() / h.sum())

    def distance(self, a, b):
        return float(np.linalg.norm(a - b) / np.sqrt(2))


@register("evidence")
class MetricLocalPatterns(Evidence):
    NAME = "metric_local_patterns"
    VERSION = "1"
    DEFAULTS = {"patch_m": 0.05, "samples": 9, "max_patches": 24, "min_std": 6.0}
    ASSUMPTIONS = [
        "Stable local visual patterns: grey-level patches of fixed PHYSICAL size (patch_m), "
        "resampled to samples x samples using stereo depth, so scale is normalised by distance.",
        "Patches must lie fully inside the region; the most varied (highest std) are kept.",
        "Each patch is zero-mean, unit-norm. Distance = 1 - symmetric mean best normalised "
        "cross-correlation between the two patch sets. Not rotation-invariant.",
        "Detects no category (not a 'scratch' or 'marking' detector): any stable pattern counts.",
    ]

    def extract(self, ctx, pix):
        H, W = ctx["gray"].shape
        Z = ctx["cam_points"][..., 2].ravel()[pix]
        Z = Z[np.isfinite(Z)]
        if len(Z) < 30:
            return None
        zmed = float(np.median(Z))
        size_px = ctx["fx"] * self.p["patch_m"] / zmed
        if size_px < 4:
            return None
        n = self.p["samples"]
        mask = np.zeros(H * W, bool)
        mask[pix] = True
        mask = mask.reshape(H, W)
        ys, xs = np.divmod(pix, W)
        stride = max(2, int(size_px / 2))
        gy = np.arange(ys.min(), ys.max() + 1, stride)
        gx = np.arange(xs.min(), xs.max() + 1, stride)
        cy, cx = np.meshgrid(gy, gx, indexing="ij")
        cy, cx = cy.ravel(), cx.ravel()
        inside = mask[cy, cx]
        cy, cx = cy[inside], cx[inside]
        if len(cy) == 0:
            return None
        off = (np.arange(n) - (n - 1) / 2) * size_px / (n - 1)
        oy, ox = np.meshgrid(off, off, indexing="ij")
        sy = cy[:, None] + oy.ravel()[None]
        sx = cx[:, None] + ox.ravel()[None]
        ri, rj = np.round(sy).astype(int), np.round(sx).astype(int)
        inb = (ri >= 0) & (ri < H) & (rj >= 0) & (rj < W)
        full = inb.all(axis=1)
        full[full] = mask[ri[full], rj[full]].all(axis=1)
        if not full.any():
            return None
        sy, sx = sy[full], sx[full]
        vals = ndimage.map_coordinates(ctx["gray"], [sy.ravel(), sx.ravel()], order=1).reshape(sy.shape)
        sd = vals.std(axis=1)
        order = np.argsort(-sd)[: self.p["max_patches"]]
        order = order[sd[order] >= self.p["min_std"]]
        if len(order) == 0:
            return None
        v = vals[order] - vals[order].mean(axis=1, keepdims=True)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def distance(self, a, b):
        S = a @ b.T
        return float(1 - 0.5 * (S.max(axis=1).mean() + S.max(axis=0).mean()))


@register("evidence")
class LocationInSelfFrame(Evidence):
    NAME = "location_self_frame"
    VERSION = "1"
    DEFAULTS = {"min_points": 30}
    ASSUMPTIONS = [
        "Median 3-D point of the region in the learner's own dead-reckoned frame (drifts).",
        "Distance = Euclidean. Location is evidence, never definitive (Q6).",
    ]

    def extract(self, ctx, pix):
        P = ctx["world_points"].reshape(-1, 3)[pix]
        P = P[np.isfinite(P).all(axis=1)]
        if len(P) < self.p["min_points"]:
            return None
        return np.median(P, axis=0)

    def distance(self, a, b):
        return float(np.linalg.norm(a - b))


@register("evidence")
class MetricLocalPatternsHighPass(MetricLocalPatterns):
    """v2 of the local-pattern extractor, added after DEV diagnosis: v1 patches were dominated by
    smooth shading gradients and face edges that every sphere / box shares, so the 'pattern'
    evidence acted as a second shape cue. v2 removes structure smoother than the patch scale."""
    NAME = "metric_local_patterns_highpass"
    VERSION = "2"
    DEFAULTS = {**MetricLocalPatterns.DEFAULTS, "highpass_sigma_frac": 0.35}
    ASSUMPTIONS = MetricLocalPatterns.ASSUMPTIONS + [
        "Before sampling, grey levels are high-pass filtered: subtract a Gaussian blur with sigma = "
        "highpass_sigma_frac * patch size (in pixels, from the region's median depth). Smooth shading "
        "is removed; local marks and fine texture remain. Sharp shading steps at face edges remain "
        "(known weakness).",
    ]

    def extract(self, ctx, pix):
        H, W = ctx["gray"].shape
        Z = ctx["cam_points"][..., 2].ravel()[pix]
        Z = Z[np.isfinite(Z)]
        if len(Z) < 30:
            return None
        size_px = ctx["fx"] * self.p["patch_m"] / float(np.median(Z))
        ys, xs = np.divmod(pix, W)
        m = int(3 * size_px) + 2
        y0, y1 = max(0, ys.min() - m), min(H, ys.max() + m + 1)
        x0, x1 = max(0, xs.min() - m), min(W, xs.max() + m + 1)
        crop = ctx["gray"][y0:y1, x0:x1]
        hp = crop - ndimage.gaussian_filter(crop, self.p["highpass_sigma_frac"] * size_px)
        g = ctx["gray"].copy()
        g[y0:y1, x0:x1] = hp
        sub = dict(ctx)
        sub["gray"] = g
        return super().extract(sub, pix)
