"""Checks for the architectural guarantees and the isolation rules.
Run: python -m pytest tests -q
"""
import ast
import os

import numpy as np
import pytest

from learner import registry
from learner.config import VARIANTS
from learner.core import Learner, Record, Unit
from learner.mechanisms.identity import NEW, SAME, UNKNOWN_D, LeaveOneOutEvidence, TailRatioUsefulness
from learner.stores import KNOWN, BELIEVED, InferenceStore, ObservationStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = {"width": 32, "height": 24, "fx": 30.0, "fy": 30.0, "cx": 15.5, "cy": 11.5,
         "baseline": 0.12, "mount_pitch": -0.5}


def test_learner_never_imports_harness():
    for dp, _, fs in os.walk(os.path.join(ROOT, "learner")):
        for f in fs:
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(dp, f)).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for n in names:
                    assert not n.startswith("harness"), (f, n)
                    assert n.split(".")[0] in {"numpy", "scipy", "json", "hashlib", "time", "traceback",
                                               "copy", "dataclasses", "__future__", ""} or node.level > 0, (f, n)


def test_rejects_extra_observation_fields():
    L = Learner(CALIB, VARIANTS["main"])
    obs = {"tick": 0, "left": np.zeros((24, 32, 3), np.uint8), "right": np.zeros((24, 32, 3), np.uint8),
           "self_motion": {"d_forward": 0.0, "d_right": 0.0, "d_yaw": 0.0}, "gt_ids": None}
    with pytest.raises(ValueError):
        L.step(obs)


def test_rejects_extra_calibration_fields():
    with pytest.raises(ValueError):
        Learner({**CALIB, "object_count": 4}, VARIANTS["main"])


def test_observations_are_immutable():
    s = ObservationStore()
    oid = s.add({"tick": 3, "left": np.ones((2, 2, 3), np.uint8), "right": np.ones((2, 2, 3), np.uint8),
                 "self_motion": {"d_forward": 0.0, "d_right": 0.0, "d_yaw": 0.0}})
    with pytest.raises(ValueError):
        s.get(oid)["left"][0, 0, 0] = 9


def test_no_inference_is_known_and_provenance_required():
    st = InferenceStore()
    with pytest.raises(ValueError):
        st.add(0, "x", "m", ["obs:0"], KNOWN)
    with pytest.raises(ValueError):
        st.add(0, "x", "m", [], BELIEVED)
    with pytest.raises(ValueError):
        st.add(0, "x", "", ["obs:0"], BELIEVED)


def test_usefulness_starts_neutral():
    u = TailRatioUsefulness()
    assert u.score(np.array([]), np.array([]), 0.3) == 0.0


def _unit_with(kinds_w, kinds_b, rec_ev):
    u = Unit(1, 0, "NEW", "inf:1")
    for k, ws in kinds_w.items():
        for w in ws:
            u.add_w(k, w)
    for k, bs in kinds_b.items():
        for b in bs:
            u.add_b(k, b)
    u.records.append(Record(id="exp:1:0", unit=1, tick=0, obs="obs:0", link="creation", segment=0,
                            ev=rec_ev, inf="inf:2", npix=100))
    return u


class _Scalar:
    def distance(self, a, b):
        return abs(a - b)


def test_no_single_kind_is_definitive_both_directions():
    rec = LeaveOneOutEvidence(threshold=2.0)
    ctx = {"extractors": {"a": _Scalar(), "b": _Scalar()}, "usefulness": TailRatioUsefulness(), "kinds": ["a", "b"]}
    # kind 'a' strongly informative, kind 'b' has no samples (neutral)
    u = _unit_with({"a": [0.01] * 50}, {"a": [1.0] * 50}, {"a": 0.0, "b": 0.0})
    dec, S, sc, _ = rec.unit_decision([{"a": 0.0, "b": 0.0}], u, ctx)
    assert sc["a"] > 2.0 and abs(sc["b"]) < 1e-9 and dec == UNKNOWN_D      # match alone is not SAME
    dec, S, sc, _ = rec.unit_decision([{"a": 5.0, "b": 0.0}], u, ctx)
    assert sc["a"] < -2.0 and dec == UNKNOWN_D                              # mismatch alone is not NEW
    # both kinds informative and agreeing -> conclusions possible
    u2 = _unit_with({"a": [0.01] * 50, "b": [0.01] * 50}, {"a": [1.0] * 50, "b": [1.0] * 50}, {"a": 0.0, "b": 0.0})
    assert rec.unit_decision([{"a": 0.0, "b": 0.0}], u2, ctx)[0] == SAME
    assert rec.unit_decision([{"a": 5.0, "b": 5.0}], u2, ctx)[0] == NEW


def test_every_mechanism_is_labelled():
    for slot, d in registry.REGISTRY.items():
        for name, cls in d.items():
            assert cls.STATUS in ("EXPERIMENTAL", "BASELINE"), name
            assert cls.ASSUMPTIONS, name


def test_variants_build():
    for v, cfg in VARIANTS.items():
        Learner(CALIB, cfg)
