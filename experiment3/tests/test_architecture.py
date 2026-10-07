"""Architecture and isolation checks for Experiment 2.  python -m pytest tests -q"""
import ast
import os

import numpy as np
import pytest

from learner import registry
from learner.config import VARIANTS
from learner.core import Learner, Record, Unit
from learner.mechanisms.identity import NEW, SAME, UNKNOWN_D, WeightedVote, WindowedAUCUsefulness
from learner.stores import BELIEVED, KNOWN, InferenceStore, ObservationStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB = {"width": 32, "height": 24, "fx": 30.0, "fy": 30.0, "cx": 15.5, "cy": 11.5,
         "baseline": 0.12, "mount_pitch": -0.5}


def test_learner_never_imports_harness():
    for dp, _, fs in os.walk(os.path.join(ROOT, "learner")):
        for f in fs:
            if f.endswith(".py"):
                for node in ast.walk(ast.parse(open(os.path.join(dp, f)).read())):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        assert not node.module.startswith("harness"), f
                    if isinstance(node, ast.Import):
                        assert not any(a.name.startswith("harness") for a in node.names), f


def test_rejects_extra_observation_and_calibration_fields():
    L = Learner(CALIB, VARIANTS["B_long"])
    obs = {"tick": 0, "left": np.zeros((24, 32, 3), np.uint8), "right": np.zeros((24, 32, 3), np.uint8),
           "self_motion": {"d_forward": 0.0, "d_right": 0.0, "d_yaw": 0.0}, "phase": 2}
    with pytest.raises(ValueError):
        L.step(obs)
    with pytest.raises(ValueError):
        Learner({**CALIB, "phase": 2}, VARIANTS["B_long"])


def test_observations_immutable_and_no_known_inference():
    s = ObservationStore()
    oid = s.add({"tick": 3, "left": np.ones((2, 2, 3), np.uint8), "right": np.ones((2, 2, 3), np.uint8),
                 "self_motion": {"d_forward": 0.0, "d_right": 0.0, "d_yaw": 0.0}})
    with pytest.raises(ValueError):
        s.get(oid)["left"][0, 0, 0] = 9
    st = InferenceStore()
    with pytest.raises(ValueError):
        st.add(0, "x", "m", ["obs:0"], KNOWN)
    with pytest.raises(ValueError):
        st.add(0, "x", "m", [], BELIEVED)


class _Scalar:
    def distance(self, a, b):
        return abs(a - b)


def _unit(uid, w=None, b=None):
    u = Unit(uid, 0, "NEW", "inf:1")
    for k, ws in (w or {}).items():
        for i, d in enumerate(ws):
            u.add_sample("W", k, d, i, f"r{i}", f"p{i}")
    for k, bs in (b or {}).items():
        for i, d in enumerate(bs):
            u.add_sample("B", k, d, i, f"q{i}", f"s{i}")
    u.records.append(Record(id=f"exp:{uid}:0", unit=uid, tick=0, obs="obs:0", link="creation", segment=0,
                            ev={"a": 0.0, "b": 0.0, "c": 0.0}, inf="inf:2", npix=100))
    return u


def test_usefulness_starts_uncommitted_and_is_local():
    U = WindowedAUCUsefulness()
    u1, u2 = _unit(1), _unit(2)
    assert U.usefulness(u1, "a") == 0.0 and U.vote(u1, "a", 0.3) == 0.0
    for i in range(40):
        u2.add_sample("W", "a", 0.01, i, "r", "p")
        u2.add_sample("B", "a", 1.0, i, "q", "s")
    assert U.usefulness(u2, "a") > 0.5
    assert U.usefulness(u1, "a") == 0.0          # learning about unit 2 did not touch unit 1


def test_recency_window_allows_revision_but_history_is_kept():
    U = WindowedAUCUsefulness(w_window=10, b_window=20)
    u = _unit(1)
    for i in range(40):
        u.add_sample("W", "a", 0.01, i, "r", "p")
        u.add_sample("B", "a", 1.0, i, "q", "s")
    high = U.usefulness(u, "a")
    for i in range(40, 60):                       # the kind stops separating same from different
        u.add_sample("W", "a", 1.5, i, "r", "p")
    assert U.usefulness(u, "a") < high * 0.2
    assert len(u.W["a"]) == 60                    # nothing overwritten


def _ctx(kinds):
    return {"extractors": {k: _Scalar() for k in kinds}, "usefulness": WindowedAUCUsefulness(), "kinds": kinds}


def _informative(u, k, n=40):
    for i in range(n):
        u.add_sample("W", k, 0.01, i, "r", "p")
        u.add_sample("B", k, 1.0, i, "q", "s")


def test_learned_weights_without_basis_give_unknown():
    rec = WeightedVote(weights="learned")
    u = _unit(1)
    dec, *_ = rec.unit_decision([{"a": 0.0, "b": 0.0, "c": 0.0}], u, _ctx(["a", "b", "c"]))
    assert dec == UNKNOWN_D


def test_learned_weight_lets_a_reliable_kind_outweigh_unreliable_agreement():
    """Unit's experience: 'a' separates same/different; 'b','c' do not (W and B overlap).
    Query: a says different, b and c look 'same'. Learned -> not SAME; equal -> may be fooled."""
    u = _unit(1)
    _informative(u, "a")
    for k in ("b", "c"):
        for i in range(40):
            u.add_sample("W", k, 0.01 * (i % 10), i, "r", "p")
            u.add_sample("B", k, 0.01 * (i % 10), i, "q", "s")
    q = [{"a": 2.0, "b": 0.0, "c": 0.0}]
    learned, *_ = WeightedVote(weights="learned").unit_decision(q, u, _ctx(["a", "b", "c"]))
    assert learned != SAME


def test_no_single_kind_is_definitive_for_every_weighting():
    for mode in ("learned", "fixed", "equal"):
        rec = WeightedVote(weights=mode, fixed_weights={"a": 0.5, "b": 0.5})
        u = _unit(1)
        _informative(u, "a")                      # only 'a' carries information
        dec, *_ = rec.unit_decision([{"a": 0.0, "b": 0.7}], u, _ctx(["a", "b"]))
        assert dec != SAME, mode
        dec, *_ = rec.unit_decision([{"a": 5.0, "b": 0.7}], u, _ctx(["a", "b"]))
        assert dec != NEW, mode


def test_every_mechanism_is_labelled_and_variants_build():
    for d in registry.REGISTRY.values():
        for name, cls in d.items():
            assert cls.STATUS in ("EXPERIMENTAL", "BASELINE") and cls.ASSUMPTIONS, name
    for cfg in VARIANTS.values():
        Learner(CALIB, cfg)


def test_long_span_samples_never_cross_a_segment_boundary():
    """Long-span W samples pair only records of ONE uninterrupted continuity segment. A recognition link
    starts a new segment, so an inferred identity can never become a 'same thing' training sample."""
    from learner.mechanisms.identity import ContinuitySpanUsefulness
    U = ContinuitySpanUsefulness(span="long", long_min_gap_ticks=30)
    u = Unit(1, 0, "NEW", "inf:1")
    mk = lambda i, tick, seg, link, x: Record(id=f"exp:1:{i}", unit=1, tick=tick, obs="o", link=link, segment=seg,
                                              ev={"a": x}, inf="i", npix=100)
    u.records += [mk(0, 0, 0, "creation", 0.0), mk(1, 40, 0, "continuity", 0.1),      # segment 0
                  mk(2, 200, 1, "recognition", 5.0), mk(3, 215, 1, "continuity", 5.1)]  # segment 1 starts by recognition
    new = mk(4, 240, 1, "continuity", 5.2)
    u.records.append(new)
    s = U.within_samples(u, new, {"a": _Scalar()})
    partners = {p for _, _, p, span in s}
    assert partners <= {"exp:1:2"}          # only the same-segment record >= 30 ticks older
    assert all(span == "long" for *_, span in s)
    short = ContinuitySpanUsefulness(span="short").within_samples(u, new, {"a": _Scalar()})
    assert {p for _, _, p, _ in short} <= {"exp:1:2", "exp:1:3"}
