"""Experiment 2 learner configurations. Variants differ ONLY in how evidence kinds are weighted
(A/B/C) or replace recognition with a shortcut (D1/D2). Every value is logged per run."""
import copy

EVIDENCE = [
    {"name": "spatial_extent"},
    {"name": "chroma_distribution"},
    {"name": "metric_local_patterns_highpass"},
    {"name": "location_self_frame"},
]

BASE = {
    "self_motion": {"name": "dead_reckoning"},
    "stereo": {"name": "sad_block_matching"},
    "candidates": {"name": "depth_chroma_boundaries"},
    "continuity": {"name": "overlap_continuity"},
    "sampling": {"name": "fixed_interval"},
    "evidence": EVIDENCE,
    "usefulness": {"name": "windowed_auc"},
    "recognition": {"name": "weighted_vote", "params": {"weights": "learned"}},
    "revision": {"name": "young_unit_reevaluation"},
}


def _variant(**changes):
    c = copy.deepcopy(BASE)
    c.update(copy.deepcopy(changes))
    return c


VARIANTS = {
    "A_learned": BASE,
    "B_fixed": _variant(recognition={"name": "weighted_vote", "params": {"weights": "fixed"}}),
    "C_equal": _variant(recognition={"name": "weighted_vote", "params": {"weights": "equal"}}),
    "D1_location_only": _variant(recognition={"name": "baseline_nearest_location"},
                                 revision={"name": "no_revision"}),
    "D2_always_new": _variant(recognition={"name": "baseline_always_new"}, revision={"name": "no_revision"}),
}
