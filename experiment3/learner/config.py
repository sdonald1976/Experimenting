"""Experiment 3 learner configurations. The primary variants (A/B/C) differ ONLY in which continuity
comparisons produce "same thing" samples (the temporal learning signal). Every value is logged per run."""
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
    "usefulness": {"name": "continuity_span", "params": {"span": "short"}},
    "recognition": {"name": "weighted_vote", "params": {"weights": "learned"}},
    "revision": {"name": "young_unit_reevaluation"},
}


def _variant(**changes):
    c = copy.deepcopy(BASE)
    c.update(copy.deepcopy(changes))
    return c


def _span(span):
    return {"name": "continuity_span", "params": {"span": span}}


VARIANTS = {
    # primary comparison: learned weights, three temporal learning signals
    "A_short": _variant(usefulness=_span("short")),          # = Experiment 2's A
    "B_long": _variant(usefulness=_span("long")),
    "C_both": _variant(usefulness=_span("both")),
    # attribution control: long-span samples but equal weights (separates the vote effect from the weight effect)
    "E_long_equal": _variant(usefulness=_span("long"),
                             recognition={"name": "weighted_vote", "params": {"weights": "equal"}}),
    # Experiment 2 baselines
    "D0_short_equal": _variant(recognition={"name": "weighted_vote", "params": {"weights": "equal"}}),
    "D1_location_only": _variant(recognition={"name": "baseline_nearest_location"}, revision={"name": "no_revision"}),
    "D2_always_new": _variant(recognition={"name": "baseline_always_new"}, revision={"name": "no_revision"}),
}
