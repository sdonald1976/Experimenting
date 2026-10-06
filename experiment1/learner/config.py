"""Learner configurations. Choosing mechanisms here is the experimenter's
choice; nothing here comes from ground truth. Every value is logged per run."""
import copy

EVIDENCE = [
    {"name": "spatial_extent"},
    {"name": "chroma_distribution"},
    {"name": "metric_local_patterns"},
    {"name": "location_self_frame"},
]

MAIN = {
    "self_motion": {"name": "dead_reckoning"},
    "stereo": {"name": "sad_block_matching"},
    "candidates": {"name": "depth_chroma_boundaries"},
    "continuity": {"name": "overlap_continuity"},
    "sampling": {"name": "fixed_interval"},
    "evidence": EVIDENCE,
    "usefulness": {"name": "tail_ratio"},
    "recognition": {"name": "leave_one_out_evidence"},
    "revision": {"name": "young_unit_reevaluation"},
}


def _variant(**changes):
    c = copy.deepcopy(MAIN)
    c.update(copy.deepcopy(changes))
    return c


VARIANTS = {
    "main": MAIN,
    # ablation: identical, but location evidence removed
    "ablation_no_location": _variant(evidence=[e for e in EVIDENCE if e["name"] != "location_self_frame"]),
    # trivial baselines: same perception, recognition replaced, no revision
    "baseline_always_new": _variant(recognition={"name": "baseline_always_new"},
                                    revision={"name": "no_revision"}),
    "baseline_nearest_location": _variant(recognition={"name": "baseline_nearest_location"},
                                          revision={"name": "no_revision"}),
    "baseline_most_recent": _variant(recognition={"name": "baseline_most_recent"},
                                     revision={"name": "no_revision"}),
    "baseline_random": _variant(recognition={"name": "baseline_random_existing"},
                                revision={"name": "no_revision"}),
}
