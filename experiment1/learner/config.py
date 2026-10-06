"""Learner configurations. Choosing mechanisms here is the experimenter's
choice; nothing here comes from ground truth. Every value is logged per run."""
import copy

EVIDENCE = [
    {"name": "spatial_extent"},
    {"name": "chroma_distribution"},
    {"name": "metric_local_patterns_highpass"},  # v2 after dev diagnosis; v1 kept registered
    {"name": "location_self_frame"},
]

MAIN = {
    "self_motion": {"name": "dead_reckoning"},
    "stereo": {"name": "sad_block_matching"},
    "candidates": {"name": "depth_chroma_boundaries"},
    "continuity": {"name": "overlap_continuity"},
    "sampling": {"name": "fixed_interval"},
    "evidence": EVIDENCE,
    "usefulness": {"name": "tail_ratio_matched"},  # v2 after dev diagnosis
    "recognition": {"name": "leave_one_out_evidence_median"},  # v2 after dev diagnosis
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

# Threshold sweep variants: dev-only, used to choose the threshold before the sealed test.
for _t in (5, 8):
    VARIANTS[f"main_threshold_{_t}"] = _variant(
        recognition={"name": "leave_one_out_evidence_median", "params": {"threshold": float(_t)}})
