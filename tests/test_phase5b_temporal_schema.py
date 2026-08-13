import numpy as np
import pytest

from src.multimodal.temporal_schema import (
    FORBIDDEN_RUNTIME_KEYS, HISTORY_FRAMES, RichTemporalSample, STREAM_DIMS, STREAM_ORDER,
    TemporalTargets, feature_registry, runtime_payload,
)


def make_sample(split="train", profile=0):
    streams = {name: np.zeros(shape, np.float32) for name, shape in STREAM_DIMS.items()}
    masks = {name: np.ones(shape, bool) for name, shape in STREAM_DIMS.items()}
    masks.update({"history_valid_mask": np.ones(HISTORY_FRAMES, bool), "history_padding_mask": np.ones(HISTORY_FRAMES, bool),
                  "candidate_future_valid_mask": np.ones(10, bool)})
    return RichTemporalSample(streams, masks, {"history": np.arange(-19, 1, dtype=np.float32) / 10,
                                              "candidate_future": np.arange(1, 11, dtype=np.float32) / 10},
                              TemporalTargets(0., False, 0., False, True, 0., False), "id:0", f"{split}:episode", split,
                              "C1_seen_motion_seen_action", (), {"person_profile_id": profile})


def test_stream_order_and_shapes_are_deterministic():
    sample = make_sample()
    assert tuple(sample.streams) == STREAM_ORDER
    assert {name: value.shape for name, value in sample.streams.items()} == STREAM_DIMS


def test_history_does_not_cross_decision_time():
    sample = make_sample()
    assert sample.timestamps["history"].max() == 0
    assert sample.timestamps["candidate_future"].min() > 0


def test_runtime_payload_excludes_targets_and_person_identity():
    payload = runtime_payload(make_sample())
    keys = set(payload) | set(payload["streams"])
    assert not keys & FORBIDDEN_RUNTIME_KEYS
    assert "targets" not in payload and "split_metadata" not in payload


def test_registry_marks_theta_and_human_future_boundaries():
    rows = {row["stream"]: row for row in feature_registry()}
    assert rows["functional_history"]["oracle_theta_forbidden"]
    assert rows["candidate_robot_future"]["gt_human_response_forbidden"]
    assert rows["targets"]["availability"] == "TRAINING_TARGET_ONLY"


def test_future_human_timestamp_is_rejected():
    sample = make_sample()
    with pytest.raises(ValueError, match="may not cross decision time"):
        RichTemporalSample(sample.streams, sample.masks, {"history": np.ones(20), "candidate_future": np.arange(1, 11)/10},
                           sample.targets, sample.sample_id, sample.episode_id, sample.split, sample.context_split,
                           sample.temporal_tags, sample.split_metadata)
