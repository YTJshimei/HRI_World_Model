import numpy as np
import pytest

from src.multimodal.temporal_collate import collate_temporal
from src.multimodal.temporal_dataset import (
    export_phase5a_static_108, fit_train_normalizer, static_bridge_audit, temporal_window, validate_split_isolation,
)
from src.multimodal.temporal_masks import left_pad, masked_values_equal
from src.multimodal.temporal_schema import HISTORY_WINDOWS
from tests.test_phase5b_temporal_schema import make_sample


def test_same_episode_cannot_cross_train_test():
    train = make_sample("train"); test = make_sample("test")
    object.__setattr__(test, "episode_id", train.episode_id)
    assert not validate_split_isolation([train, test], held_out_profiles=())["passed"]


def test_held_out_person_cannot_enter_train():
    assert not validate_split_isolation([make_sample("train", profile=2)])["passed"]


def test_train_normalizer_rejects_validation_and_retains_fit_ids():
    normalizer = fit_train_normalizer([make_sample("train")])
    assert normalizer.fit_split == "train" and normalizer.fit_sample_ids == ("id:0",)
    with pytest.raises(ValueError, match="only access train"):
        fit_train_normalizer([make_sample("validation")])


def test_missing_zero_has_mask_and_padding_does_not_change_real_values():
    values = np.arange(6, dtype=np.float32).reshape(3, 2); valid = np.ones_like(values, bool)
    padded, mask, padding = left_pad(values, valid, 5)
    changed = padded.copy(); changed[:2] = 99
    assert np.array_equal(padded[-3:], values) and not padding[:2].any()
    assert masked_values_equal(padded, changed, padding)
    assert not mask[:2].any()


def test_candidate_permutation_preserves_stream_shapes_and_ids():
    samples = []
    for action in (0, 1, 2):
        sample = make_sample(); sample.streams["candidate_action"][action] = 1
        object.__setattr__(sample, "sample_id", f"id:{action}"); samples.append(sample)
    original = collate_temporal(samples); order = [2, 0, 1]; permuted = collate_temporal([samples[i] for i in order])
    assert permuted["streams"]["candidate_action"].shape == original["streams"]["candidate_action"].shape
    assert np.array_equal(permuted["streams"]["candidate_action"], original["streams"]["candidate_action"][order])


def test_declared_windows_are_suffixes_and_never_include_future():
    sample = make_sample(); sample.streams["robot_history"][:, 0] = np.arange(20)
    for frames in HISTORY_WINDOWS.values():
        view = temporal_window(sample, frames)
        assert view["streams"]["robot_history"].shape[0] == frames
        assert view["timestamps"]["history"].max() <= 0


def test_static_bridge_is_exact_audit_metadata_only():
    sample = make_sample(); metadata = dict(sample.split_metadata); metadata["static_context_108"] = np.arange(108, dtype=np.float32)
    object.__setattr__(sample, "split_metadata", metadata)
    audit = static_bridge_audit([sample])
    assert audit["passed"] and not audit["runtime_input"]
    assert np.array_equal(export_phase5a_static_108(sample), metadata["static_context_108"])


def test_same_seed_construction_primitives_are_reproducible():
    rng1 = np.random.default_rng(42); rng2 = np.random.default_rng(42)
    assert np.array_equal(rng1.normal(size=(20, 17, 3)), rng2.normal(size=(20, 17, 3)))
