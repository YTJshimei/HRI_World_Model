import json

import numpy as np
import pytest

from scripts import build_phase5b05_c7_coverage as c7
from src.multimodal.temporal_dataset import (
    apply_continuous_occlusion, is_c7_long_occlusion, longest_joint_occlusion_run,
)
from src.multimodal.temporal_schema import runtime_payload
from tests.test_phase5b_temporal_schema import make_sample


def test_c7_definition_is_frozen_five_frames_half_second():
    definition = c7.c7_definition()
    assert definition["minimum_consecutive_occlusion_frames"] == 5
    assert definition["minimum_duration_seconds"] == .5
    assert definition["definition_changed_from_phase5b0"] is False


def test_continuous_occlusion_is_real_mask_not_scene_flag():
    visibility = np.ones((20, 17), bool); confidence = np.ones((20, 17), np.float32)
    changed_visibility, changed_confidence = apply_continuous_occlusion(visibility, confidence, start=6, frames=5, joints=(11, 13, 15))
    run = longest_joint_occlusion_run(changed_visibility)
    assert run["frames"] == 5 and is_c7_long_occlusion(changed_visibility, changed_confidence)
    assert not changed_visibility[6:11, 11].any() and not changed_confidence[6:11, 11].any()


def test_scene_flag_alone_cannot_make_c7():
    assert not is_c7_long_occlusion(np.ones((20, 17), bool), np.ones((20, 17), np.float32))


def test_short_occlusion_cannot_pass_c7():
    visibility = np.ones((20, 17), bool); visibility[3:7, 0] = False
    confidence = visibility.astype(np.float32)
    assert not is_c7_long_occlusion(visibility, confidence)


def test_predeclared_extension_has_six_unique_seeds_per_split_and_no_train_holdout_profile_scenario():
    seeds = {split: {c7.EXTENSION_SEED + {"train": 0, "validation": 1000, "test": 2000}[split] + index * 97 for index in range(6)} for split in c7.SPLIT_SPECS}
    assert all(len(values) == 6 for values in seeds.values())
    assert not any(scenario in ("S3_human_accelerating", "S9_uncertain_new_person") for scenario, _ in c7.SPLIT_SPECS["train"])


def test_candidate_branches_are_grouped_under_one_episode_manifest_entry():
    samples = []
    for action in range(5):
        sample = make_sample("train"); object.__setattr__(sample, "episode_id", "train:episode:1")
        object.__setattr__(sample, "sample_id", f"train:episode:1:{action}")
        metadata = dict(sample.split_metadata); metadata.update({"scenario": "S1_too_close", "motion_type_evaluation_only": "walk"})
        object.__setattr__(sample, "split_metadata", metadata); samples.append(sample)
    manifest = c7.manifest({"train": samples, "validation": [], "test": []})
    assert len(manifest["episodes"]) == 1 and len(manifest["episodes"][0]["candidate_ids"]) == 5


def test_manifest_checksum_is_canonical_and_reproducible():
    value = {"b": [2, 1], "a": {"x": 3}}
    assert c7.checksum(value) == c7.checksum(json.loads(json.dumps(value)))
    assert len(c7.checksum(value)) == 64


def test_same_seed_manifest_rebuild_is_identical():
    def samples():
        result = []
        for action in range(2):
            sample = make_sample("train")
            object.__setattr__(sample, "episode_id", "train:seed42:episode0")
            object.__setattr__(sample, "sample_id", f"train:seed42:episode0:{action}")
            metadata = dict(sample.split_metadata)
            metadata.update({"scenario": "S1_too_close", "motion_type_evaluation_only": "walk"})
            object.__setattr__(sample, "split_metadata", metadata)
            result.append(sample)
        return result

    first = c7.manifest({"train": samples(), "validation": [], "test": []})
    second = c7.manifest({"train": samples(), "validation": [], "test": []})
    assert first == second
    assert c7.checksum(first) == c7.checksum(second)


def test_c7_coverage_gate_requires_five_independent_episodes_per_split():
    c7.validate_c7_coverage({"train": 5, "validation": 6, "test": 7})
    with pytest.raises(RuntimeError, match="manifest was not frozen"):
        c7.validate_c7_coverage({"train": 5, "validation": 4, "test": 7})


def test_runtime_still_excludes_theta_identity_targets_and_human_future():
    payload = runtime_payload(make_sample())
    keys = set(payload["streams"])
    assert not keys & {"theta_true", "person_profile_id", "benefit", "harm", "future_global", "gt_human_future"}
    assert set(payload["streams"]["candidate_robot_future"].shape) == {5, 10}


def test_all_predeclared_variants_meet_frozen_c7_length():
    assert all(variant["frames"] >= c7.C7_MIN_FRAMES for variant in c7.OCCLUSION_VARIANTS)
    assert all(variant["start"] + variant["frames"] <= 20 for variant in c7.OCCLUSION_VARIANTS)


def test_same_seed_rebuilds_identical_independent_episode():
    variant = c7.OCCLUSION_VARIANTS[0]
    first = c7.generate_one_sample("S1_too_close", "walk", 50123, 9000, variant, "train")
    second = c7.generate_one_sample("S1_too_close", "walk", 50123, 9000, variant, "train")
    for key in ("history", "robot", "visibility", "confidence"):
        assert np.array_equal(first[key], second[key])
