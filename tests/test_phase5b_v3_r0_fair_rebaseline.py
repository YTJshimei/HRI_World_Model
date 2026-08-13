"""Regression gates for the manifest-v3 fair model rebaseline."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts import run_phase5b_v3_r0_fair_rebaseline as stage
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.evaluation.cracs_selector import MAX_MAE_RATIO, MAX_SIGN_DROP
from src.evaluation.probabilistic_harm import phs_select
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.phase5b_v3_dataset import (
    RichTemporalSampleV3,
    build_v3_temporal_samples,
    v3_runtime_contract_audit,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def paired_samples():
    episodes = build_development_split("train", 2, GENERATOR_SEED, RISK_SEED)
    return episodes, build_v2_temporal_samples(episodes), build_v3_temporal_samples(episodes)


def test_manifest_v3_and_v2_frozen_checksums():
    v3 = ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json"
    v2 = ROOT / "results_dev/phase5b17c_adverse_response_expansion/phase5b_manifest_v2.json"
    assert hashlib.sha256(v3.read_bytes()).hexdigest() == stage.EXPECTED_V3_SHA
    assert hashlib.sha256(v2.read_bytes()).hexdigest() == stage.EXPECTED_V2_SHA


def test_v3_has_six_candidates_and_hold_in_development(paired_samples):
    _, _, samples = paired_samples
    audit = v3_runtime_contract_audit(samples)
    assert audit["passed"] and audit["all_action_shapes_valid"]
    assert audit["hold_sample_count"] == 2
    assert all(sample.streams["candidate_action"].shape == (12,) for sample in samples)
    assert {sample.split_metadata["candidate_action_id_audit"] for sample in samples} == {0, 1, 2, 3, 4, HOLD_ACTION_ID}


def test_shared_v2_v3_targets_and_runtime_information_are_exact(paired_samples):
    _, v2, v3 = paired_samples
    v3_by_id = {sample.sample_id: sample for sample in v3}
    for old in v2:
        new = v3_by_id[old.sample_id]
        assert new.targets == old.targets
        for name in old.streams:
            if name == "candidate_action":
                assert np.array_equal(new.streams[name][:7], old.streams[name][:7])
                assert np.array_equal(new.streams[name][8:], old.streams[name][7:])
            else:
                assert np.array_equal(new.streams[name], old.streams[name])
        assert np.array_equal(new.split_metadata["static_context_108"], old.split_metadata["static_context_108"])


def test_v3_builder_refuses_test_materialization():
    class FakeEpisode:
        split = "test"
    with pytest.raises(ValueError, match="refuses TEST"):
        build_v3_temporal_samples([FakeEpisode()])


def test_v3_model_only_changes_action_projection_and_forwards(paired_samples):
    torch = pytest.importorskip("torch")
    from scripts import run_phase5b1_static_vs_temporal as b1
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3

    _, _, samples = paired_samples
    normalizers = b1.fit_normalizers(samples)
    torch.manual_seed(42); v2 = RichTemporalSmallTransformer()
    torch.manual_seed(42); v3 = RichTemporalSmallTransformerV3()
    assert v3.action_projection.in_features == 12
    assert v3.architecture_audit()["parameter_count"] == sum(p.numel() for p in v2.parameters()) + 128
    output = v3(b1.temporal_batch(samples[:2], normalizers, torch, torch.device("cpu")))
    assert output.benefit_mean.shape == (2,)
    assert torch.isfinite(output.benefit_mean).all()


def test_cracs_and_phs_v1_contracts_unchanged():
    assert MAX_MAE_RATIO == 1.25
    assert MAX_SIGN_DROP == 0.05
    selected = phs_select([
        {"epoch": 1, "NLL": .5, "Brier": .2, "AUROC": .8},
        {"epoch": 2, "NLL": .4, "Brier": .3, "AUROC": .7},
    ])
    assert selected["epoch"] == 2


def test_stage_contains_no_threshold_decision_or_relative_advantage_execution():
    source = inspect.getsource(stage.main)
    assert "select_threshold" not in source
    assert "decision_evaluation" not in source
    assert "arbitrate" not in source
    assert "GenericAnchored" not in source
    assert "build_development_split(\"test\"" not in source
    assert stage.V3_CANDIDATE_ACTION_DIM == 12


def test_v3_shape_contract_rejects_unknown_action_shape(paired_samples):
    _, _, samples = paired_samples
    sample = samples[0]
    streams = dict(sample.streams); streams["candidate_action"] = np.zeros(11, np.float32)
    masks = dict(sample.masks); masks["candidate_action"] = np.ones(11, bool)
    with pytest.raises(ValueError, match="candidate_action"):
        RichTemporalSampleV3(
            streams, masks, sample.timestamps, sample.targets, sample.sample_id, sample.episode_id,
            sample.split, sample.context_split, sample.temporal_tags, sample.split_metadata,
        )
