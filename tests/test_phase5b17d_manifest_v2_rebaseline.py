"""Contract tests for Phase 5B-1.7D; no full training is run here."""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_phase5b17d_manifest_v2_rebaseline as stage
from src.data.adverse_response_dataset import build_development_split
from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
from src.multimodal.phase5b_v2_dataset import DEPRECATED_HARM_TARGET, build_v2_temporal_samples
from src.training.candidate_ranking import LAMBDA_RANK, pairwise_logistic_ranking_loss

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_manifest_v2_hash_matches_and_v1_remains_present():
    v2 = ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json"
    assert hashlib.sha256(v2.read_bytes()).hexdigest() == stage.EXPECTED_MANIFEST_SHA
    assert (ROOT / "results_dev" / "phase5b05_c7_coverage" / "phase5b_manifest_v1.json").is_file()


def test_runtime_bridge_keeps_episode_branches_together_and_static_108():
    episodes = build_development_split("train", 2, 51_703, 77_117)
    samples = build_v2_temporal_samples(episodes)
    assert len(samples) == 10
    assert {sample.split for sample in samples} == {"train"}
    assert all(np.asarray(sample.split_metadata["static_context_108"]).shape == (108,) for sample in samples)
    assert all(len({sample.split for sample in samples if sample.episode_id == episode.episode_id}) == 1 for episode in episodes)
    assert not any(sample.split == "test" for sample in samples)


def test_runtime_bridge_does_not_use_label_side_natural_future():
    episode = build_development_split("train", 1, 51_703, 77_117)[0]
    baseline = build_v2_temporal_samples([episode])
    altered = copy.copy(episode)
    object.__setattr__(altered, "natural_future", np.full_like(episode.natural_future, 999.0))
    replay = build_v2_temporal_samples([altered])
    for left, right in zip(baseline, replay):
        assert np.array_equal(left.split_metadata["static_context_108"], right.split_metadata["static_context_108"])
        for name in left.streams: assert np.array_equal(left.streams[name], right.streams[name])


def test_test_split_builder_is_rejected_before_runtime_read():
    with pytest.raises(ValueError, match="TRAIN/VALIDATION"):
        build_development_split("test", 1, 1, 2)


def test_b1_r1_architecture_and_initial_checksum_are_identical():
    torch.manual_seed(42); b1 = RichTemporalSmallTransformer(); r1 = copy.deepcopy(b1)
    assert b1.architecture_audit() == r1.architecture_audit()
    assert stage.model_sha(b1) == stage.model_sha(r1)


def test_rank_configuration_is_frozen_and_ties_are_excluded():
    assert LAMBDA_RANK == .25
    predicted = torch.tensor((0.1, 0.2, -0.1, 0.3))
    target = torch.tensor((1.0, 1.0, 0.0, 2.0))
    feasible = torch.tensor((True, True, True, False))
    loss, audit = pairwise_logistic_ranking_loss(predicted, target, ("e", "e", "e", "e"), feasible)
    assert torch.isfinite(loss) and audit.pair_count == 2 and audit.feasible_candidate_count == 3


def test_old_harm_is_deprecated_and_harm_v2_never_enters_loss():
    source = inspect.getsource(stage.loss_terms)
    assert "DEPRECATED_AUXILIARY_CONTROL_TARGET" in DEPRECATED_HARM_TARGET
    assert "harm_v2" not in source
    assert "deprecated_old_harm" in source


def test_normalizer_rejects_validation_only_fit():
    episodes = build_development_split("validation", 1, 52_703, 78_117)
    with pytest.raises(ValueError, match="train"):
        stage.b1.fit_normalizers(build_v2_temporal_samples(episodes))


def test_checkpoint_protocol_and_test_read_contract_are_explicit():
    source = inspect.getsource(stage)
    assert "validation_selection_key" in source
    assert '"test_reads": 0' in source
    assert "formal_decision_gate_performed" in source
    assert "phase5b17e_started" in source
