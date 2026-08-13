"""Protocol and math tests for Phase 5B-1.7E-C."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts import audit_phase5b17ec_representation_risk as audit
from src.evaluation.representation_risk_audit import (candidate_conditioning_distances,
    fixed_noisy_or, pairwise_discrimination)
from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer


def test_noisy_or_is_fixed_and_exact():
    values = np.asarray([[.1, .2, .3, .4], [0, 0, 0, 0], [1, .2, .3, .4]])
    actual = fixed_noisy_or(values)
    assert actual == pytest.approx([1 - .9 * .8 * .7 * .6, 0, 1])
    source = inspect.getsource(fixed_noisy_or)
    assert "np.prod" in source and "torch" not in source and "optimiz" not in source


@pytest.mark.parametrize("bad", [np.zeros((4, 3)), np.asarray([[0, 0, 0, 2]]), np.asarray([[0, 0, np.nan, 0]])])
def test_noisy_or_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError): fixed_noisy_or(bad)


def test_audit_interface_does_not_change_normal_forward():
    import torch
    from src.multimodal.temporal_collate import collate_temporal
    from tests.test_phase5b_temporal_schema import make_sample
    model = RichTemporalSmallTransformer().eval(); batch = collate_temporal([make_sample(), make_sample()], as_torch=True)
    with torch.inference_mode():
        normal = model.encode(batch); stages = model.audit_representations(batch)
    assert torch.equal(normal, stages["R0_FINAL_FUSED"])
    assert stages["R0_FINAL_FUSED"].shape[-1] == 128
    assert stages["R6_PREFINAL_FUSION_CONCAT"].shape[-1] == 384
    assert stages["R1_HISTORY_CONTEXT_PREFUSION"].shape[-1] == 1024
    assert stages["R2_CANDIDATE_PREFUSION"].shape[-1] == 256


def test_registry_reports_unavailable_stage_instead_of_inventing_it():
    registry = audit.stage_registry({})
    item = next(row for row in registry if row["representation_stage"] == "POST_TRANSFORMER_HUMAN_BEFORE_CANDIDATE_FUSION")
    assert item["available"] is False and item["provenance"].startswith("NOT AVAILABLE")


def test_all_probes_are_linear_train_only_and_unweighted():
    source = inspect.getsource(audit.train_linear)
    assert "torch.nn.Linear(width, 1)" in source
    assert "e.train_head" in source
    assert "MinimalNonlinear" not in inspect.getsource(audit)
    training = inspect.getsource(audit.e.train_head)
    assert "unweighted_harm_v2_loss" in training


def test_test_is_sealed_profile_is_audit_only_and_gt_future_absent():
    source = inspect.getsource(audit.main)
    assert '"test_reads": 0' in source
    cache = inspect.getsource(audit.save_cache)
    assert '"splits": ["train", "validation"]' in cache
    assert '"profile_id_in_probe_input": False' in cache
    assert '"GT_future_in_representation": False' in cache


def test_frozen_contract_has_all_checksums_and_no_optimizer_or_checkpoint():
    source = inspect.getsource(audit.main)
    for field in ("full_model_checksum_before", "temporal_backbone_checksum_unchanged",
                  "benefit_head_checksum_unchanged", "ranking_behavior_checksum_unchanged"):
        assert field in source
    assert '"optimizer_created_for_backbone": False' in source
    assert "torch.save" not in source
    assert '"formal_harm_checkpoint_written": False' in source


def test_subtype_labels_reuse_frozen_definitions():
    assert tuple(audit.SUBTYPE_PREDICATES) == audit.SUBTYPE_ORDER
    source = inspect.getsource(audit.SUBTYPE_PREDICATES["EXCESSIVE_DECELERATION"])
    assert "excessive_deceleration_evaluation_only" in source
    assert "threshold" not in source


def test_pairwise_discrimination_uses_only_mixed_episodes():
    result = pairwise_discrimination([.8, .2, .3, .4], [1, 0, 0, 0], ["a", "a", "b", "b"])
    assert result["mixed_episode_count"] == 1
    assert result["positive_negative_pair_count"] == 1
    assert result["pairwise_discrimination_accuracy"] == 1


def test_candidate_conditioning_distance_separates_label_change_pairs():
    result = candidate_conditioning_distances([[0, 0], [1, 0], [0, 0], [0, 1]], [0, 1, 0, 0], ["a", "a", "b", "b"])
    assert result["episodes_with_candidate_dependent_harm_label"] == 1
    assert result["different_label_mean_distance"] == 1
    assert result["same_label_mean_distance"] == 1


def test_protocol_forbids_formal_interventions():
    source = inspect.getsource(audit.main)
    assert '"threshold_calibration_performed": False' in source
    assert '"intervention_implemented": False' in source
    assert "class weighting" not in source.lower()


def test_safe_beneficial_gate_has_no_posthoc_tolerance():
    source = inspect.getsource(audit.main)
    assert "safe_mean_not_higher_than_global_linear" in source
    assert "safe_P90_not_higher_than_global_linear" in source
    assert "safe_P95_not_higher_than_global_linear" in source
    assert "increase_at_most" not in source
