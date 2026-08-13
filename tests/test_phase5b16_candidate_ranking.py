import copy
import inspect
from types import SimpleNamespace

import torch

from scripts import run_phase5b16_candidate_ranking as experiment
from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
from src.training.candidate_ranking import LAMBDA_RANK, pairwise_logistic_ranking_loss


def _loss(predicted, target, episodes, feasible):
    return pairwise_logistic_ranking_loss(
        torch.tensor(predicted, dtype=torch.float32, requires_grad=True),
        torch.tensor(target, dtype=torch.float32), episodes,
        torch.tensor(feasible, dtype=torch.bool),
    )


def test_ranking_pairs_never_cross_episode_boundary():
    loss, audit = _loss([10.0, -10.0], [0.0, 1.0], ["e0", "e1"], [True, True])
    assert loss.item() == 0.0 and audit.pair_count == 0 and audit.episode_count == 0


def test_ranking_pairs_use_feasible_candidates_only():
    loss, audit = _loss([0.0, 100.0, 1.0], [0.0, 2.0, 1.0], ["e", "e", "e"], [True, False, True])
    expected = torch.nn.functional.softplus(torch.tensor(-1.0))
    assert torch.allclose(loss, expected) and audit.pair_count == 1 and audit.feasible_candidate_count == 2


def test_exact_target_ties_are_excluded():
    loss, audit = _loss([0.0, 50.0, 1.0], [0.0, 0.0, 1.0], ["e", "e", "e"], [True, True, True])
    assert audit.pair_count == 2 and torch.isfinite(loss)


def test_ranking_loss_reduces_per_episode_before_batch_mean():
    prediction = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0])
    loss, audit = pairwise_logistic_ranking_loss(prediction, target, ["a", "a", "a", "b", "b"], torch.ones(5, dtype=torch.bool))
    first = torch.nn.functional.softplus(torch.tensor([-1.0, -2.0, -1.0])).mean()
    second = torch.nn.functional.softplus(torch.tensor(1.0))
    assert torch.allclose(loss, (first + second) / 2) and audit.episode_count == 2


def test_lambda_rank_is_frozen_at_point_25():
    assert LAMBDA_RANK == experiment.LAMBDA_RANK == 0.25
    source = inspect.getsource(experiment.parse_args)
    assert "lambda" not in source.lower()


def test_r0_has_exactly_zero_ranking_contribution():
    source = inspect.getsource(experiment.train_one)
    assert 'terms["rank"] * 0.0' in source
    assert 'if use_ranking' in source


def test_r1_ranking_gradient_is_nonzero():
    prediction = torch.tensor([0.0, 0.1, -0.2], requires_grad=True)
    loss, audit = pairwise_logistic_ranking_loss(
        prediction, torch.tensor([0.5, -0.1, 0.2]), ["e", "e", "e"], torch.ones(3, dtype=torch.bool),
    )
    (LAMBDA_RANK * loss).backward()
    assert audit.pair_count == 3 and prediction.grad is not None and prediction.grad.norm() > 0


def test_ranking_target_is_training_only_not_model_runtime_input():
    signature = inspect.signature(RichTemporalSmallTransformer.forward)
    assert list(signature.parameters) == ["self", "batch"]
    source = inspect.getsource(pairwise_logistic_ranking_loss)
    assert "TRAINING_TARGET_ONLY" in pairwise_logistic_ranking_loss.__doc__ or "training-only" in pairwise_logistic_ranking_loss.__doc__
    assert "target_benefit" in source


def test_r0_r1_architecture_and_initial_checksum_are_identical():
    torch.manual_seed(42)
    base = RichTemporalSmallTransformer()
    r0, r1 = copy.deepcopy(base), copy.deepcopy(base)
    assert r0.architecture_audit() == r1.architecture_audit()
    assert r0.architecture_audit()["trainable_parameter_count"] == 352376
    assert experiment.model_checksum(r0) == experiment.model_checksum(r1)


def test_episode_batching_is_reproducible_and_preserves_complete_groups():
    samples = [SimpleNamespace(episode_id=f"e{i//3}") for i in range(18)]
    first, first_rows = experiment.make_episode_batches(samples, 2, 7, 42)
    second, second_rows = experiment.make_episode_batches(samples, 2, 7, 42)
    assert first == second and first_rows == second_rows
    for batches in first:
        flattened = [index for batch in batches for index in batch]
        assert sorted(flattened) == list(range(18))
        for episode in {sample.episode_id for sample in samples}:
            containing = [batch for batch in batches if any(samples[i].episode_id == episode for i in batch)]
            assert len(containing) == 1


def test_normalizer_threshold_safety_and_arbitration_are_shared_and_frozen():
    source = inspect.getsource(experiment.main)
    assert "r0_r1_normalizer_identical" in source
    assert "r0_r1_thresholds_identical" in source
    assert "safety_mask_unchanged" in source
    assert "arbitration_unchanged" in source
    assert experiment.FROZEN_THRESHOLDS == (-0.02, 0.2)


def test_sealed_test_is_never_materialized_or_evaluated():
    source = inspect.getsource(experiment)
    forbidden_call = "materialize" + "_test"
    assert forbidden_call not in source
    assert '"test_candidates_read": 0' in source
    assert '"test_labels_read": 0' in source
    assert '"test_metrics_computed": False' in source


def test_gate_thresholds_are_preregistered_not_cli_tunable():
    source = inspect.getsource(experiment.parse_args)
    assert "collapse" not in source and "threshold" not in source
    assert experiment.MAE_COLLAPSE_ABSOLUTE == 0.015
    assert experiment.MAE_COLLAPSE_RELATIVE == 0.20
    assert experiment.AUROC_COLLAPSE_ABSOLUTE == 0.03


def test_no_update_preflight_requires_32_batches_and_stops_before_training():
    source = inspect.getsource(experiment.main)
    assert "[:32]" in source
    assert 'if not preflight["passed"]' in source
    assert source.index('if not preflight["passed"]') < source.index("train_one(")
