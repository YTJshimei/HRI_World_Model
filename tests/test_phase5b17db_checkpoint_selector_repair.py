"""Protocol and selector tests for Phase 5B-1.7D-B."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts import run_phase5b17d_manifest_v2_rebaseline as frozen
from scripts import run_phase5b17db_checkpoint_selector_repair as repair
from src.evaluation.cracs_selector import (
    MAX_MAE_RATIO, MAX_SIGN_DROP, annotate, calibration_limits,
    ranking_score, select_cracs, spearman_score,
)
from src.training.candidate_ranking import LAMBDA_RANK

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MAE = 1.9542271157957118
REFERENCE_SIGN = .6841666666666667


def row(epoch=1, mae=2.0, sign=.68, spearman=.6, pairwise=.7, top1=.6, top2=.9, bias=.1):
    return {"epoch": epoch, "Benefit_MAE": mae, "Benefit_Sign_Accuracy": sign,
            "mean_feasible_within_episode_spearman": spearman,
            "mean_feasible_pairwise_accuracy": pairwise, "gt_best_top1_accuracy": top1,
            "gt_best_top2_recall": top2, "mean_gt_best_rank": 1.5, "global_bias": bias}


def test_fixed_eligibility_calculation():
    limits = calibration_limits(REFERENCE_MAE, REFERENCE_SIGN)
    assert MAX_MAE_RATIO == 1.25 and MAX_SIGN_DROP == .05
    assert limits["max_mae"] == pytest.approx(2.4427838947446397)
    assert limits["min_sign_accuracy"] == pytest.approx(.6341666666666667)
    assert annotate(row(mae=limits["max_mae"], sign=limits["min_sign_accuracy"]), REFERENCE_MAE, REFERENCE_SIGN)["cracs_eligible"]
    assert not annotate(row(mae=limits["max_mae"] + 1e-8), REFERENCE_MAE, REFERENCE_SIGN)["cracs_eligible"]
    assert not annotate(row(sign=limits["min_sign_accuracy"] - 1e-8), REFERENCE_MAE, REFERENCE_SIGN)["cracs_eligible"]


@pytest.mark.parametrize(("value", "expected"), [(-1, 0), (0, .5), (1, 1)])
def test_spearman_maps_to_unit_interval(value, expected):
    assert spearman_score(value) == expected


def test_ranking_score_is_four_way_equal_weight():
    metrics = row(spearman=.2, pairwise=.4, top1=.6, top2=.8)
    assert ranking_score(metrics) == pytest.approx(((.2 + 1) / 2 + .4 + .6 + .8) / 4)


def test_ineligible_epoch_cannot_be_selected():
    bad = row(epoch=2, mae=100, spearman=1, pairwise=1, top1=1, top2=1)
    good = row(epoch=1)
    selected, _ = select_cracs([bad, good], REFERENCE_MAE, REFERENCE_SIGN)
    assert selected["epoch"] == 1


def test_tie_break_is_deterministic_mae_then_bias_then_epoch():
    equal_score = [row(epoch=4, mae=2.1, bias=.01), row(epoch=3, mae=2.0, bias=.5)]
    assert select_cracs(equal_score, REFERENCE_MAE, REFERENCE_SIGN)[0]["epoch"] == 3
    equal_mae = [row(epoch=4, bias=.2), row(epoch=3, bias=-.1)]
    assert select_cracs(equal_mae, REFERENCE_MAE, REFERENCE_SIGN)[0]["epoch"] == 3
    equal_all = [row(epoch=4), row(epoch=3)]
    assert select_cracs(equal_all, REFERENCE_MAE, REFERENCE_SIGN)[0]["epoch"] == 3


def test_nonfinite_metric_is_ineligible():
    assert not annotate(row(pairwise=float("nan")), REFERENCE_MAE, REFERENCE_SIGN)["cracs_eligible"]


def test_frozen_training_contract_and_only_selector_changes():
    source = inspect.getsource(repair)
    assert LAMBDA_RANK == .25
    assert '"optimizer": "AdamW"' in source and '"learning_rate": args.learning_rate' in source
    assert '"only_protocol_change": "validation checkpoint selector OLD -> CRACS-v1"' in source
    assert "d.loss_terms" in source and "b16.make_episode_batches" in source
    assert "RichTemporalSmallTransformer" in source
    assert "harm_v2" not in inspect.getsource(frozen.loss_terms)


def test_zero_test_reads_and_deprecated_harm_not_safety():
    source = inspect.getsource(repair)
    assert '"test_reads": 0' in source
    assert '"deprecated_harm_used_for_safety_conclusion": False' in source
    assert '"formal_safety_decision_conclusion_allowed": False' in source


def test_manifest_architecture_and_frozen_results_are_not_overwritten():
    assert frozen.EXPECTED_MANIFEST_SHA == "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"
    assert repair.parse_args.__module__.endswith("run_phase5b17db_checkpoint_selector_repair")
    assert (ROOT / "results_dev" / "phase5b17d_manifest_v2_rebaseline" / "summary.json").is_file()
    assert (ROOT / "results_dev" / "phase5b17da_ranking_failure_audit" / "summary.json").is_file()


def test_historical_counterfactual_cannot_write_formal_checkpoint():
    source = inspect.getsource(repair.historical_counterfactual)
    assert "torch.save" not in source
    assert "historical_epoch_cannot_be_promoted" in source
    assert "formal_checkpoint_written_from_historical_audit" in source


def test_formal_checkpoint_requires_all_three_gates():
    source = inspect.getsource(repair.save_formal_checkpoint)
    assert '("Gate_A", "Gate_B", "Gate_C")' in source
    assert "torch.save" in source


def test_generated_replay_proves_frozen_protocol_and_zero_test_reads():
    output = ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair"
    if not (output / "summary.json").is_file():
        pytest.skip("formal replay artifact is not present in this checkout")
    contract = json.loads((output / "frozen_contract.json").read_text(encoding="utf-8"))
    config = json.loads((output / "training_config.json").read_text(encoding="utf-8"))
    assert contract["test_reads"] == 0
    assert contract["manifest_sha256"] == frozen.EXPECTED_MANIFEST_SHA
    assert contract["all_frozen_artifacts_unchanged"]
    assert contract["initial_checksum_identical"] and contract["data_order_identical"]
    assert contract["training_trajectory"]["max_absolute_delta"] == 0.0
    assert config["lambda_rank"] == .25 and config["learning_rate"] == 3e-4
    assert config["optimizer"] == "AdamW" and not config["ranking_loss_definition_changed"]
    assert config["only_protocol_change"] == "validation checkpoint selector OLD -> CRACS-v1"


def test_generated_checkpoint_is_from_formal_replay_not_historical_oracle():
    output = ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair"
    if not (output / "summary.json").is_file():
        pytest.skip("formal replay artifact is not present in this checkout")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["historical_counterfactual"]["historical_epoch_cannot_be_promoted"]
    assert not summary["historical_counterfactual"]["formal_checkpoint_written_from_historical_audit"]
    assert summary["formal_replay"]["same_training_trajectory"]
    assert summary["gates"]["all_passed"] and summary["gates"]["checkpoint_frozen"]
