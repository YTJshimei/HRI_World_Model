import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.decision.action_selector import select_model_action
from src.decision.safety_calibration import (
    SAFETY_FEATURE_DIM, SafetyCalibration, SafetyResidualHead,
    apply_safety_calibration, worst_case_regret,
)
from src.decision.safety_gate import (
    choose_fallback_action, risk_aware_candidate_mask,
)
from src.decision.safety_targets import (
    build_safety_targets_for_training_or_evaluation, false_safe_rate,
)


def test_gt_safety_target_builder_is_training_evaluation_only() -> None:
    distance = np.asarray(((1.0, 0.7, 0.9), (1.2, 1.1, 1.0)), dtype=np.float32)
    target = build_safety_targets_for_training_or_evaluation(distance, 0.8, 10.0)
    assert target.violation_any.tolist() == [True, False]
    assert "gt" not in inspect.signature(select_model_action).parameters
    assert "minimum_distance" not in inspect.signature(select_model_action).parameters


def test_calibration_rejects_nonvalidation_split() -> None:
    with pytest.raises(ValueError):
        SafetyCalibration(1.0, 1.0, 1.0, 0.5, 1.64, "test")


def test_lcb_gate_logic() -> None:
    allowed, reasons = risk_aware_candidate_mask(
        np.ones(2, dtype=bool), np.asarray((1.0, 1.0)), np.asarray((0.05, 0.20)),
        np.asarray((0.1, 0.1)), 0.8, 0.8, 1.64,
    )
    assert allowed[0] and not allowed[1]
    assert reasons[1] == "distance_lcb_below_threshold"


def test_probability_gate_logic() -> None:
    allowed, reasons = risk_aware_candidate_mask(
        np.ones(2, dtype=bool), np.ones(2), np.zeros(2),
        np.asarray((0.2, 0.9)), 0.8, 0.75, 1.64,
    )
    assert allowed.tolist() == [True, False]
    assert reasons[1] == "unsafe_probability_above_threshold"


def test_invalid_candidate_remains_hard_veto() -> None:
    allowed, reasons = risk_aware_candidate_mask(
        np.asarray((False, True)), np.ones(2), np.zeros(2),
        np.zeros(2), 0.8, 0.5, 1.64,
    )
    assert not allowed[0] and reasons[0] == "candidate_marked_infeasible"


def test_safe_candidate_not_all_rejected() -> None:
    allowed, _ = risk_aware_candidate_mask(
        np.ones(3, dtype=bool), np.asarray((0.7, 1.1, 1.2)),
        np.asarray((0.2, 0.03, 0.04)), np.asarray((0.9, 0.1, 0.2)),
        0.8, 0.7, 1.64,
    )
    assert allowed.any()


@pytest.mark.parametrize(
    "policy,expected",
    (("FALLBACK_KEEP", 0), ("FALLBACK_RULE_SAFE", 3), ("FALLBACK_MIN_RISK", 1)),
)
def test_fallback_policies(policy, expected) -> None:
    selected = choose_fallback_action(
        policy, np.asarray((0, 1, 2, 3, 4)), np.ones(5, dtype=bool),
        1.0, 1.5, np.asarray((0.4, 0.1, 0.8, 0.2, 0.9)),
    )
    assert selected == expected


def test_risk_gate_candidate_permutation_invariance() -> None:
    feasible = np.asarray((True, False, True)); distance = np.asarray((1.0, 1.2, 0.7))
    sigma = np.asarray((0.05, 0.01, 0.02)); probability = np.asarray((0.1, 0.1, 0.9))
    first, _ = risk_aware_candidate_mask(feasible, distance, sigma, probability, 0.8, 0.7, 1.64)
    permutation = np.asarray((2, 0, 1))
    second, _ = risk_aware_candidate_mask(
        feasible[permutation], distance[permutation], sigma[permutation],
        probability[permutation], 0.8, 0.7, 1.64,
    )
    np.testing.assert_array_equal(first, second[np.argsort(permutation)])


def test_distance_residual_head_shape_and_finite_uncertainty() -> None:
    head = SafetyResidualHead(future_frames=10)
    output = head(torch.randn(4, SAFETY_FEATURE_DIM))
    assert output["distance_residual"].shape == (4, 10)
    assert output["minimum_residual"].shape == (4,)
    calibration = SafetyCalibration(1.2, 1.3, 0.9, 0.6, 1.64, "validation")
    values = {name: value.detach().numpy() for name, value in output.items()}
    calibrated = apply_safety_calibration(values, calibration)
    assert np.isfinite(calibrated["sigma_distance"]).all()
    assert np.isfinite(calibrated["p_unsafe"]).all()


def test_false_safe_metric() -> None:
    assert false_safe_rate(
        np.asarray((True, False, True, False)),
        np.asarray((True, True, False, False)),
    ) == pytest.approx(0.5)


def test_worst_case_regret_metrics() -> None:
    result = worst_case_regret(np.asarray((0.0, 1.0, 2.0, 3.0)))
    assert result["median"] == pytest.approx(1.5)
    assert result["maximum"] == 3.0
    assert result["P95"] >= result["P90"]
