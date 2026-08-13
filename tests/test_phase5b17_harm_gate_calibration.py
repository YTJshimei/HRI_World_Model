import inspect
from types import SimpleNamespace

import numpy as np
import torch

from scripts import run_phase5b17_harm_gate_calibration as calibration
from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer


def _sample(episode, candidate, *, benefit=0.0, harm=False, profile=0, contexts=()):
    return SimpleNamespace(
        sample_id=f"validation:{episode}:{candidate}", episode_id=f"validation:{episode}", split="validation",
        targets=SimpleNamespace(benefit=benefit, harm=harm, feasible=True),
        split_metadata={"person_profile_id": profile}, temporal_tags=tuple(contexts), context_split="validation",
    )


def test_validation_split_has_no_episode_overlap_or_candidate_crossing():
    samples = []
    for episode in range(8):
        for candidate in range(4):
            samples.append(_sample(f"e{episode}", candidate, benefit=.1 if candidate == 1 else 0,
                                   harm=candidate == 3, profile=episode % 2,
                                   contexts=("C7_long",) if episode < 2 else ("C9_motion",)))
    split = calibration.stratified_episode_split(samples, seed=42)
    left, right = set(split["calibration_episode_ids"]), set(split["evaluation_episode_ids"])
    assert len(left) == len(right) == 4 and not left & right
    assert left | right == {sample.episode_id for sample in samples}
    assert split["candidate_cross_subset_count"] == split["episode_overlap_count"] == 0


def test_validation_split_is_fixed_by_seed_and_checksum():
    samples = [_sample(f"e{i}", 0, benefit=.1 if i % 2 else 0, harm=i % 3 == 0, profile=i % 2) for i in range(10)]
    first = calibration.stratified_episode_split(samples, 42)
    second = calibration.stratified_episode_split(samples, 42)
    assert first == second and first["split_checksum_sha256"] == second["split_checksum_sha256"]


def test_split_balances_beneficial_and_harmful_counts_when_exact_balance_exists():
    samples = []
    for episode in range(8):
        samples += [_sample(f"e{episode}", 0, benefit=.1 if episode < 4 else 0, harm=False, profile=episode % 2),
                    _sample(f"e{episode}", 1, benefit=0, harm=episode % 2 == 0, profile=episode % 2)]
    split = calibration.stratified_episode_split(samples, 42)
    left, right = split["distributions"]["calibration"], split["distributions"]["evaluation"]
    assert left["beneficial_candidate_count"] == right["beneficial_candidate_count"] == 2
    assert left["harmful_candidate_count"] == right["harmful_candidate_count"] == 2


def test_threshold_selection_accepts_calibration_only_not_evaluation():
    signature = inspect.signature(calibration.select_harm_threshold)
    assert list(signature.parameters) == ["calibration_samples", "calibration_prediction"]
    source = inspect.getsource(calibration.select_harm_threshold)
    assert "evaluation_samples" not in source and "evaluation_prediction" not in source


def test_threshold_priority_requires_zero_harmful_switch_then_recall_regret_conservatism(monkeypatch):
    monkeypatch.setattr(calibration, "threshold_candidates", lambda _: [0.1, 0.2, 0.3, 0.4])
    table = {
        .1: (0, .2, .08), .2: (1, .9, .01), .3: (0, .5, .06), .4: (0, .5, .06),
    }
    def fake(_, __, ___, threshold):
        harmful, recall, regret = table[threshold]
        metrics = {"Harmful_Switch_Count": harmful, "Beneficial_Switch_Count": 1,
                   "Beneficial_Switch_Recall": recall, "Beneficial_Switch_Precision": .5,
                   "Mean_Regret": regret, "P95_Regret": .1, "Safety_Violation": 0}
        funnel = {"harm_threshold_pass": 2}
        return [], [], metrics, {}, funnel
    monkeypatch.setattr(calibration, "evaluate_threshold", fake)
    selected, _, record = calibration.select_harm_threshold([], {"harm": np.asarray([.1])})
    assert selected == .3 and record["h1_threshold"] == .3 and record["locked"]


def test_evaluation_cannot_modify_locked_threshold():
    source = inspect.getsource(calibration.main)
    write_lock = source.index('"harm_threshold_selection.json"')
    evaluation_access = source.index("Formal Evaluation access starts")
    assert write_lock < evaluation_access
    assert '"evaluation_may_not_modify": True' in inspect.getsource(calibration.select_harm_threshold)


def test_model_architecture_and_parameters_are_not_modified():
    torch.manual_seed(42)
    model = RichTemporalSmallTransformer(); before = calibration.b16.model_checksum(model)
    model.eval()
    after = calibration.b16.model_checksum(model)
    assert before == after and model.architecture_audit()["trainable_parameter_count"] == 352376
    source = inspect.getsource(calibration)
    assert "torch.optim" not in source and ".backward(" not in source


def test_test_split_cannot_be_materialized_or_evaluated():
    source = inspect.getsource(calibration)
    forbidden = "materialize" + "_test"
    assert forbidden not in source
    assert '"test_candidates_read": 0' in source
    assert '"test_labels_read": 0' in source
    assert '"test_metrics_computed": False' in source


def test_lambda_rank_and_benefit_threshold_are_frozen():
    assert calibration.LAMBDA_RANK == .25
    assert calibration.BENEFIT_THRESHOLD == -.02
    source = inspect.getsource(calibration.parse_args)
    assert "lambda" not in source.lower() and "benefit-threshold" not in source


def test_h0_harm_threshold_is_frozen_at_point_two():
    assert calibration.H0_HARM_THRESHOLD == .2


def test_only_harm_threshold_differs_between_h0_and_h1_evaluation():
    source = inspect.getsource(calibration.main)
    assert 'conditions = {MODELS[0]: H0_HARM_THRESHOLD, MODELS[1]: h1_threshold}' in source
    evaluation_source = inspect.getsource(calibration.evaluate_threshold)
    assert "BENEFIT_THRESHOLD, harm_threshold" in evaluation_source


def test_safety_generic_personalized_cost_and_arbitration_are_checksum_audited():
    source = inspect.getsource(calibration.main)
    for field in ("safety_mask_unchanged", "generic_score_unchanged", "personalized_cost_unchanged", "arbitration_unchanged"):
        assert field in source


def test_probability_audit_reports_required_calibration_metrics():
    probability = np.asarray([.05, .2, .8, .9])
    target = np.asarray([False, False, True, True])
    metrics = calibration.harm_calibration_metrics(probability, target)
    assert set(("AUROC", "AUPRC", "Brier_Score", "ECE_10_bins", "Binary_NLL")) <= set(metrics)
    assert metrics["AUROC"] == 1.0 and all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))


def test_harm_probability_summary_has_all_required_quantiles():
    summary = calibration.probability_statistics(np.arange(10) / 10)
    assert set(("mean", "median", "P10", "P25", "P50", "P75", "P90", "P95")) <= set(summary)


def test_probability_audit_distinguishes_overall_from_beneficial_subgroup_inflation():
    samples = [_sample("e", 0, benefit=.2), _sample("e", 1, harm=True), _sample("e", 2)]
    _, metrics = calibration.probability_audit(samples, {"harm": np.asarray([.7, .95, .05])})
    diagnostic = metrics["systematic_high_probability_diagnostic"]
    assert diagnostic["systematically_high_for_beneficial_candidates"]
    assert "subgroup" in diagnostic["diagnosis"]


def test_any_new_harmful_switch_forces_gate_b_failure_contract():
    source = inspect.getsource(calibration.main)
    assert '"harmful_switch_count_zero": decision_metrics[h1]["Harmful_Switch_Count"] == 0' in source
    assert '"any_new_harmful_switch_forces_fail": True' in source
