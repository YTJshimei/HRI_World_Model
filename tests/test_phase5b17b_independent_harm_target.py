import inspect
from types import SimpleNamespace

from scripts import run_phase5b17b_independent_harm_target as audit


def _sample(episode, candidate, benefit, *, unsafe=False, feasible=True, profile=0, tags=()):
    return SimpleNamespace(
        sample_id=f"validation:{episode}:{candidate}", episode_id=f"validation:{episode}", split="validation",
        targets=SimpleNamespace(benefit=benefit, harm=benefit < -1e-6, gt_unsafe=unsafe, feasible=feasible),
        split_metadata={"candidate_action_id_audit": candidate, "motion_type_evaluation_only": "walk",
                        "person_profile_id": profile}, temporal_tags=tags,
    )


def test_harm_a_is_gt_unsafe_not_negative_benefit():
    beneficial_unsafe = _sample("e", 0, .2, unsafe=True)
    harmful_safe = _sample("e", 1, -.2, unsafe=False)
    assert audit.label_for(beneficial_unsafe, "harm_A") is True
    assert audit.label_for(harmful_safe, "harm_A") is False


def test_unreliable_response_and_disturbance_candidates_are_not_fabricated():
    sample = _sample("e", 0, .1)
    assert audit.label_for(sample, "harm_B") is None
    assert audit.label_for(sample, "harm_C") is None
    definitions = audit.candidate_definitions()["definitions"]
    assert definitions["harm_A"]["constructible"]
    assert not definitions["harm_B"]["constructible"]
    assert not definitions["harm_C"]["constructible"]


def test_new_harm_contains_every_gt_unsafe_candidate():
    samples = [_sample("e0", 0, .3, unsafe=True), _sample("e0", 1, -.4, unsafe=True), _sample("e1", 0, .1)]
    assert all(not sample.targets.gt_unsafe or audit.label_for(sample, "harm_A") for sample in samples)


def test_safe_beneficial_is_exact_conjunction():
    samples = [_sample("e0", 0, .2), _sample("e0", 1, .2, unsafe=True),
               _sample("e1", 0, .2, feasible=False), _sample("e2", 0, -.2)]
    row = audit.safe_beneficial_row("validation", "harm_A", samples)
    assert row["candidate_count"] == 1 and row["episode_count"] == 1


def test_four_quadrants_sum_to_total_and_preserve_episode_counts():
    samples = [_sample("e0", 0, .2), _sample("e0", 1, .2, unsafe=True),
               _sample("e1", 0, -.2, unsafe=True), _sample("e2", 0, -.2)]
    rows = audit.quadrant_rows("validation", "harm_A", samples)
    assert sum(row["candidate_count"] for row in rows) == len(samples)
    assert [row["candidate_count"] for row in rows] == [1, 1, 1, 1]
    assert all(row["episode_count"] == 1 for row in rows)


def test_invalid_candidate_quadrants_are_explicit_not_zero_filled():
    row = audit.quadrant_rows("validation", "harm_B", [_sample("e", 0, .2)])[0]
    assert not row["definition_valid"] and row["candidate_count"] is None


def test_profile_id_is_audit_only_and_gt_not_runtime_input():
    source = inspect.getsource(audit.main)
    assert '"profile_id_runtime_input": False' in source
    assert '"profile_id_audit_only": True' in source
    assert '"GT_fields_runtime_input": False' in source
    assert '"GT_fields_audit_only": True' in source


def test_no_test_materialization_training_optimizer_or_backward():
    source = inspect.getsource(audit)
    forbidden = "materialize" + "_test"
    assert forbidden not in source
    assert "torch.optim" not in source and ".backward(" not in source
    assert '"test_candidates_read": 0' in source
    assert '"optimizer_step_count": 0' in source and '"backward_call_count": 0' in source


def test_model_checkpoint_and_frozen_files_are_checksum_only():
    source = inspect.getsource(audit.main)
    assert "torch.load" not in source
    assert '"model_checkpoint_loaded": False' in source
    assert "file_checksums_before" in source and "file_checksums_after" in source
    assert "old_target_overwritten" in source and "manifest_unchanged" in source


def test_signal_registry_does_not_claim_nonexistent_adverse_events():
    registry = audit.harm_signal_registry()
    assert registry["independent_binary_signals"] == ["GT unsafe"]
    assert not registry["adverse_response_event_available"]
    assert not registry["excess_disturbance_event_available"]


def test_harm_a_is_not_exactly_equivalent_to_negative_benefit():
    samples = [_sample("e0", 0, .2, unsafe=True), _sample("e1", 0, -.2, unsafe=False)]
    row = audit.correlation_rows("validation", "harm_A", samples)
    assert not row["exact_negative_benefit_equivalence"]

