import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 4B model tests require PyTorch")

from src.data.personal_interaction_memory import (
    OBSERVABLE_INTERACTION_FEATURE_DIM,
    PersonalInteractionMemory,
    generate_personal_interaction_corpus,
    support_to_padded_features,
    validate_support_query,
)
from src.evaluation.personal_response_metrics import (
    human_response_ranking_per_sample,
    oracle_gap,
    uncertainty_metrics,
)
from src.models.personal_response_encoder import PersonalResponseEncoder
from src.models.personalized_response_world_model import PersonalizedRootPoseWorldModel


@pytest.fixture(scope="module")
def corpus():
    return generate_personal_interaction_corpus(
        profile_ids=(5,), persons_per_profile=1, interactions_per_person=12,
        query_start=10, seed=420, split_label="phase4b_test",
        noise_std=0.0, occlusion_rate=0.0,
    )


def test_memory_record_and_support_shapes(corpus) -> None:
    query = corpus.query_records()[0]
    memory = PersonalInteractionMemory(corpus.records)
    assert memory.select_support(query, 0) == ()
    for k in (1, 3, 5, 10):
        support = memory.select_support(query, k)
        features, mask = support_to_padded_features(support)
        assert features.shape == (10, OBSERVABLE_INTERACTION_FEATURE_DIM)
        assert mask.sum() == k


def test_support_is_same_person_strictly_past_and_excludes_query(corpus) -> None:
    query = corpus.query_records()[0]
    support = PersonalInteractionMemory(corpus.records).select_support(query, 10)
    assert all(item.person_instance_id == query.person_instance_id for item in support)
    assert all(item.order_index < query.order_index for item in support)
    assert all(item.timestamp < query.timestamp for item in support)
    assert query.interaction_id not in {item.interaction_id for item in support}
    with pytest.raises(ValueError, match="own support"):
        validate_support_query((query,), query)


def test_support_mixed_person_is_rejected(corpus) -> None:
    other = generate_personal_interaction_corpus(
        (6,), 1, 12, 10, 421, "other_person", noise_std=0.0
    )
    query = corpus.query_records()[0]
    with pytest.raises(ValueError, match="mix persons"):
        validate_support_query((other.records[0],), query)


def test_personal_encoder_k0_null_and_determinism() -> None:
    encoder = PersonalResponseEncoder().eval()
    features = torch.randn(2, 10, OBSERVABLE_INTERACTION_FEATURE_DIM)
    empty = torch.zeros(2, 10, dtype=torch.bool)
    with torch.inference_mode():
        first = encoder(features, empty)
        second = encoder(features * 100.0, empty)
        third = encoder(features, empty)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, third)
    assert first.shape == (2, 32)


def test_support_changes_latent_but_identical_support_does_not() -> None:
    encoder = PersonalResponseEncoder().eval()
    features = torch.randn(1, 10, OBSERVABLE_INTERACTION_FEATURE_DIM)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, :3] = True
    with torch.inference_mode():
        same_a = encoder(features, mask)
        same_b = encoder(features.clone(), mask.clone())
        changed = encoder(features + mask[..., None] * 0.5, mask)
    torch.testing.assert_close(same_a, same_b)
    assert not torch.allclose(same_a, changed)


def model_inputs(batch: int = 2):
    return {
        "history": torch.randn(batch, 20, 17, 3),
        "robot_history": torch.randn(batch, 20, 7),
        "action_ids": torch.tensor([[0, 1, 2, 3, 4]]).expand(batch, -1),
        "confidence": torch.ones(batch, 20, 17),
        "visibility": torch.ones(batch, 20, 17, dtype=torch.bool),
        "support_features": torch.randn(batch, 10, OBSERVABLE_INTERACTION_FEATURE_DIM),
        "support_mask": torch.ones(batch, 10, dtype=torch.bool),
    }


def test_p2_has_no_person_id_or_oracle_parameter_path() -> None:
    model = PersonalizedRootPoseWorldModel("P2")
    names = tuple(name for name, _ in model.named_modules())
    assert not any("person_id" in name for name in names)
    assert not any("oracle" in name for name in names)
    assert hasattr(model, "personal_response_encoder")


def test_p3_requires_oracle_and_p2_does_not() -> None:
    values = model_inputs(1)
    p3 = PersonalizedRootPoseWorldModel("P3").eval()
    with pytest.raises(ValueError, match="oracle"):
        p3(values["history"], values["robot_history"], values["action_ids"], values["confidence"], values["visibility"])
    p2 = PersonalizedRootPoseWorldModel("P2").eval()
    with torch.inference_mode():
        output = p2(**values)
    assert output.future_by_action.shape == (1, 5, 10, 17, 3)
    assert output.action_effect_root_log_std_by_action.shape == (1, 5, 10, 3)


def test_p2_action_order_equivariance_and_encode_once() -> None:
    model = PersonalizedRootPoseWorldModel("P2").eval()
    values = model_inputs()
    calls = []
    handle = model.root_encoder.register_forward_hook(lambda *unused: calls.append(1))
    with torch.inference_mode():
        original = model(**values).future_by_action
    handle.remove()
    assert len(calls) == 1
    permutation = torch.tensor([4, 1, 3, 0, 2])
    shuffled = dict(values)
    shuffled["action_ids"] = values["action_ids"][:, permutation]
    with torch.inference_mode():
        permuted = model(**shuffled).future_by_action
    torch.testing.assert_close(original, permuted[:, torch.argsort(permutation)], atol=1e-6, rtol=1e-6)


def test_disabled_action_conditioning_produces_identical_branches() -> None:
    model = PersonalizedRootPoseWorldModel("P2").eval()
    values = model_inputs()
    with torch.inference_mode():
        output = model(**values, action_conditioning=False).future_by_action
    torch.testing.assert_close(output, output[:, :1].expand_as(output))


def test_human_response_ranking_has_no_robot_reward_input() -> None:
    expected = np.zeros((1, 5, 2, 17, 3), dtype=np.float32)
    predicted = np.zeros_like(expected)
    for action in range(1, 5):
        expected[:, action] = action * 0.01
        predicted[:, action] = action * 0.02
    actions = np.asarray([[0, 1, 2, 3, 4]])
    score = human_response_ranking_per_sample(predicted, expected, actions)
    assert score[0] == pytest.approx(1.0)


def test_uncertainty_metrics_and_oracle_gap_are_finite_or_explicit() -> None:
    prediction = np.zeros((2, 5, 10, 3), dtype=np.float32)
    target = np.full_like(prediction, 0.1)
    log_std = np.zeros_like(prediction)
    metrics, curve = uncertainty_metrics(prediction, target, log_std)
    assert np.isfinite(metrics["Root_NLL"])
    assert len(curve) == 5
    assert oracle_gap(1.0, 0.7, 0.5) == pytest.approx(0.6)
    assert oracle_gap(1.0, 0.7, 1.1) is None
