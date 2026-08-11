import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 4B.5 tests require PyTorch")

from src.data.personal_interaction_memory import generate_personal_interaction_corpus
from src.data.personalization_diagnostics import (
    RESPONSE_COVERED_PROFILES,
    TRUE_EFFECT_DESCRIPTOR_DIM,
    generate_covered_personal_corpus,
    true_effect_descriptor,
)
from src.data.synthetic_interaction import PROFILE_BY_ID
from src.evaluation.personalization_diagnostics import fit_uncertainty_scale
from src.models.personalization_diagnostics import (
    MetaPersonalizedWorldModel,
    OracleEffectWorldModel,
    ResponseOracleWorldModel,
)
from src.training.train_meta_personalization import (
    MetaEpisodeDataset,
    MetaLossWeights,
    meta_query_loss,
)


@pytest.fixture(scope="module")
def corpus():
    return generate_personal_interaction_corpus(
        (0,), 1, 12, 10, 550, "phase4b5_test",
        noise_std=0.0, occlusion_rate=0.0,
    )


def test_oracle_effect_descriptor_has_no_future_coordinate_input(corpus) -> None:
    signature = inspect.signature(true_effect_descriptor)
    assert "natural_future" not in signature.parameters
    assert "future_global" not in signature.parameters
    split = corpus.split
    descriptor = true_effect_descriptor(
        split.human_history[0], split.robot_history[0], 4, PROFILE_BY_ID[0]
    )
    assert descriptor.shape == (TRUE_EFFECT_DESCRIPTOR_DIM,)
    assert np.isfinite(descriptor).all()


def test_meta_episode_same_person_past_only_and_k0(corpus) -> None:
    dataset = MetaEpisodeDataset(corpus, (0, 3), "earliest")
    empty = dataset[0]
    supported = dataset[1]
    assert empty["support_mask"].sum().item() == 0
    assert supported["support_mask"].sum().item() == 3
    query = corpus.records[int(supported["source_index"])]
    support = dataset.memory.select_support(query, 3, "earliest")
    assert all(item.person_instance_id == query.person_instance_id for item in support)
    assert all(item.order_index < query.order_index for item in support)


def test_query_loss_backpropagates_to_personal_encoder(corpus) -> None:
    batch = next(iter(torch.utils.data.DataLoader(MetaEpisodeDataset(corpus, 3), batch_size=1)))
    model = MetaPersonalizedWorldModel()
    personalized, generic = model.paired_forward(
        batch["history"], batch["robot"], batch["actions"],
        batch["confidence"], batch["visibility"],
        batch["support_features"], batch["support_mask"],
    )
    loss, _ = meta_query_loss(
        personalized, generic, batch, MetaLossWeights(amplitude=1.0, personal_gain=1.0)
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.personal_response_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_latent_interpolation_forward_shape(corpus) -> None:
    batch = next(iter(torch.utils.data.DataLoader(MetaEpisodeDataset(corpus, 3), batch_size=1)))
    model = MetaPersonalizedWorldModel().eval()
    with torch.inference_mode():
        encoded = model.encode_context(
            batch["history"], batch["robot"], batch["confidence"], batch["visibility"],
            support_features=batch["support_features"], support_mask=batch["support_mask"],
        )
        output = model.interpolate_forward(encoded[:-1], encoded[-1], batch["actions"])
    assert output.future_by_action.shape == (1, 5, 10, 17, 3)


def test_p2_meta_has_no_profile_or_oracle_encoder() -> None:
    names = tuple(name for name, _ in MetaPersonalizedWorldModel().named_modules())
    assert not any("profile_to" in name for name in names)
    assert not any("oracle" in name for name in names)
    oracle_names = tuple(name for name, _ in ResponseOracleWorldModel().named_modules())
    assert any("profile_to_response_statistics" in name for name in oracle_names)


def test_uncertainty_calibration_rejects_nonvalidation_fit() -> None:
    error = np.ones((2, 3), dtype=np.float32)
    sigma = np.ones_like(error)
    assert fit_uncertainty_scale(error, sigma, "validation") == pytest.approx(1.0)
    with pytest.raises(ValueError, match="validation"):
        fit_uncertainty_scale(error, sigma, "test")


def test_response_covered_training_excludes_test_profiles() -> None:
    assert {profile.profile_id for profile in RESPONSE_COVERED_PROFILES}.isdisjoint({5, 6})
    corpus = generate_covered_personal_corpus(
        persons_per_profile=1, interactions_per_person=12, query_start=10, seed=99
    )
    assert set(corpus.split.person_profile_id.tolist()).isdisjoint({5, 6})


def test_oracle_models_forward_shapes(corpus) -> None:
    batch = next(iter(torch.utils.data.DataLoader(
        MetaEpisodeDataset(corpus, 0, oracle_access=True), batch_size=1
    )))
    effect_model = OracleEffectWorldModel().eval()
    response_model = ResponseOracleWorldModel().eval()
    with torch.inference_mode():
        effect_output = effect_model(
            batch["history"], batch["robot"], batch["actions"],
            batch["confidence"], batch["visibility"], batch["effect_descriptors"],
        )
        response_output = response_model(
            batch["history"], batch["robot"], batch["actions"],
            batch["confidence"], batch["visibility"], batch["profile_parameters"],
        )
    assert effect_output.future_by_action.shape == (1, 5, 10, 17, 3)
    assert response_output.future_by_action.shape == (1, 5, 10, 17, 3)
    assert response_output.predicted_response_statistics.shape == (1, 7)
