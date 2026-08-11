import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Phase 4B.6 tests require PyTorch")

from src.data.functional_response_state import (
    RESPONSE_STATE_DIM,
    RESPONSE_STATE_NAMES,
    functional_state_from_profile,
    response_state_mask_for_action,
)
from src.data.personal_interaction_memory import generate_personal_interaction_corpus
from src.data.response_statistics import (
    RESPONSE_STATISTIC_DIM,
    extract_response_statistics,
)
from src.data.synthetic_interaction import PROFILE_BY_ID
from src.models.functional_response_decoder import (
    FunctionalResponseDecoder,
    FunctionalResponseWorldModel,
)
from src.models.functional_response_estimator import FunctionalResponseEstimator
from src.training.train_functional_response import FunctionalEpisodeDataset


@pytest.fixture(scope="module")
def corpus():
    return generate_personal_interaction_corpus(
        (1,), 1, 12, 10, 660, "phase4b6_test",
        noise_std=0.0, occlusion_rate=0.0,
    )


def test_functional_response_schema_has_six_nonidentity_dimensions() -> None:
    assert RESPONSE_STATE_DIM == 6
    assert len(set(RESPONSE_STATE_NAMES)) == 6
    assert not any("person" in name or "identity" in name for name in RESPONSE_STATE_NAMES)
    theta = functional_state_from_profile(PROFILE_BY_ID[1])
    assert theta.shape == (6,)


def test_observable_statistics_extractor_cannot_receive_hidden_profile(corpus) -> None:
    signature = inspect.signature(extract_response_statistics)
    assert "profile" not in signature.parameters
    statistics = extract_response_statistics(corpus.records[0])
    assert statistics.values.shape == (RESPONSE_STATISTIC_DIM,)
    assert np.isfinite(statistics.values).all()


def test_action_specific_response_masks() -> None:
    keep = response_state_mask_for_action(0)
    speed = response_state_mask_for_action(1)
    distance = response_state_mask_for_action(3)
    assert not keep.any()
    assert speed[0] and not speed[1] and not speed[2]
    assert distance[1] and distance[2] and not distance[0]


def test_k0_and_unobserved_dimension_use_generic_uncertainty() -> None:
    estimator = FunctionalResponseEstimator().eval()
    statistics = torch.randn(2, 10, RESPONSE_STATISTIC_DIM)
    support = torch.zeros(2, 10, dtype=torch.bool)
    state_mask = torch.zeros(2, 10, RESPONSE_STATE_DIM, dtype=torch.bool)
    with torch.inference_mode():
        output = estimator(statistics, support, state_mask)
    assert output.theta_mean.shape == (2, 6)
    assert output.theta_log_std.shape == (2, 6)
    assert not output.observed_dimension_mask.any()
    torch.testing.assert_close(output.theta_log_std[0], output.theta_log_std[1])


def test_functional_episode_support_is_past_and_same_person(corpus) -> None:
    dataset = FunctionalEpisodeDataset(corpus, 3, "speed_only")
    item = dataset[0]
    query = corpus.records[int(item["source_index"])]
    support = dataset._support(query, 3, int(item["source_index"]))
    assert all(record.person_instance_id == query.person_instance_id for record in support)
    assert all(record.order_index < query.order_index for record in support)
    assert all(record.executed_action in (1, 2) for record in support)


def test_f2_interface_has_no_person_id_or_raw_profile() -> None:
    signature = inspect.signature(FunctionalResponseWorldModel.forward)
    assert "person_id" not in signature.parameters
    assert "profile" not in signature.parameters
    names = tuple(name for name, _ in FunctionalResponseWorldModel().named_modules())
    assert not any("person_id" in name or "profile" in name for name in names)


def test_f3_decoder_accepts_only_functional_oracle_state() -> None:
    signature = inspect.signature(FunctionalResponseDecoder.forward)
    assert "theta_response" in signature.parameters
    assert "person_id" not in signature.parameters
    assert "profile" not in signature.parameters


def decoder_inputs(batch=1):
    history = torch.randn(batch, 20, 17, 3)
    natural = torch.randn(batch, 10, 17, 3)
    robot = torch.randn(batch, 20, 7)
    actions = torch.tensor([[0, 1, 2, 3, 4]]).expand(batch, -1)
    theta = torch.tensor([[0.6, 0.7, 0.5, 0.3, 0.4, 2.0]]).expand(batch, -1).clone()
    return history, natural, robot, actions, theta


def effect_magnitude(future, natural, action_index):
    return float(torch.linalg.vector_norm(
        future[:, action_index] - natural, dim=-1
    ).mean())


def test_functional_interventions_are_directionally_consistent() -> None:
    decoder = FunctionalResponseDecoder().eval()
    history, natural, robot, actions, theta = decoder_inputs()
    speed_low, speed_high = theta.clone(), theta.clone()
    speed_low[:, 0] *= 0.5; speed_high[:, 0] *= 1.5
    distance_low, distance_high = theta.clone(), theta.clone()
    distance_low[:, 1] *= 0.5; distance_high[:, 1] *= 1.5
    lateral_low, lateral_high = theta.clone(), theta.clone()
    lateral_low[:, 2] *= 0.5; lateral_high[:, 2] *= 1.5
    delay_low, delay_high = theta.clone(), theta.clone()
    delay_low[:, 3] *= 0.5; delay_high[:, 3] *= 1.5
    with torch.inference_mode():
        sl = decoder.decode_response(natural, history, robot, actions, speed_low)
        sh = decoder.decode_response(natural, history, robot, actions, speed_high)
        dl = decoder.decode_response(natural, history, robot, actions, distance_low)
        dh = decoder.decode_response(natural, history, robot, actions, distance_high)
        ll = decoder.decode_response(natural, history, robot, actions, lateral_low)
        lh = decoder.decode_response(natural, history, robot, actions, lateral_high)
        tlo = decoder.decode_response(natural, history, robot, actions, delay_low)
        thi = decoder.decode_response(natural, history, robot, actions, delay_high)
    assert effect_magnitude(sh, natural, 2) > effect_magnitude(sl, natural, 2)
    assert effect_magnitude(dh, natural, 4) > effect_magnitude(dl, natural, 4)
    assert effect_magnitude(lh, natural, 4) > effect_magnitude(ll, natural, 4)
    assert effect_magnitude(thi, natural, 2) < effect_magnitude(tlo, natural, 2)


def test_action_routing_blocks_unrelated_lateral_gain_for_speed_action() -> None:
    decoder = FunctionalResponseDecoder().eval()
    history, natural, robot, actions, theta = decoder_inputs()
    changed = theta.clone(); changed[:, 2] *= 5.0
    with torch.inference_mode():
        base = decoder.decode_response(natural, history, robot, actions, theta)
        altered = decoder.decode_response(natural, history, robot, actions, changed)
    torch.testing.assert_close(base[:, 1:3], altered[:, 1:3])
    assert not torch.allclose(base[:, 3:5], altered[:, 3:5])


def test_functional_swap_and_candidate_permutation_follow_theta() -> None:
    decoder = FunctionalResponseDecoder().eval()
    history, natural, robot, actions, _ = decoder_inputs()
    theta_a = torch.from_numpy(functional_state_from_profile(PROFILE_BY_ID[2]))[None]
    theta_b = torch.from_numpy(functional_state_from_profile(PROFILE_BY_ID[1]))[None]
    with torch.inference_mode():
        future_a = decoder.decode_response(natural, history, robot, actions, theta_a)
        future_b = decoder.decode_response(natural, history, robot, actions, theta_b)
    assert not torch.allclose(future_a, future_b)
    permutation = torch.tensor([4, 2, 0, 1, 3])
    with torch.inference_mode():
        permuted = decoder.decode_response(
            natural, history, robot, actions[:, permutation], theta_a
        )
    torch.testing.assert_close(future_a, permuted[:, torch.argsort(permutation)])
