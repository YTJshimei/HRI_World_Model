import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts import run_phase5b1_static_vs_temporal as phase5b1
from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
from src.multimodal.temporal_collate import collate_temporal
from src.multimodal.temporal_schema import runtime_payload
from tests.test_phase5b_temporal_schema import make_sample


def batch(samples):
    return collate_temporal(samples, as_torch=True)


def test_frozen_manifest_hash_matches_phase5b_v1():
    path = Path("results_dev/phase5b05_c7_coverage/phase5b_manifest_v1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == phase5b1.EXPECTED_MANIFEST_SHA


def test_b1_fixed_architecture_and_forward_shapes():
    model = RichTemporalSmallTransformer().eval()
    with torch.inference_mode(): prediction = model(batch([make_sample(), make_sample()]))
    audit = model.architecture_audit()
    assert prediction.context_embedding.shape == (2, 128)
    assert prediction.benefit_mean.shape == prediction.benefit_log_variance.shape == prediction.harm_logit.shape == (2,)
    assert audit["trainable_parameter_count"] < 1_000_000
    assert audit["whole_sample_flattened"] is False


def test_non_frozen_b1_architecture_is_rejected():
    with pytest.raises(ValueError, match="architecture is frozen"):
        RichTemporalSmallTransformer(d_model=256)


def test_missing_skeleton_is_masked_and_not_treated_as_zero_observation():
    sample = make_sample(); sample.streams["skeleton_history"][5, 3] = 999
    sample.masks["skeleton_history"][5, 3] = False
    model = RichTemporalSmallTransformer(dropout=.1).eval()
    first = batch([sample])
    changed = make_sample(); changed.streams["skeleton_history"][5, 3] = -999
    changed.masks["skeleton_history"][5, 3] = False
    with torch.inference_mode():
        a = model(first).context_embedding
        b = model(batch([changed])).context_embedding
    assert torch.allclose(a, b, atol=1e-6)


def test_attention_padding_mask_blocks_padded_token_values():
    first = make_sample(); second = make_sample()
    first.masks["history_padding_mask"][:4] = False
    second.masks["history_padding_mask"][:4] = False
    for name in phase5b1.STREAM_ORDER[:7]:
        first.streams[name][:4] = 100
        second.streams[name][:4] = -100
        first.masks[name][:4] = False
        second.masks[name][:4] = False
    model = RichTemporalSmallTransformer().eval()
    with torch.inference_mode():
        a = model(batch([first])).context_embedding
        b = model(batch([second])).context_embedding
    assert torch.allclose(a, b, atol=1e-6)


def test_candidate_future_is_robot_only_and_temporal_input_has_no_gt_or_identity():
    sample = make_sample(); payload = runtime_payload(sample)
    assert payload["streams"]["candidate_robot_future"].shape == (10, 5)
    keys = set(payload["streams"])
    assert not keys & {"future_global", "gt_human_future", "person_profile_id", "theta_true", "gt_theta"}


def test_same_samples_imply_identical_b0_b1_ids_splits_targets_and_tags():
    samples = [make_sample("train"), make_sample("validation")]
    b0 = phase5b1.sample_contract(samples); b1 = phase5b1.sample_contract(samples)
    assert b0 == b1
    assert all(tuple(row["tags"]) == tuple(b1[key]["tags"]) for key, row in b0.items())


def test_normalizers_are_train_only_and_feasible_only():
    samples = [make_sample("train"), make_sample("train")]
    metadata = dict(samples[0].split_metadata); metadata["static_context_108"] = np.zeros(108, np.float32)
    object.__setattr__(samples[0], "split_metadata", metadata)
    metadata = dict(samples[1].split_metadata); metadata["static_context_108"] = np.ones(108, np.float32)
    object.__setattr__(samples[1], "split_metadata", metadata)
    object.__setattr__(samples[1], "targets", type(samples[1].targets)(0., False, 0., False, False, 0., False))
    fitted = phase5b1.fit_normalizers(samples)
    assert fitted["fit_split"] == "train" and fitted["fit_sample_ids"] == [samples[0].sample_id]
    with pytest.raises(ValueError, match="train candidates"):
        phase5b1.fit_normalizers([make_sample("validation")])


def test_formal_source_locks_before_one_test_materialization():
    source = inspect.getsource(phase5b1.main)
    assert source.index("checkpoint_selection.json") < source.index("materialize_test_once")
    assert '"test_materialization_count": 0' in source
    assert 'guard["test_materialization_count"] != 1' in source


def test_full_heteroscedastic_nll_has_no_variance_detach():
    source = inspect.getsource(phase5b1.train)
    assert "torch.exp(-output.benefit_log_variance)" in source
    assert "benefit_log_variance.detach" not in source


def test_candidate_permutation_preserves_contract_and_output_order():
    samples = [make_sample() for _ in range(3)]
    for index, sample in enumerate(samples):
        object.__setattr__(sample, "sample_id", f"id:{index}")
        sample.streams["candidate_action"][index] = 1
    order = (2, 0, 1)
    assert [sample.sample_id for sample in [samples[i] for i in order]] == ["id:2", "id:0", "id:1"]
    assert batch([samples[i] for i in order])["streams"]["candidate_action"].shape == (3, 11)
