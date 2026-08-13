import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from scripts import run_phase5a_lora_formal as formal_lora


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__(); self.projection = nn.Linear(2, 2); self.benefit = nn.Linear(2, 1)
        self.harm = nn.Linear(2, 1); self.uncertainty = nn.Linear(2, 1)
        self.backbone = nn.Module(); self.backbone.lora_A = nn.Parameter(torch.ones(2, 2))
    def trainable_state_dict(self):
        prefixes = ("projection.", "benefit.", "harm.", "uncertainty.")
        return {name: value.detach().clone() for name, value in self.state_dict().items() if name.startswith(prefixes)}
    def load_trainable_state_dict(self, state):
        own = self.state_dict()
        with torch.no_grad():
            for name, value in state.items(): own[name].copy_(value)


def test_lora_config_is_frozen_and_explicit():
    targets = [f"model.language_model.layers.{i}.self_attn.{leaf}" for i in range(36) for leaf in ("q_proj", "v_proj")]
    config = formal_lora.lora_config_contract(targets)
    assert (config["r"], config["lora_alpha"], config["lora_dropout"], config["bias"]) == (8, 16, 0.05, "none")
    assert config["target_count"] == 72 and all("self_attn" in path for path in targets)


def test_lora_aware_checkpoint_roundtrip():
    model = TinyAdapter(); state = formal_lora.task_state_dict(model); expected = formal_lora.state_checksum(state)
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    formal_lora.load_task_state_dict(model, state)
    assert formal_lora.state_checksum(formal_lora.task_state_dict(model)) == expected


def test_prepared_contract_detects_required_mismatch():
    state = {"parameter_dtype_counts": {"torch.float32": 1}, "module_dtype_counts": {}, "layer_norm_dtype_counts": {},
             "embedding_dtype": "torch.float32", "lm_head_dtype": "torch.float32", "linear4bit_count": 1,
             "is_loaded_in_4bit": True, "requires_grad_parameter_count": 0}
    current = {"after": dict(state), "preparation_source_sha256": "a"}
    reference = {"after": dict(state), "preparation_source_sha256": "a"}
    assert formal_lora.compare_prepared_contract(current, reference)["matched"]
    reference["after"]["linear4bit_count"] = 2
    assert not formal_lora.compare_prepared_contract(current, reference)["matched"]


def test_training_source_retains_full_gaussian_nll_without_detach_or_clip():
    source = inspect.getsource(formal_lora.train_lora)
    likelihood = source[source.index("benefit ="):source.index("uncertainty =")]
    assert "torch.exp(-output.benefit_log_variance)" in likelihood
    assert ".detach()" not in likelihood and "clip_grad" not in source


def test_test_is_materialized_once_after_checkpoint_and_threshold_lock():
    source = inspect.getsource(formal_lora.main)
    assert source.count("base.materialize_test") == 1
    assert source.index("checkpoint_selection.json") < source.index("guard.lock") < source.index("base.materialize_test")
    assert "test_can_change_checkpoint_or_threshold\": False" in source


def test_gate_a_requires_complex_split_decision_gain():
    prepared = {"Beneficial_Switch_Recall": .02, "Beneficial_Switch_Precision": .02, "Mean_Regret": .04,
                "Benefit_Spearman": .5, "Harm_AUROC": .8}
    l1 = {"Beneficial_Switch_Recall": .06, "Beneficial_Switch_Precision": .06, "Mean_Regret": .04, "Safety_Violation": .01}
    metrics = {"Beneficial_Switch_Recall": .08, "Beneficial_Switch_Precision": .08, "Harmful_Switch_Rate": 0.,
               "Mean_Regret": .03, "Benefit_Spearman": .51, "Harm_AUROC": .81, "Safety_Violation": 0.}
    no_gain = {name: {"decision_level_improved": False} for name in ("C4", "C5", "C6")}
    assert not formal_lora.gates(metrics, {"L2-P-PREPARED-NO-LORA": prepared, "L1": l1}, no_gain, no_gain)["Gate_A"]["passed"]
