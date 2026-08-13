import inspect
from types import SimpleNamespace
import pytest
torch=pytest.importorskip("torch")
from torch import nn
from scripts import run_phase5a_lora_smoke as d0
from scripts import run_phase5a_prepared_nolora_control as control


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(8,4);self.norm=nn.LayerNorm(4);self.linear=nn.Linear(4,4);self.config=SimpleNamespace(use_cache=True)
    def get_input_embeddings(self):return self.embedding
    def get_output_embeddings(self):return None


def test_prepared_snapshot_records_required_dtype_state():
    snapshot=d0.prepared_base_snapshot(FakeBackbone())
    assert snapshot["embedding_dtype"]=="torch.float32"
    assert snapshot["layer_norm_dtype_counts"]=={"torch.float32":1}
    assert snapshot["linear4bit_count"]==0 and snapshot["lora_parameter_count"]==0


def test_control_source_forbids_lora_injection():
    source=inspect.getsource(control)
    assert "get_peft_model(" not in source and "LoraConfig(" not in source
    assert "d0.prepare_kbit_backbone" in source
    assert '"lora_module_count":0' in source and '"lora_parameter_count":0' in source


def test_control_optimizer_only_uses_projection_heads():
    source=inspect.getsource(control.smoke_50)
    assert "parameters = base.trainable_parameters(model)" in source
    assert "optimizer_only_projection_heads" in source
    assert "qwen_optimizer_parameter_count" in source


def test_control_retains_gaussian_and_has_no_clip_or_detach():
    source=inspect.getsource(control.smoke_50)
    likelihood=source[source.index("benefit="):source.index("uncertainty=")]
    assert "torch.exp(-output.benefit_log_variance)" in likelihood and ".detach()" not in likelihood
    assert "clip_grad" not in source


def test_prepared_contract_is_shared_with_d0_and_future_d_r1():
    source=inspect.getsource(control)
    assert "d0.prepared_base_contract" in source
    assert "d0.prepare_kbit_backbone" in source
    contract=d0.prepared_base_contract(d0.prepared_base_snapshot(FakeBackbone()),d0.prepared_base_snapshot(FakeBackbone()))
    assert contract["D0_D_C0_future_D_R1_shared_contract"]


def test_test_materialization_occurs_only_after_selection_lock():
    source=inspect.getsource(control.main)
    assert source.index('"checkpoint_selection.json"') < source.index("guard.lock") < source.index("base.materialize_test")
    assert source.count("base.materialize_test") == 1
