import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from scripts import run_phase5a_lora_smoke as smoke


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj=nn.Linear(4,4);self.k_proj=nn.Linear(4,4);self.v_proj=nn.Linear(4,4);self.o_proj=nn.Linear(4,4)


class Layer(nn.Module):
    def __init__(self):super().__init__();self.self_attn=Attention();self.mlp=nn.Linear(4,4)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module();self.model.language_model=nn.Module();self.model.language_model.layers=nn.ModuleList([Layer(),Layer()])
        self.model.visual=nn.Module();self.model.visual.layers=nn.ModuleList([Layer()])


def test_only_language_q_v_paths_are_selected():
    audit=smoke.discover_attention_modules(FakeQwen())
    assert len(audit["explicit_target_modules"])==4
    assert all(name.endswith(("q_proj","v_proj")) for name in audit["explicit_target_modules"])
    assert all("visual" not in name for name in audit["explicit_target_modules"])
    assert audit["vision_target_count"]==0 and audit["forbidden_k_o_or_mlp_target_count"]==0


def test_vision_path_classifier_is_explicit():
    assert smoke.is_vision_path("model.visual.blocks.0.attn.q_proj")
    assert smoke.is_vision_path("vision_tower.encoder.layers.0.self_attn.v_proj")
    assert not smoke.is_vision_path("model.language_model.layers.0.self_attn.q_proj")


def test_lora_configuration_and_optimizer_are_frozen():
    assert (smoke.LORA_R,smoke.LORA_ALPHA,smoke.LORA_DROPOUT)==(8,16,.05)
    assert smoke.LORA_LEAVES==("q_proj","v_proj")
    assert (smoke.LR,smoke.WEIGHT_DECAY,smoke.BETAS,smoke.EPS)==(3e-5,1e-3,(.9,.999),1e-8)
    assert smoke.STEPS==50 and smoke.BATCH_SIZE==8


def test_training_path_has_full_gaussian_and_no_detach_clip_or_test():
    source=inspect.getsource(smoke.main)
    likelihood=source[source.index("benefit ="):source.index("uncertainty =")]
    assert "torch.exp(-output.benefit_log_variance)" in likelihood and ".detach()" not in likelihood
    assert "clip_grad" not in source
    assert "materialize_test" not in source
    construction=source[source.index("# Build train-only tensors"):source.index("from src.models.large_context_adapter")]
    assert "build_development_data" not in construction and "build_train_only_data" in construction
    assert "prepare_train_only_tensors" in construction


def test_parameter_group_classification_excludes_original_base():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=nn.Module();self.backbone.base=nn.Linear(4,4);self.backbone.lora_A=nn.Linear(4,2,bias=False);self.proj=nn.Linear(4,4);self.head=nn.Linear(4,1)
        def trainable_parameter_groups(self):return {"projection":list(self.proj.parameters()),"benefit_head":list(self.head.parameters()),"harm_head":[],"uncertainty_head":[]}
    model=Wrapper();original,lora,heads=smoke.parameter_groups(model)
    assert lora and heads and original
    assert not {id(parameter) for _,parameter in original}&{id(parameter) for _,parameter in lora+heads}


def test_scale_normalizer_and_adapter_reload_contracts_are_present():
    source=inspect.getsource(smoke.main)
    assert "scale_alignment_enabled" in source
    assert "len(normalizer.fit_sample_ids) != 616" in source
    assert "save_pretrained(adapter_dir" in source and "PeftModel.from_pretrained" in source
    assert '"test_materialized": False' in source


def test_zero_init_audit_decomposes_kbit_preparation_from_lora_injection():
    source=inspect.getsource(smoke.main)
    assert "prepared_output" in source and "lora_only_differences" in source
    assert '"kbit_prepared_no_LoRA_vs_zero_init_LoRA"' in source
    assert "unexplained <= 1e-5" in source
