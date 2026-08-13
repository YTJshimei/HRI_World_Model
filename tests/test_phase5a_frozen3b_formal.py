import inspect
import json

import pytest

torch = pytest.importorskip("torch")

from scripts import run_phase5a_frozen3b as base
from scripts import run_phase5a_frozen3b_formal as formal


def test_formal_optimizer_contract_is_frozen_to_cs6():
    assert formal.FORMAL_LR == 3e-5
    assert formal.WEIGHT_DECAY == 1e-3
    assert formal.BETAS == (0.9, 0.999)
    assert formal.EPS == 1e-8
    assert formal.BATCH_SIZE == 8
    assert formal.MAX_EPOCHS == 25 and formal.PATIENCE == 5


def test_formal_training_has_full_gaussian_path_and_no_clipping_or_detach():
    source = inspect.getsource(formal.train_formal)
    assert "torch.exp(-output.benefit_log_variance)" in source
    assert ".detach()" not in source[source.index("benefit ="):source.index("uncertainty =")]
    assert "clip_grad" not in source and "clip_trainable_gradients" not in source
    assert "optimizer.step()" in source
    assert "scale_alignment_enabled" in inspect.getsource(formal.assert_formal_contract)


def test_test_guard_refuses_early_and_second_materialization():
    guard = formal.TestAccessGuard()
    with pytest.raises(RuntimeError, match="before checkpoint"):
        guard.consume()
    guard.lock("a" * 64, (0.02, 0.5)); guard.consume()
    with pytest.raises(RuntimeError, match="only be materialized once"):
        guard.consume()


def test_checkpoint_selection_and_test_order_is_source_enforced():
    source = inspect.getsource(formal.main)
    selection_write = source.index('"checkpoint_selection.json"')
    guard_lock = source.index("guard.lock")
    materialize = source.index("base.materialize_test")
    assert selection_write < guard_lock < materialize
    assert "test_used_for_selection" not in inspect.getsource(formal.train_formal)


def test_formal_comparison_uses_frozen_l1_reference_values():
    source = inspect.getsource(formal.main)
    assert 'frozen_l1 = frozen_summary["models"]["L1"]' in source
    assert '"Beneficial_Switch_Recall": frozen_l1["Beneficial_Switch_Recall"]' in source


def test_qwen_optimizer_is_limited_to_projection_and_heads():
    source = inspect.getsource(formal.train_formal)
    assert "parameters = base.trainable_parameters(model)" in source
    assert "base.frozen_audit(model, optimizer)" in source
    assert "qwen_optimizer_parameter_count" in source


def test_clean_json_rejects_nonfinite_by_conversion():
    cleaned = base.clean({"nan": float("nan"), "inf": float("inf"), "ok": 1.0})
    assert cleaned == {"nan": None, "inf": None, "ok": 1.0}
    json.dumps(cleaned, allow_nan=False)
