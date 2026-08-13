"""Safety and immutability gates for the Phase 5B-1.7D-A audit."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from scripts import audit_phase5b17da_ranking_failure as audit
from scripts import run_phase5b17d_manifest_v2_rebaseline as frozen
from src.training.candidate_ranking import LAMBDA_RANK

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_constants_are_unchanged():
    assert LAMBDA_RANK == .25
    assert frozen.EXPECTED_MANIFEST_SHA == "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"
    assert frozen.FROZEN_THRESHOLDS == (-.02, .2)


def test_audit_contract_has_zero_test_reads_and_no_checkpoint_save():
    source = inspect.getsource(audit)
    assert '"test_reads": 0' in source
    assert "torch.save" not in source
    assert "formal_checkpoint_written" in source
    assert "oracle_cannot_replace_formal_checkpoint" in source


def test_gradient_audit_has_no_optimizer_step():
    source = inspect.getsource(audit.gradient_audit)
    assert ".step(" not in source and "AdamW" not in source
    assert '"optimizer_step_count": 0' in source


def test_harm_v2_does_not_enter_frozen_r1_loss():
    assert "harm_v2" not in inspect.getsource(frozen.loss_terms)


def test_distribution_and_pair_statistics_are_deterministic():
    values = np.asarray((-2.0, 0.0, 1.0, 5.0))
    first, second = audit.describe(values), audit.describe(values.copy())
    assert first == second and first["count"] == 4 and first["min"] == -2.0 and first["max"] == 5.0


def test_old_17d_summary_exists_and_audit_uses_separate_output():
    assert (ROOT / "results_dev" / "phase5b17d_manifest_v2_rebaseline" / "summary.json").is_file()
    assert audit.parse_args.__module__.endswith("audit_phase5b17da_ranking_failure")
