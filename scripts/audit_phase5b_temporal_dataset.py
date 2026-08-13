"""Read-only audit of generated Phase 5B-0 protocol artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = ("existing_temporal_signal_audit.json", "temporal_feature_registry.json", "temporal_protocol.json",
            "dataset_statistics.json", "split_manifest.json", "split_leakage_audit.json",
            "static_temporal_bridge_audit.json", "mask_audit.json", "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("directory", type=Path); args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.directory / name).exists()]
    if missing: raise FileNotFoundError(f"missing Phase5B-0 artifacts: {missing}")
    summary = json.loads((args.directory / "summary.json").read_text(encoding="utf-8"))
    leakage = json.loads((args.directory / "split_leakage_audit.json").read_text(encoding="utf-8"))
    if not leakage["passed"] or summary["model_training_performed"]: raise RuntimeError("Phase5B-0 audit failed")
    print(json.dumps({"label": summary["label"], "passed": True, "phase5b1_ready": summary["phase5b1_ready"]}, indent=2))


if __name__ == "__main__": main()
