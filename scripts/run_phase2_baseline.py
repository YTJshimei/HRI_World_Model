"""Train and evaluate M0/M1/M2 on deterministic synthetic trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    return parser.parse_args()


def constant_velocity(history: np.ndarray, future_length: int = 10) -> np.ndarray:
    velocity = history[:, -1] - history[:, -2]
    steps = np.arange(1, future_length + 1, dtype=np.float32)[None, :, None]
    return history[:, -1:, :] + steps * velocity[:, None, :]


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs 和 --batch-size 必须为正整数")
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("错误：训练需要已有的 PyTorch 环境；不会自动安装依赖。", file=sys.stderr)
        return 1

    from src.data.synthetic_trajectory import as_tensor_dataset, create_splits
    from src.evaluation.trajectory_metrics import ade_fde, parameter_count
    from src.models.lstm_trajectory import LSTMTrajectoryPredictor
    from src.models.transformer_trajectory import TransformerTrajectoryPredictor
    from src.training.train_trajectory import evaluate_model, train_model

    if args.device == "cuda" and not torch.cuda.is_available():
        print("错误：请求了 --device cuda，但 torch.cuda.is_available() 为 False。", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    splits = create_splits(args.train_size, args.val_size, args.test_size, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(as_tensor_dataset(splits.train), batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(as_tensor_dataset(splits.val), batch_size=args.batch_size)
    test_loader = DataLoader(as_tensor_dataset(splits.test), batch_size=args.batch_size)

    started = time.perf_counter()
    m0_prediction = constant_velocity(splits.test[0])
    m0_ms = (time.perf_counter() - started) * 1000 / args.test_size
    m0_ade, m0_fde = ade_fde(m0_prediction, splits.test[1])
    results = {"M0_constant_velocity": {"ADE": m0_ade, "FDE": m0_fde, "inference_ms_per_sample": m0_ms, "parameters": 0}}
    output_dir = PROJECT_ROOT / "results_dev"
    output_dir.mkdir(exist_ok=True)

    models = {
        "M1_LSTM": LSTMTrajectoryPredictor(),
        "M2_Transformer": TransformerTrajectoryPredictor(),
    }
    for name, model in models.items():
        print(f"\n训练 {name}")
        train_model(model, train_loader, val_loader, device, args.epochs)
        ade, fde, inference_ms = evaluate_model(model, test_loader, device)
        results[name] = {"ADE": ade, "FDE": fde, "inference_ms_per_sample": inference_ms, "parameters": parameter_count(model)}
        torch.save(model.state_dict(), output_dir / f"{name.lower()}.pt")

    payload = {"config": vars(args), "results": results}
    result_path = output_dir / "phase2_baseline.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已保存：{result_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
