"""Validate that PyTorch can execute a representative CUDA operation."""

from __future__ import annotations

import sys
import time


def main() -> int:
    try:
        import torch
    except ImportError:
        print("错误：未安装 PyTorch。请激活已有的 hri-wm 环境；本脚本不会安装依赖。", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"错误：PyTorch 导入失败：{exc}", file=sys.stderr)
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA 构建版本: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("错误：torch.cuda.is_available() 为 False，无法运行 GPU 基线。", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print("CUDA available: True")
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")

    try:
        left = torch.randn((2048, 2048), device=device)
        right = torch.randn((2048, 2048), device=device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = left @ right
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"2048x2048 矩阵乘法: 成功（{elapsed_ms:.3f} ms，checksum={result[0, 0].item():.6f}）")
    except RuntimeError as exc:
        print(f"错误：CUDA 矩阵乘法失败：{exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
