"""Print a non-destructive summary of the local project environment."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_ROOT: Final[Path] = Path("E:/HRI_World_Model_Data")
DATA_SUBDIRECTORIES: Final[tuple[str, ...]] = (
    "00_raw_rosbags",
    "01_metadata",
    "02_synchronized_trials",
    "03_processed_skeleton",
    "04_rgb_keyframes",
    "07_datasets",
    "08_models",
    "09_checkpoints",
    "10_results",
)
DEPENDENCIES: Final[dict[str, str]] = {
    "NumPy": "numpy",
    "Pandas": "pandas",
    "PyYAML": "yaml",
    "Matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "pytest": "pytest",
}


def print_status(label: str, available: bool, detail: str = "") -> None:
    """Print one consistently formatted status line."""
    marker = "是" if available else "否"
    suffix = f"（{detail}）" if detail else ""
    print(f"- {label}: {marker}{suffix}")


def module_available(module_name: str) -> bool:
    """Return whether a module can be discovered without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def print_torch_status() -> None:
    """Report PyTorch and CUDA details without failing when unavailable."""
    if not module_available("torch"):
        print_status("PyTorch 已安装", False, "仅提示，不自动安装")
        print_status("CUDA 可用", False, "PyTorch 未安装，无法检测")
        print("- GPU 名称: 无法检测")
        print("- GPU 专用显存: 无法检测")
        return

    try:
        import torch
    except Exception as exc:  # Import can fail because of binary/runtime issues.
        print_status("PyTorch 已安装", False, f"发现模块但导入失败：{exc}")
        print_status("CUDA 可用", False, "PyTorch 无法导入")
        print("- GPU 名称: 无法检测")
        print("- GPU 专用显存: 无法检测")
        return

    print_status("PyTorch 已安装", True, f"版本 {torch.__version__}")
    try:
        cuda_available = bool(torch.cuda.is_available())
        print_status("CUDA 可用", cuda_available)
        if not cuda_available:
            print("- GPU 名称: 未通过 PyTorch 检测到 CUDA GPU")
            print("- GPU 专用显存: 无法检测")
            return

        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / (1024**3)
        print(f"- GPU 名称: {properties.name}")
        print(f"- GPU 专用显存: {memory_gib:.2f} GiB")
    except Exception as exc:
        print_status("CUDA 可用", False, f"检测失败：{exc}")
        print("- GPU 名称: 无法检测")
        print("- GPU 专用显存: 无法检测")


def main() -> int:
    """Run all environment checks and always complete with a readable report."""
    print("HRI World Model 项目环境检查")
    print("=" * 36)
    print(f"- Python 版本: {platform.python_version()}")
    print(f"- 当前操作系统: {platform.system()} {platform.release()} ({platform.machine()})")
    print_status("项目根目录存在", PROJECT_ROOT.exists(), str(PROJECT_ROOT))

    print("\n数据路径（仅检查是否存在）")
    print_status("数据根目录存在", DATA_ROOT.exists(), str(DATA_ROOT))
    for name in DATA_SUBDIRECTORIES:
        path = DATA_ROOT / name
        print_status(name, path.exists(), str(path))

    print("\n计算环境")
    print_torch_status()

    print("\n基础 Python 依赖")
    for display_name, module_name in DEPENDENCIES.items():
        installed = module_available(module_name)
        detail = "" if installed else "未安装；仅提示，不自动安装"
        print_status(f"{display_name} 已安装", installed, detail)

    print("\n检查完成。未对系统、依赖或数据进行任何修改。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
