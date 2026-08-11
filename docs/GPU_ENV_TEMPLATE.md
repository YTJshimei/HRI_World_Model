# GPU 环境记录模板

每次建立可复现实验环境后，复制并填写本模板。版本信息应以目标 WSL 虚拟环境中的命令输出为准。

| 项目 | 当前 Phase 2 环境 |
|---|---|
| 环境名称 | `~/venvs/hri-wm` |
| Python 版本 | `3.10.12` |
| PyTorch 版本 | `2.11.0+cu128` |
| CUDA 版本（PyTorch 构建版本） | `12.8` |
| GPU 型号 | `NVIDIA GeForce RTX 4070 Laptop GPU` |
| VRAM | `8 GB` |

建议保存以下命令的完整输出：

```bash
source ~/venvs/hri-wm/bin/activate
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
python scripts/check_gpu.py
```

注意：`torch.version.cuda` 是 PyTorch 的 CUDA 构建版本，不一定等于系统驱动支持的最高 CUDA 版本。
