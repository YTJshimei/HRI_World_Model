# HRI World Model

## 项目目标

本项目研究“面向人机共适应的个体化主动人体响应世界模型与移动机器人闭环决策”。目标是建立能够预测个体人体响应、随长期交互持续适应，并支持移动机器人安全主动决策的多模态世界模型。

当前仅进行 **Phase 1：运行环境和统一数据格式**。完整阶段规划见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 工程目录

```text
C:\HRI_World_Model
├── configs                  # 示例配置
├── docs                     # 设计与研究文档
├── scripts                  # 独立检查和辅助脚本
├── src
│   ├── data                 # 数据接口与处理
│   ├── models               # 模型定义
│   ├── training             # 训练流程
│   ├── evaluation           # 评估流程
│   └── utils                # 通用工具
└── tests                    # 单元测试
```

## 代码和数据分离

- 项目代码位于 `C:\HRI_World_Model`，纳入版本控制。
- 实验数据位于 `E:\HRI_World_Model_Data`，不复制进本仓库，也不纳入版本控制。
- 路径模板位于 `configs/paths.example.yaml`。需要本地配置时应复制为单独的本地配置文件，并避免提交包含机器特定信息或敏感信息的配置。
- 原始 ROS 系统保持独立，本仓库不直接修改 Windows、ROS、VMware、CUDA 或显卡配置。
- 模型权重、检查点、数据集和实验结果不存放在代码仓库内。

## 项目环境检查

在项目根目录运行：

```powershell
python scripts/check_project.py
```

脚本只读取环境和路径状态并打印结果，不安装依赖、不修改系统，也不修改实验数据。运行脚本会检查 `E:\HRI_World_Model_Data` 及约定子目录是否存在。

## 运行测试

在项目根目录运行：

```powershell
python -m pytest
```

基础依赖清单位于 `requirements-base.txt`。缺失依赖需由项目成员按既定环境管理流程处理，本项目脚本不会自动安装。

## 模型依赖状态

当前尚未下载或安装任何大模型，也未在基础依赖中加入 `torch`、`transformers`、`bitsandbytes` 或其他大型模型依赖。相关依赖将在后续阶段经过资源与环境评审后再确定。
