# Phase 2C 真实人体中心轨迹数据协议

## 1. 范围与不可变约束

本阶段只读取 rosbag 并生成派生的标准轨迹 CSV，不控制机器人，不修改或删除原始 rosbag，不读取未经授权的身份信息。Phase 2B 的 M0/M1/M2/M3 模型结构和既有结果保持冻结。

mock 数据只用于单元测试，必须标记为测试 fixture，不得进入正式实验表格或结论。

## 2. 标准数据格式

标准载体为 UTF-8 CSV，一行表示同一人体 track 在一个时间戳上的观测。完整表头由 `RosTrajectoryRecord.CSV_FIELDS` 固定，包括：

- 标识与时间：`timestamp`、`trial_id`、`session_id`、`person_id_anonymous`、`scene_id`、`track_id`。
- 坐标系：`coordinate_frame`。所有人体和机器人位置必须已表达在该坐标系中。
- 人体：`human_x/y/z`、`human_vx/vy`。
- 机器人：`robot_x/y/yaw`、`robot_linear_velocity`、`robot_angular_velocity`。
- 命令：`cmd_vel_linear`、`cmd_vel_angular`。
- 相对状态：`human_robot_distance`、`relative_bearing`。
- 动作审计：`requested_action`、`validated_action`、`executed_action`。

暂时不存在的数值或动作字段使用空 CSV 单元格，不用零值伪装缺失。训练窗口要求 `timestamp`、分组标识、`coordinate_frame` 和 `human_x/y` 有效。

## 3. rosbag 到标准 CSV

### 3.1 ROS1 Noetic `.bag`（当前实际机器人主路径）

实际机器人为 Ubuntu 20.04 + ROS1 Noetic，catkin workspace 为 `~/gstest`。推荐在线运行只读 `phase2c_human_pose_adapter.py`，利用 tf2 将 `/human/target` 转为固定世界坐标的 `/wm/human_center_pose`，并同时发布 `/wm/human_track_id`。随后使用 `rosbag record` 记录标准化 topic、原始人体 topic、odom、已有 cmd_vel 与 TF。

ROS1 `.bag` 使用 `scripts/convert_ros1_bag_trajectory.py`。`follow_msgs/HumanTarget` 没有有效 Header 时，转换器使用 rosbag record time，并在 sidecar metadata 中写明 `timestamp_source=bag_message_timestamp`；它不会假装存在 header stamp，也无法从离线 bag 恢复 callback receipt time。source frame 必须显式传入，缺失时拒绝转换。

详细部署与采集步骤见 `docs/ROS1_DATA_COLLECTION.md`。

### 3.2 ROS2 `.db3`/MCAP（保留兼容，非当前主路径）

现有 `scripts/convert_rosbag_trajectory.py` 保留用于 ROS2 Humble 的 `.db3`/MCAP 数据。它不是当前机器人平台的主转换路径，不得用来直接读取 ROS1 `.bag`。

1. 对原始 bag 做只读 topic/type/时间范围检查，并保存 bag 的只读校验信息。
2. 选择人体中心 topic，确认 `track_id` 的来源及人体 pose 的 frame。
3. 使用 bag 中 `/tf`、`/tf_static` 将人体与机器人 pose 统一到同一固定 frame（推荐 `map` 或实验定义的 world frame）。本仓库转换器遇到 frame 不一致会终止，不会猜测变换。
4. 以人体观测时间戳为基准，采用最近邻同步机器人里程计、`cmd_vel` 和动作事件；默认最大同步误差 0.05 s。连续位置字段随后在数据适配阶段重采样。
5. 使用匿名 person ID 导出派生 CSV。禁止把姓名、面部标识、学号或原始受试者编号写入代码仓库。
6. 先运行质量检查，再冻结 split manifest，之后才允许重采样和生成窗口。

ROS2 转换器只支持 ROS2 原生 Python 环境中能够反序列化的消息。ROS1 转换器支持 `geometry_msgs/PoseStamped` 和 `follow_msgs/HumanTarget`。其他消息类型应在独立适配器中显式映射，不能静默猜字段。

## 4. 时间同步与重采样

- 标准 timestamp 单位为 Unix/ROS 秒，必须单调排序。
- 默认目标频率 10 Hz，即 timestep 0.1 s，可通过 CLI 配置。
- 默认只插值不超过 0.3 s 的短缺帧；动作字段不插值。
- 大于阈值的间隔强制切段，不允许跨段生成窗口。
- 重复时间戳、无法对齐目标时间网格、NaN/Inf、跳变和 track 切换必须进入质量报告。
- track、scene、person 或 coordinate frame 改变时不得跨界插值。

## 5. 坐标统一

人体、机器人位置和速度必须使用同一右手坐标系。`robot_yaw`、`relative_bearing` 用弧度，bearing 归一化到 `[-pi, pi]`。禁止把 `camera_link`、`base_link` 和 `map` 中的数值直接拼接。需要变换时必须使用 rosbag 同期记录的 `/tf` 与 `/tf_static`，并记录目标 frame 与变换版本。

## 6. 分组、划分与禁止泄漏规则

split manifest 是 JSON，且只允许三个顶层字段：

```json
{
  "train_trials": ["trial_001"],
  "validation_trials": ["trial_002"],
  "test_trials": ["trial_003"]
}
```

必须先按完整 `(trial_id, session_id, person_id_anonymous)` 分组并依据 trial manifest 划分原始记录，然后在每个 split 内独立排序、重采样和生成窗口。严禁先生成所有窗口再随机拆分；严禁同一 trial 出现在多个 split；不得使用 test 指标选择 epoch、阈值、插值策略或模型。训练只按 validation ADE 保存 checkpoint，训练结束后 test 只物化和评价一次。

如果研究协议要求 person-independent 或 session-independent 泛化，应进一步保证同一 `person_id_anonymous` 或 `session_id` 不跨 split，并在 manifest 审核阶段显式检查。

## 7. 匿名化

- `person_id_anonymous` 必须由受控映射生成；映射密钥不得进入本仓库或模型产物。
- 标准 CSV 不保存姓名、联系方式、面部图像路径或其他直接身份标识。
- trial/session ID 不得编码身份信息。
- 访问、保存期限和删除流程服从伦理审批与数据管理方案。
- mock fixture 使用明显的虚构 ID，并且只用于测试。

## 8. 最少 topic 与转换命令

当前 ROS1 主路径推荐记录 `/wm/human_center_pose`、`/wm/human_track_id`、`/human/target`、`/human/keypoints`、`/human/motion_feature`、`/odom`、`/cmd_vel`、`/tf`、`/tf_static`。这里记录 `/cmd_vel` 仅是只读数据采集，不生成或发布控制命令。

仅训练人体中心轨迹时，最低必须有一个带 timestamp、frame 和人体中心 `(x,y)` 的 topic，例如 `/human/center_pose`。为了填充机器人相对状态并审计交互，建议同时具备：

- `/human/center_pose`：`geometry_msgs/PoseStamped` 或兼容 pose 消息；必须。
- `/human/track_id`：`std_msgs/String` 或由人体消息自身提供稳定 track ID；建议，缺失时只能为单人 trial 指定固定 ID。
- `/odom`：`nav_msgs/Odometry`；建议。
- `/cmd_vel`：`geometry_msgs/Twist`；建议。
- `/tf` 和 `/tf_static`：当人体与机器人不是同一固定 frame 时必须。
- requested/validated/executed action 的审计 topic：有交互动作时建议记录。

命令模板见项目交付说明。输出应写入派生数据目录，绝不能写进 rosbag 目录。
