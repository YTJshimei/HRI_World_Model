# Phase 2C ROS1 Noetic 数据采集与转换

## 实际平台与边界

- 机器人系统：Ubuntu 20.04。
- 中间件：ROS1 Noetic。
- catkin workspace：`~/gstest`。
- 当前组件只读取 `/human/target` 与 TF，并发布标准化的人体 pose/track；不订阅、生成或发布 `/cmd_vel`，不控制机器人。
- `rosbag record` 可以只读记录已有 `/cmd_vel`，这不等于发布控制命令。

## 推荐部署位置

将模板放入一个 catkin package（文档约定 package 名为 `hri_world_model_adapter`）：

```text
~/gstest/src/hri_world_model_adapter/
├── scripts/
│   └── phase2c_human_pose_adapter.py
└── launch/
    └── phase2c_data_adapter.launch
```

模板源文件位于本仓库 `ros1/`。复制后为 Python 节点设置可执行权限，随后在 `~/gstest` 运行 `catkin_make`。若使用其他 package 名，必须同步修改 launch 中 `<node pkg="...">`。

## catkin_make 前的只读检查

```bash
source /opt/ros/noetic/setup.bash
source ~/gstest/devel/setup.bash

rospack find rospy
rospack find geometry_msgs
rospack find std_msgs
rospack find tf2_ros
rospack find tf2_geometry_msgs
rospack find follow_msgs

rosmsg show follow_msgs/HumanTarget

python3 -c "import rospy, rosbag, tf2_ros, tf2_geometry_msgs"
python3 -c "from follow_msgs.msg import HumanTarget"

cd ~/gstest
rosdep check --from-paths src --ignore-src
```

这些命令只检查环境。缺失项应按机器人既有依赖管理流程解决，本项目不会自动安装。

`follow_msgs/HumanTarget` 至少应确认存在：`track_id`、`position.x/y/z`、`distance`、`bearing`、`confidence`、`locked`、`is_valid`。还应人工确认它是否真的定义了 `std_msgs/Header`，不能仅凭 topic 频率推断 timestamp 或 frame。

## 在线 adapter

启动前必须知道 HumanTarget 的真实 source frame，并确认 TF 树能在消息时间提供 `target_frame <- source_frame`：

```bash
source /opt/ros/noetic/setup.bash
source ~/gstest/devel/setup.bash

roslaunch hri_world_model_adapter phase2c_data_adapter.launch \
  source_frame:=camera_link \
  target_frame:=map \
  input_topic:=/human/target \
  pose_output_topic:=/wm/human_center_pose \
  track_output_topic:=/wm/human_track_id
```

行为约定：

- HumanTarget 有非零 Header stamp 时沿用真实 stamp。
- HumanTarget 没有 Header 或 stamp 为零时使用 callback receipt time，并计入 `receipt_timestamp` 统计。
- `is_valid=false`、position 缺失、track ID 非整数、source frame 冲突或 TF 失败时不发布该帧。
- TF lookup/connectivity/extrapolation 分别计数并在 shutdown 时输出。
- 发布 pose 为 `geometry_msgs/PoseStamped`；track 为 `std_msgs/Int64`。
- 原始 `/human/target` 不被修改。

## 推荐录制 topics

```text
/wm/human_center_pose
/wm/human_track_id
/human/target
/human/keypoints
/human/motion_feature
/odom
/cmd_vel
/tf
/tf_static
```

其中 `/wm/human_center_pose` 和 `/wm/human_track_id` 是 Phase 2C 轨迹训练的主输入；原始人体 topics 用于审计和失败分析；`/odom`、`/cmd_vel` 用于机器人状态与命令上下文；`/tf`、`/tf_static` 用于复核坐标变换。录包进程只订阅并写盘，不发布任何控制命令。

第一次录包模板：

```bash
mkdir -p ~/hri_bags
rosbag record --buffsize=1024 -O ~/hri_bags/phase2c_trial_001.bag \
  /wm/human_center_pose \
  /wm/human_track_id \
  /human/target \
  /human/keypoints \
  /human/motion_feature \
  /odom \
  /cmd_vel \
  /tf \
  /tf_static
```

录制前用 `rostopic list` 删除机器人上实际不存在的可选 topic；`/wm/human_center_pose`、`/wm/human_track_id` 和支持其坐标变换的 TF 必须存在。

录制后只读检查：

```bash
rosbag info --yaml ~/hri_bags/phase2c_trial_001.bag
rosbag check ~/hri_bags/phase2c_trial_001.bag
```

## 为什么必须使用固定世界坐标系

人体在 `camera_link` 或 `base_link` 中的轨迹同时包含人体运动和机器人自身的平移/旋转。同一个静止人体可能因为机器人转弯而在传感器坐标中产生大幅运动，模型会错误地学习机器人自运动。将人体中心变换到 `map` 等固定世界坐标后，速度、加速度、ADE/FDE 和 constant-velocity prior 才具有一致物理意义；跨 trial、跨机器人位姿的窗口也才可比较。

不得把不同 frame 的数值直接拼入同一个 track。TF 不可用时宁可丢弃并统计该帧，也不能使用单位变换或最近 frame 冒充。

## ROS1 bag 转换

主路径应转换适配后的 `/wm/human_center_pose`，因为它已经位于固定世界 frame。转换器只读 `.bag`，输出标准 CSV 和同名 `.metadata.json`；metadata 保存 timestamp 来源与 HumanTarget 质量统计。

对于原始、无 Header 的 `follow_msgs/HumanTarget`，离线转换只能使用 rosbag record time，metadata 会写 `bag_message_timestamp`。离线 bag 不包含 callback receipt time，转换器不会伪造。若在线 adapter 用 receipt time 发布 PoseStamped，应在转换时用 `--header-stamp-semantics receipt_time` 明确记录。

转换后的 CSV 仍必须经过 `data_quality`、10 Hz resampler、window builder 和 split manifest；不得直接绕过这些步骤训练。
