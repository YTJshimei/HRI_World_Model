"""Read a ROS 2 bag and export synchronized records in the Phase 2C CSV schema.

This script requires the ROS 2 Python environment already associated with the bag. It
does not install ROS packages, transform frames, mutate the bag, or control a robot.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument("--human-topic", required=True)
    parser.add_argument("--robot-odom-topic")
    parser.add_argument("--cmd-vel-topic")
    parser.add_argument("--track-id-topic")
    parser.add_argument("--requested-action-topic")
    parser.add_argument("--validated-action-topic")
    parser.add_argument("--executed-action-topic")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--person-id-anonymous", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--default-track-id", default="track_0")
    parser.add_argument("--coordinate-frame", required=True)
    parser.add_argument("--max-sync-gap", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _pose(message: Any) -> Any:
    value = getattr(message, "pose", None)
    if value is None:
        raise ValueError(f"消息 {type(message).__name__} 没有 pose 字段")
    return value.pose if hasattr(value, "pose") else value


def _twist(message: Any) -> Any | None:
    value = getattr(message, "twist", None)
    if value is None:
        return None
    return value.twist if hasattr(value, "twist") else value


def _twist_message(message: Any | None) -> Any | None:
    """Accept geometry_msgs/Twist as well as messages containing a twist field."""
    if message is None:
        return None
    nested = _twist(message)
    return message if nested is None and hasattr(message, "linear") else nested


def _frame_id(message: Any) -> str:
    header = getattr(message, "header", None)
    return str(getattr(header, "frame_id", "")).strip()


def _yaw(quaternion: Any) -> float:
    siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny, cosy)


def _nearest(samples: list[tuple[float, Any]], timestamp: float, max_gap: float) -> Any | None:
    if not samples:
        return None
    times = [item[0] for item in samples]
    index = bisect.bisect_left(times, timestamp)
    candidates = []
    if index < len(samples):
        candidates.append(samples[index])
    if index:
        candidates.append(samples[index - 1])
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest[1] if abs(nearest[0] - timestamp) <= max_gap else None


def _string_value(message: Any | None) -> str | None:
    if message is None:
        return None
    value = getattr(message, "data", None)
    return None if value is None else str(value)


def _protect_bag(args: argparse.Namespace) -> None:
    bag = args.bag.resolve()
    output = args.output.resolve()
    if not bag.exists():
        raise FileNotFoundError(f"rosbag 不存在：{bag}")
    if output == bag or (bag.is_dir() and bag in output.parents):
        raise ValueError("输出不得指向 rosbag 或 rosbag 目录内部")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在；如确认覆盖派生 CSV，请添加 --overwrite：{output}")


def read_topics(args: argparse.Namespace) -> dict[str, list[tuple[float, Any]]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "需要已有 ROS 2 Humble Python 环境（rosbag2_py/rclpy）；本脚本不会安装依赖。"
        ) from exc

    storage_options = rosbag2_py.StorageOptions(uri=str(args.bag), storage_id=args.storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    type_map = {item.name: item.type for item in reader.get_all_topics_and_types()}
    requested = {
        topic
        for topic in (
            args.human_topic,
            args.robot_odom_topic,
            args.cmd_vel_topic,
            args.track_id_topic,
            args.requested_action_topic,
            args.validated_action_topic,
            args.executed_action_topic,
        )
        if topic
    }
    missing = requested - set(type_map)
    if missing:
        raise ValueError(f"rosbag 缺少请求的 topic：{sorted(missing)}")
    message_types = {topic: get_message(type_map[topic]) for topic in requested}
    samples = {topic: [] for topic in requested}
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic in requested:
            message = deserialize_message(serialized, message_types[topic])
            samples[topic].append((timestamp_ns / 1e9, message))
    return samples


def convert(args: argparse.Namespace) -> list[Any]:
    from src.data.ros_trajectory_schema import RosTrajectoryRecord

    samples = read_topics(args)
    humans = samples[args.human_topic]
    if not humans:
        raise ValueError(f"human topic 没有消息：{args.human_topic}")
    records = []
    for timestamp, human_message in humans:
        human_frame = _frame_id(human_message)
        if human_frame and human_frame != args.coordinate_frame:
            raise ValueError(
                f"human topic frame={human_frame!r}，目标 frame={args.coordinate_frame!r}；"
                "请先使用记录的 /tf 与 /tf_static 统一坐标，禁止直接混用。"
            )
        human_pose = _pose(human_message)
        human_twist = _twist(human_message)
        robot_message = _nearest(
            samples.get(args.robot_odom_topic, []), timestamp, args.max_sync_gap
        )
        robot_frame = _frame_id(robot_message) if robot_message is not None else ""
        if robot_frame and robot_frame != args.coordinate_frame:
            raise ValueError(
                f"robot odometry frame={robot_frame!r}，目标 frame={args.coordinate_frame!r}；"
                "必须先统一坐标系。"
            )
        cmd_message = _nearest(
            samples.get(args.cmd_vel_topic, []), timestamp, args.max_sync_gap
        )
        robot_pose = _pose(robot_message) if robot_message is not None else None
        robot_twist = _twist(robot_message) if robot_message is not None else None
        robot_x = float(robot_pose.position.x) if robot_pose is not None else None
        robot_y = float(robot_pose.position.y) if robot_pose is not None else None
        robot_yaw = float(_yaw(robot_pose.orientation)) if robot_pose is not None else None
        human_x = float(human_pose.position.x)
        human_y = float(human_pose.position.y)
        distance = bearing = None
        if robot_x is not None and robot_y is not None and robot_yaw is not None:
            dx, dy = human_x - robot_x, human_y - robot_y
            distance = math.hypot(dx, dy)
            bearing = math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw), math.cos(math.atan2(dy, dx) - robot_yaw))
        track_message = _nearest(
            samples.get(args.track_id_topic, []), timestamp, args.max_sync_gap
        )
        track_id = _string_value(track_message) or args.default_track_id
        cmd_twist = _twist_message(cmd_message)
        records.append(
            RosTrajectoryRecord(
                timestamp=timestamp,
                trial_id=args.trial_id,
                session_id=args.session_id,
                person_id_anonymous=args.person_id_anonymous,
                scene_id=args.scene_id,
                track_id=track_id,
                coordinate_frame=args.coordinate_frame,
                human_x=human_x,
                human_y=human_y,
                human_z=float(human_pose.position.z),
                human_vx=float(human_twist.linear.x) if human_twist is not None else None,
                human_vy=float(human_twist.linear.y) if human_twist is not None else None,
                robot_x=robot_x,
                robot_y=robot_y,
                robot_yaw=robot_yaw,
                robot_linear_velocity=float(robot_twist.linear.x) if robot_twist is not None else None,
                robot_angular_velocity=float(robot_twist.angular.z) if robot_twist is not None else None,
                cmd_vel_linear=float(cmd_twist.linear.x) if cmd_twist is not None else None,
                cmd_vel_angular=float(cmd_twist.angular.z) if cmd_twist is not None else None,
                human_robot_distance=distance,
                relative_bearing=bearing,
                requested_action=_string_value(_nearest(samples.get(args.requested_action_topic, []), timestamp, args.max_sync_gap)),
                validated_action=_string_value(_nearest(samples.get(args.validated_action_topic, []), timestamp, args.max_sync_gap)),
                executed_action=_string_value(_nearest(samples.get(args.executed_action_topic, []), timestamp, args.max_sync_gap)),
            )
        )
    # Derive missing human velocity by finite differences without changing positions.
    derived = []
    for index, record in enumerate(records):
        if record.human_vx is not None and record.human_vy is not None:
            derived.append(record)
            continue
        if index == 0:
            derived.append(record)
            continue
        previous = records[index - 1]
        delta = record.timestamp - previous.timestamp
        if delta > 0:
            derived.append(
                replace(
                    record,
                    human_vx=(record.human_x - previous.human_x) / delta,
                    human_vy=(record.human_y - previous.human_y) / delta,
                )
            )
        else:
            derived.append(record)
    return derived


def main() -> int:
    args = parse_args()
    try:
        _protect_bag(args)
        records = convert(args)
        from src.data.ros_trajectory_loader import write_trajectory_csv

        write_trajectory_csv(args.output, records)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"已只读转换 {len(records)} 条记录：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
