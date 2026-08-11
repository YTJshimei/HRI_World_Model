"""Read a ROS1 Noetic .bag and export Phase 2C trajectory CSV plus metadata.

ROS imports are intentionally lazy so parser tests work without a ROS installation.
The converter is read-only: it never publishes topics or mutates the input bag.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ParsedHumanObservation:
    timestamp: float
    timestamp_source: str
    source_frame: str
    track_id: str | None
    x: float
    y: float
    z: float
    distance: float | None
    bearing: float | None
    confidence: float | None
    locked: bool | None
    is_valid: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--human-topic", required=True)
    parser.add_argument(
        "--human-message-type",
        choices=("auto", "pose_stamped", "human_target"),
        default="auto",
    )
    parser.add_argument(
        "--timestamp-policy", choices=("auto", "header", "bag"), default="auto"
    )
    parser.add_argument(
        "--header-stamp-semantics",
        choices=("sensor_header", "receipt_time"),
        default="sensor_header",
        help="Describe what a valid PoseStamped/Header timestamp represents.",
    )
    parser.add_argument("--source-frame", required=True)
    parser.add_argument("--coordinate-frame", required=True)
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
    parser.add_argument("--max-sync-gap", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _seconds(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "to_sec"):
        return float(value.to_sec())
    if hasattr(value, "secs"):
        return float(value.secs) + float(getattr(value, "nsecs", 0)) * 1e-9
    raise ValueError(f"无法解析 ROS timestamp：{value!r}")


def _header_info(message: Any) -> tuple[float | None, str]:
    header = getattr(message, "header", None)
    if header is None:
        return None, ""
    frame = str(getattr(header, "frame_id", "")).strip()
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None, frame
    seconds = _seconds(stamp)
    return (seconds if seconds > 0 else None), frame


def _position(message: Any) -> Any:
    if hasattr(message, "position"):
        return message.position
    pose = getattr(message, "pose", None)
    if pose is not None:
        pose = pose.pose if hasattr(pose, "pose") else pose
        if hasattr(pose, "position"):
            return pose.position
    raise ValueError(f"消息 {type(message).__name__} 缺少 position 或 pose.position")


def _optional_float(message: Any, name: str) -> float | None:
    value = getattr(message, name, None)
    return None if value is None else float(value)


def infer_human_message_kind(message_type: str, message: Any) -> str:
    normalized = message_type.lower()
    if normalized.endswith("geometry_msgs/posestamped") or normalized.endswith("pose_stamped"):
        return "pose_stamped"
    if normalized.endswith("follow_msgs/humantarget") or normalized.endswith("human_target"):
        return "human_target"
    if hasattr(message, "track_id") and hasattr(message, "position"):
        return "human_target"
    if hasattr(message, "pose"):
        return "pose_stamped"
    raise ValueError(f"不支持的人体消息类型：{message_type}")


def parse_human_message(
    message: Any,
    message_type: str,
    bag_timestamp: Any,
    source_frame: str,
    timestamp_policy: str = "auto",
) -> ParsedHumanObservation:
    """Pure-Python parser for PoseStamped and follow_msgs/HumanTarget-like objects."""
    source_frame = source_frame.strip()
    if not source_frame:
        raise ValueError("source_frame 缺失；禁止猜测 HumanTarget 坐标系")
    kind = infer_human_message_kind(message_type, message)
    header_timestamp, header_frame = _header_info(message)
    if header_frame and header_frame != source_frame:
        raise ValueError(
            f"消息 header.frame_id={header_frame!r} 与 --source-frame={source_frame!r} 不一致"
        )
    if timestamp_policy not in ("auto", "header", "bag"):
        raise ValueError(f"无效 timestamp_policy：{timestamp_policy}")
    if timestamp_policy == "header" or (timestamp_policy == "auto" and header_timestamp is not None):
        if header_timestamp is None:
            raise ValueError("请求 header timestamp，但消息没有真实有效的 Header stamp")
        timestamp = header_timestamp
        timestamp_source = "header_stamp"
    else:
        timestamp = _seconds(bag_timestamp)
        timestamp_source = "bag_message_timestamp"
    position = _position(message)
    track_value = getattr(message, "track_id", None) if kind == "human_target" else None
    return ParsedHumanObservation(
        timestamp=timestamp,
        timestamp_source=timestamp_source,
        source_frame=source_frame,
        track_id=None if track_value is None else str(track_value),
        x=float(position.x),
        y=float(position.y),
        z=float(position.z),
        distance=_optional_float(message, "distance") if kind == "human_target" else None,
        bearing=_optional_float(message, "bearing") if kind == "human_target" else None,
        confidence=_optional_float(message, "confidence") if kind == "human_target" else None,
        locked=(bool(message.locked) if kind == "human_target" and hasattr(message, "locked") else None),
        is_valid=(bool(message.is_valid) if kind == "human_target" and hasattr(message, "is_valid") else True),
    )


def _pose(message: Any) -> Any:
    pose = getattr(message, "pose", None)
    if pose is None:
        raise ValueError(f"消息 {type(message).__name__} 没有 pose")
    return pose.pose if hasattr(pose, "pose") else pose


def _twist(message: Any | None) -> Any | None:
    if message is None:
        return None
    value = getattr(message, "twist", None)
    if value is None:
        return message if hasattr(message, "linear") else None
    return value.twist if hasattr(value, "twist") else value


def _yaw(quaternion: Any) -> float:
    siny = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    return math.atan2(siny, cosy)


def _nearest(samples: list[tuple[float, Any]], timestamp: float, max_gap: float) -> Any | None:
    if not samples:
        return None
    times = [item[0] for item in samples]
    index = bisect.bisect_left(times, timestamp)
    candidates = samples[max(0, index - 1) : min(len(samples), index + 1)]
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest[1] if abs(nearest[0] - timestamp) <= max_gap else None


def _scalar(message: Any | None) -> str | None:
    if message is None:
        return None
    value = getattr(message, "data", None)
    return None if value is None else str(value)


def _metadata_path(args: argparse.Namespace) -> Path:
    return args.metadata_output or args.output.with_suffix(".metadata.json")


def protect_input_bag(args: argparse.Namespace) -> None:
    bag = args.bag.resolve()
    output = args.output.resolve()
    metadata = _metadata_path(args).resolve()
    if not bag.is_file() or bag.suffix.lower() != ".bag":
        raise FileNotFoundError(f"ROS1 .bag 文件不存在：{bag}")
    if output == bag or metadata == bag:
        raise ValueError("输出或 metadata 不得覆盖原始 .bag")
    if output == metadata:
        raise ValueError("CSV 输出与 metadata 输出必须是不同文件")
    for path in (output, metadata):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"派生输出已存在；确认后才可添加 --overwrite：{path}")


def read_ros1_bag(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, list[tuple[float, Any]]]]:
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError(
            "未找到 ROS1 rosbag Python 模块。请在 Ubuntu 20.04/ROS Noetic 且已 source 环境中运行；"
            "本脚本不会安装依赖。"
        ) from exc
    topics = {
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
    try:
        with rosbag.Bag(str(args.bag), "r") as bag:
            info = bag.get_type_and_topic_info().topics
            missing = topics - set(info)
            if missing:
                raise ValueError(f"ROS1 bag 缺少请求的 topic：{sorted(missing)}")
            topic_types = {topic: info[topic].msg_type for topic in topics}
            samples = {topic: [] for topic in topics}
            for topic, message, bag_time in bag.read_messages(topics=sorted(topics)):
                samples[topic].append((_seconds(bag_time), message))
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "无法读取 ROS1 bag。若包含 follow_msgs/HumanTarget，请先 source 对应 catkin workspace。"
        ) from exc
    return topic_types, samples


def convert_samples(
    args: argparse.Namespace,
    topic_types: dict[str, str],
    samples: dict[str, list[tuple[float, Any]]],
) -> tuple[list[Any], dict[str, Any]]:
    from src.data.ros_trajectory_schema import RosTrajectoryRecord

    if args.source_frame != args.coordinate_frame:
        raise ValueError(
            "离线转换器不执行 TF：--source-frame 必须等于 --coordinate-frame。"
            "推荐录制在线适配后的 /wm/human_center_pose。"
        )
    human_samples = samples.get(args.human_topic, [])
    if not human_samples:
        raise ValueError(f"人体 topic 没有消息：{args.human_topic}")
    detected_type = topic_types[args.human_topic]
    requested_kind = args.human_message_type
    records = []
    parsed_observations = []
    invalid_count = 0
    for bag_timestamp, message in human_samples:
        kind = infer_human_message_kind(detected_type, message)
        if requested_kind != "auto" and requested_kind != kind:
            raise ValueError(
                f"--human-message-type={requested_kind} 与 bag 类型 {detected_type} 不一致"
            )
        observation = parse_human_message(
            message, detected_type, bag_timestamp, args.source_frame, args.timestamp_policy
        )
        parsed_observations.append(observation)
        if not observation.is_valid:
            invalid_count += 1
            continue
        robot_message = _nearest(
            samples.get(args.robot_odom_topic, []), observation.timestamp, args.max_sync_gap
        )
        robot_pose = _pose(robot_message) if robot_message is not None else None
        robot_twist = _twist(robot_message)
        if robot_message is not None:
            _, robot_frame = _header_info(robot_message)
            if robot_frame and robot_frame != args.coordinate_frame:
                raise ValueError(
                    f"odom frame={robot_frame!r} 与 coordinate_frame={args.coordinate_frame!r} 不一致"
                )
        robot_x = float(robot_pose.position.x) if robot_pose is not None else None
        robot_y = float(robot_pose.position.y) if robot_pose is not None else None
        robot_yaw = float(_yaw(robot_pose.orientation)) if robot_pose is not None else None
        distance = observation.distance
        bearing = observation.bearing
        if distance is None and robot_x is not None and robot_y is not None:
            distance = math.hypot(observation.x - robot_x, observation.y - robot_y)
        if bearing is None and robot_x is not None and robot_y is not None and robot_yaw is not None:
            raw = math.atan2(observation.y - robot_y, observation.x - robot_x) - robot_yaw
            bearing = math.atan2(math.sin(raw), math.cos(raw))
        track_message = _nearest(
            samples.get(args.track_id_topic, []), observation.timestamp, args.max_sync_gap
        )
        track_id = observation.track_id or _scalar(track_message) or args.default_track_id
        cmd = _twist(_nearest(samples.get(args.cmd_vel_topic, []), observation.timestamp, args.max_sync_gap))
        records.append(
            RosTrajectoryRecord(
                timestamp=observation.timestamp,
                trial_id=args.trial_id,
                session_id=args.session_id,
                person_id_anonymous=args.person_id_anonymous,
                scene_id=args.scene_id,
                track_id=track_id,
                coordinate_frame=args.coordinate_frame,
                human_x=observation.x,
                human_y=observation.y,
                human_z=observation.z,
                robot_x=robot_x,
                robot_y=robot_y,
                robot_yaw=robot_yaw,
                robot_linear_velocity=(float(robot_twist.linear.x) if robot_twist else None),
                robot_angular_velocity=(float(robot_twist.angular.z) if robot_twist else None),
                cmd_vel_linear=(float(cmd.linear.x) if cmd else None),
                cmd_vel_angular=(float(cmd.angular.z) if cmd else None),
                human_robot_distance=distance,
                relative_bearing=bearing,
                requested_action=_scalar(_nearest(samples.get(args.requested_action_topic, []), observation.timestamp, args.max_sync_gap)),
                validated_action=_scalar(_nearest(samples.get(args.validated_action_topic, []), observation.timestamp, args.max_sync_gap)),
                executed_action=_scalar(_nearest(samples.get(args.executed_action_topic, []), observation.timestamp, args.max_sync_gap)),
            )
        )
    records.sort(key=lambda record: record.timestamp)
    derived = []
    for index, record in enumerate(records):
        if index == 0 or record.track_id != records[index - 1].track_id:
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
    timestamp_counts = Counter(item.timestamp_source for item in parsed_observations)
    confidence_values = [item.confidence for item in parsed_observations if item.confidence is not None]
    locked_values = [item.locked for item in parsed_observations if item.locked is not None]
    metadata = {
        "platform": "Ubuntu 20.04 + ROS1 Noetic",
        "input_bag": str(args.bag),
        "human_topic": args.human_topic,
        "human_message_type": detected_type,
        "source_coordinate_frame": args.source_frame,
        "output_coordinate_frame": args.coordinate_frame,
        "timestamp_policy": args.timestamp_policy,
        "timestamp_source_counts": dict(timestamp_counts),
        "timestamp_source": next(iter(timestamp_counts)) if len(timestamp_counts) == 1 else "mixed",
        "header_stamp_semantics": args.header_stamp_semantics,
        "receipt_timestamp_note": (
            "Offline raw HumanTarget without Header uses rosbag record time. Receipt time is only "
            "available if the online adapter encoded it into the published PoseStamped header."
        ),
        "input_message_count": len(human_samples),
        "invalid_human_target_count": invalid_count,
        "output_record_count": len(derived),
        "confidence": {
            "count": len(confidence_values),
            "min": min(confidence_values) if confidence_values else None,
            "max": max(confidence_values) if confidence_values else None,
        },
        "locked_true_count": sum(bool(value) for value in locked_values),
        "parsed_human_target_fields": [
            "track_id", "position.x", "position.y", "position.z", "distance", "bearing",
            "confidence", "locked", "is_valid",
        ],
    }
    return derived, metadata


def main() -> int:
    args = parse_args()
    try:
        if not args.source_frame.strip():
            raise ValueError("--source-frame 不能为空")
        if args.max_sync_gap < 0:
            raise ValueError("--max-sync-gap 不能为负")
        protect_input_bag(args)
        topic_types, samples = read_ros1_bag(args)
        records, metadata = convert_samples(args, topic_types, samples)
        from src.data.ros_trajectory_loader import write_trajectory_csv

        write_trajectory_csv(args.output, records)
        metadata_path = _metadata_path(args)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"已只读转换 {len(records)} 条 ROS1 记录：{args.output}")
    print(f"metadata：{_metadata_path(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
