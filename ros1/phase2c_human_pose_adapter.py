#!/usr/bin/env python3
"""ROS1 Noetic read-only adapter from follow_msgs/HumanTarget to world-frame pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AdapterStatistics:
    received: int = 0
    published: int = 0
    invalid_target: int = 0
    missing_position: int = 0
    invalid_track_id: int = 0
    source_frame_mismatch: int = 0
    tf_lookup_failure: int = 0
    tf_connectivity_failure: int = 0
    tf_extrapolation_failure: int = 0
    header_timestamp: int = 0
    receipt_timestamp: int = 0

    def summary(self) -> str:
        return " ".join(f"{key}={value}" for key, value in vars(self).items())


def choose_timestamp(message: Any, receipt_time: Any) -> tuple[Any, str]:
    """Use a real nonzero header stamp, otherwise explicitly use callback receipt time."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is not None:
        seconds = stamp.to_sec() if hasattr(stamp, "to_sec") else float(stamp)
        if seconds > 0:
            return stamp, "header_stamp"
    return receipt_time, "receipt_timestamp"


def main() -> int:
    try:
        import rospy
        import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped conversion.
        import tf2_ros
        from follow_msgs.msg import HumanTarget
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Int64
    except ImportError as exc:
        print(
            "ROS1 运行环境不可用。请在 Ubuntu 20.04/ROS Noetic 上 source /opt/ros/noetic/setup.bash "
            "和 ~/gstest/devel/setup.bash；本节点不会安装依赖。"
        )
        print(f"缺失模块：{exc}")
        return 1

    rospy.init_node("phase2c_human_pose_adapter")
    source_frame = str(rospy.get_param("~source_frame", "")).strip()
    target_frame = str(rospy.get_param("~target_frame", "map")).strip()
    input_topic = str(rospy.get_param("~input_topic", "/human/target"))
    pose_output_topic = str(
        rospy.get_param("~pose_output_topic", "/wm/human_center_pose")
    )
    track_output_topic = str(
        rospy.get_param("~track_output_topic", "/wm/human_track_id")
    )
    tf_timeout_seconds = float(rospy.get_param("~tf_timeout", 0.1))
    log_every = int(rospy.get_param("~statistics_log_every", 100))
    if not source_frame:
        rospy.logfatal("~source_frame 必须明确设置；禁止猜测 HumanTarget 坐标系")
        return 2
    if not target_frame:
        rospy.logfatal("~target_frame 不能为空")
        return 2

    statistics = AdapterStatistics()
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
    _listener = tf2_ros.TransformListener(tf_buffer)
    pose_publisher = rospy.Publisher(pose_output_topic, PoseStamped, queue_size=20)
    track_publisher = rospy.Publisher(track_output_topic, Int64, queue_size=20)

    def log_statistics() -> None:
        rospy.loginfo("Phase2C adapter statistics: %s", statistics.summary())

    def callback(message: Any) -> None:
        statistics.received += 1
        if hasattr(message, "is_valid") and not bool(message.is_valid):
            statistics.invalid_target += 1
            if log_every > 0 and statistics.received % log_every == 0:
                log_statistics()
            return
        position = getattr(message, "position", None)
        if position is None:
            statistics.missing_position += 1
            rospy.logwarn_throttle(5.0, "HumanTarget 缺少 position；该帧未发布")
            return
        header = getattr(message, "header", None)
        input_frame = str(getattr(header, "frame_id", "")).strip() if header else ""
        if input_frame and input_frame != source_frame:
            statistics.source_frame_mismatch += 1
            rospy.logwarn_throttle(
                5.0,
                "HumanTarget header frame 与 ~source_frame 不一致；该帧未发布",
            )
            return
        try:
            track_id = int(message.track_id)
        except (AttributeError, TypeError, ValueError, OverflowError):
            statistics.invalid_track_id += 1
            rospy.logwarn_throttle(5.0, "HumanTarget track_id 无法转换为 Int64；该帧未发布")
            return

        receipt_time = rospy.Time.now()
        stamp, timestamp_source = choose_timestamp(message, receipt_time)
        if timestamp_source == "header_stamp":
            statistics.header_timestamp += 1
        else:
            statistics.receipt_timestamp += 1
        source_pose = PoseStamped()
        source_pose.header.stamp = stamp
        source_pose.header.frame_id = source_frame
        source_pose.pose.position.x = float(position.x)
        source_pose.pose.position.y = float(position.y)
        source_pose.pose.position.z = float(position.z)
        source_pose.pose.orientation.w = 1.0
        try:
            transformed = tf_buffer.transform(
                source_pose, target_frame, rospy.Duration(tf_timeout_seconds)
            )
        except tf2_ros.LookupException as exc:
            statistics.tf_lookup_failure += 1
            rospy.logwarn_throttle(5.0, "TF lookup 失败，丢弃该帧：%s", str(exc))
            return
        except tf2_ros.ConnectivityException as exc:
            statistics.tf_connectivity_failure += 1
            rospy.logwarn_throttle(5.0, "TF connectivity 失败，丢弃该帧：%s", str(exc))
            return
        except tf2_ros.ExtrapolationException as exc:
            statistics.tf_extrapolation_failure += 1
            rospy.logwarn_throttle(5.0, "TF extrapolation 失败，丢弃该帧：%s", str(exc))
            return
        transformed.header.stamp = stamp
        transformed.header.frame_id = target_frame
        pose_publisher.publish(transformed)
        track_publisher.publish(Int64(data=track_id))
        statistics.published += 1
        if log_every > 0 and statistics.received % log_every == 0:
            log_statistics()

    rospy.Subscriber(input_topic, HumanTarget, callback, queue_size=20)
    rospy.on_shutdown(log_statistics)
    rospy.loginfo(
        "Phase2C read-only adapter: %s (%s) -> %s + %s (%s)",
        input_topic,
        source_frame,
        pose_output_topic,
        track_output_topic,
        target_frame,
    )
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
