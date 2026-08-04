"""rclpy port of deployment/goal_pose_node.py.

Publishes the current keypoint tracking goal for the object, advancing
through the object's trajectory once the tracked object pose stays within
threshold of the current goal for success_steps consecutive ticks. Unlike the
ROS1 node this resets the streak on a miss, so consecutive means consecutive.
Keypoint offsets come from the object spec's object_scales and the contract's
keypoint_corners and keypoint_factor, not hardcoded per-object numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile

from .contract import Contract
from .object_spec import load_object_spec

KEYPOINT_SUCCESS_SCALE = 1.5
GOAL_RATE_HZ = 60.0


def _quat_rotate(quat_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * (q_vec @ v) * 2.0
    return a + b + c


def _keypoint_positions(pose_xyz_xyzw: np.ndarray, kp_offsets: np.ndarray) -> np.ndarray:
    pos = pose_xyz_xyzw[:3]
    quat = pose_xyz_xyzw[3:7]
    return np.array([pos + _quat_rotate(quat, offset) for offset in kp_offsets])


def keypoint_distance(
    pose1_xyzw: np.ndarray, pose2_xyzw: np.ndarray, kp_offsets: np.ndarray
) -> float:
    kp1 = _keypoint_positions(pose1_xyzw, kp_offsets)
    kp2 = _keypoint_positions(pose2_xyzw, kp_offsets)
    return float(np.linalg.norm(kp1 - kp2, axis=-1).max())


def _pose_to_xyz_xyzw(msg: PoseStamped) -> np.ndarray:
    p, o = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])


class GoalNode(Node):
    def __init__(self) -> None:
        super().__init__("goal_node")

        self.declare_parameter("object_spec", "")
        spec_path = self.get_parameter("object_spec").value
        assert spec_path, "object_spec parameter is required"
        spec = load_object_spec(spec_path)
        object_scales = np.array(spec["object_scales"], dtype=np.float64)

        contract = Contract()
        self.kp_offsets = contract.keypoint_corners * (
            object_scales * contract.keypoint_factor
        )

        self.declare_parameter("success_threshold", 0.02)
        self.declare_parameter("success_steps", 1)
        success_threshold = self.get_parameter("success_threshold").value
        self.success_steps = self.get_parameter("success_steps").value
        self.keypoint_success_threshold = success_threshold * KEYPOINT_SUCCESS_SCALE
        self.current_success_steps = 0

        # goal_trajectory is a dextoolbench-format trajectory JSON. The
        # pipeline that writes it owns the frame, so poses are published as
        # loaded with no world-to-robot-frame conversion here.
        traj = json.loads(Path(spec["goal_trajectory"]).read_text())
        goals = np.array(traj["goals"], dtype=np.float64)
        assert goals.ndim == 2 and goals.shape[1] == 7, (
            f"goals.shape: {goals.shape}, expected (N, 7)"
        )
        self.goals = goals
        self.goal_index = 0
        self.current_object_pose: np.ndarray | None = None

        qos = QoSProfile(depth=1)
        self.goal_pub = self.create_publisher(
            PoseStamped, "/robot_frame/goal_object_pose", qos
        )
        self.create_subscription(
            PoseStamped, "/robot_frame/current_object_pose", self._on_object_pose, qos
        )
        self.create_timer(1.0 / GOAL_RATE_HZ, self._tick)

    def _on_object_pose(self, msg: PoseStamped) -> None:
        self.current_object_pose = _pose_to_xyz_xyzw(msg)

    def _tick(self) -> None:
        if self.current_object_pose is None:
            self.get_logger().warn(
                "waiting for current object pose", throttle_duration_sec=1.0
            )
            return
        self._advance_goal()
        self._publish_goal()

    def _advance_goal(self) -> None:
        num_goals = self.goals.shape[0]
        if self.goal_index >= num_goals:
            return
        distance = keypoint_distance(
            self.current_object_pose, self.goals[self.goal_index], self.kp_offsets
        )
        if distance < self.keypoint_success_threshold:
            self.current_success_steps += 1
            if self.current_success_steps >= self.success_steps:
                self.current_success_steps = 0
                self.goal_index += 1
                self.get_logger().info(
                    f"goal reached, advancing to {self.goal_index}/{num_goals}"
                )
        else:
            self.current_success_steps = 0

    def _publish_goal(self) -> None:
        idx = min(self.goal_index, self.goals.shape[0] - 1)
        goal = self.goals[idx]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "robot_frame"
        msg.pose.position.x = goal[0]
        msg.pose.position.y = goal[1]
        msg.pose.position.z = goal[2]
        msg.pose.orientation.x = goal[3]
        msg.pose.orientation.y = goal[4]
        msg.pose.orientation.z = goal[5]
        msg.pose.orientation.w = goal[6]
        self.goal_pub.publish(msg)


def main() -> None:
    rclpy.init()
    rclpy.spin(GoalNode())


if __name__ == "__main__":
    main()
