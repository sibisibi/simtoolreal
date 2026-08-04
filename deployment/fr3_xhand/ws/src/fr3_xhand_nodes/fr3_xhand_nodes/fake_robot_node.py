"""rclpy port of deployment/fake/fake_robot_node.py.

Fakes the arm and hand controllers, interpolates toward the latest joint
targets at a per group rate limit and publishes joint states. Joint names and
the initial pose come from the contract instead of hardcoded 7/22 DoF arrays.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from .contract import Contract

CONTROL_HZ = 60.0
MAX_DELTA = 0.1


class FakeRobotNode(Node):
    def __init__(self) -> None:
        super().__init__("fake_robot_node")
        contract = Contract()
        self.num_arm = contract.num_arm
        self.arm_names = contract.joint_order[: self.num_arm]
        self.hand_names = contract.joint_order[self.num_arm :]
        self.dt = 1.0 / CONTROL_HZ

        self.q = contract.default_joint_pos.copy()
        self.qd = np.zeros(contract.num_joints)
        self.arm_target: np.ndarray | None = None
        self.hand_target: np.ndarray | None = None

        qos = QoSProfile(depth=1)
        self.create_subscription(
            JointState, "/fr3/joint_target", self._on_arm_target, qos
        )
        self.create_subscription(
            JointState, "/xhand/joint_target", self._on_hand_target, qos
        )
        self.arm_pub = self.create_publisher(JointState, "/fr3/joint_states", qos)
        self.hand_pub = self.create_publisher(JointState, "/xhand/joint_states", qos)
        self.create_timer(self.dt, self._tick)

    def _on_arm_target(self, msg: JointState) -> None:
        assert list(msg.name) == self.arm_names, (
            f"joint_target names {list(msg.name)} do not match arm order {self.arm_names}"
        )
        self.arm_target = np.array(msg.position)

    def _on_hand_target(self, msg: JointState) -> None:
        assert list(msg.name) == self.hand_names, (
            f"joint_target names {list(msg.name)} do not match hand order {self.hand_names}"
        )
        self.hand_target = np.array(msg.position)

    def _tick(self) -> None:
        if self.arm_target is not None and self.hand_target is not None:
            delta_arm = self.arm_target - self.q[: self.num_arm]
            delta_hand = self.hand_target - self.q[self.num_arm :]
            norm_arm = np.linalg.norm(delta_arm)
            norm_hand = np.linalg.norm(delta_hand)
            if norm_arm > MAX_DELTA:
                delta_arm = MAX_DELTA * delta_arm / norm_arm
            if norm_hand > MAX_DELTA:
                delta_hand = MAX_DELTA * delta_hand / norm_hand
            self.q[: self.num_arm] += delta_arm
            self.q[self.num_arm :] += delta_hand
            self.qd[: self.num_arm] = delta_arm / self.dt
            self.qd[self.num_arm :] = delta_hand / self.dt
        else:
            self.get_logger().warn(
                "waiting for joint targets", throttle_duration_sec=1.0
            )
        self._publish()

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()

        arm_msg = JointState()
        arm_msg.header.stamp = stamp
        arm_msg.name = self.arm_names
        arm_msg.position = self.q[: self.num_arm].tolist()
        arm_msg.velocity = self.qd[: self.num_arm].tolist()
        self.arm_pub.publish(arm_msg)

        hand_msg = JointState()
        hand_msg.header.stamp = stamp
        hand_msg.name = self.hand_names
        hand_msg.position = self.q[self.num_arm :].tolist()
        hand_msg.velocity = self.qd[self.num_arm :].tolist()
        self.hand_pub.publish(hand_msg)


def main() -> None:
    rclpy.init()
    rclpy.spin(FakeRobotNode())


if __name__ == "__main__":
    main()
