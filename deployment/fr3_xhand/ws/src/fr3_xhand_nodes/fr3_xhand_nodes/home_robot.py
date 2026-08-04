"""rclpy port of deployment/home_robot.py.

Reads the current arm and hand joint states and linearly interpolates both to
the contract's default pose over MOVE_SECONDS, publishing joint targets at
CONTROL_HZ, then exits.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from .contract import Contract

MOVE_SECONDS = 10.0
CONTROL_HZ = 60.0


class HomeRobotNode(Node):
    def __init__(self) -> None:
        super().__init__("home_robot")
        contract = Contract()
        self.num_arm = contract.num_arm
        self.arm_names = contract.joint_order[: self.num_arm]
        self.hand_names = contract.joint_order[self.num_arm :]
        self.target_pos = contract.default_joint_pos

        self.arm_q: np.ndarray | None = None
        self.hand_q: np.ndarray | None = None

        qos = QoSProfile(depth=1)
        self.create_subscription(JointState, "/fr3/joint_states", self._on_arm_state, qos)
        self.create_subscription(
            JointState, "/xhand/joint_states", self._on_hand_state, qos
        )
        self.arm_pub = self.create_publisher(JointState, "/fr3/joint_target", qos)
        self.hand_pub = self.create_publisher(JointState, "/xhand/joint_target", qos)

    def _on_arm_state(self, msg: JointState) -> None:
        self.arm_q = np.array(msg.position)

    def _on_hand_state(self, msg: JointState) -> None:
        self.hand_q = np.array(msg.position)

    def ready(self) -> bool:
        return self.arm_q is not None and self.hand_q is not None

    def run(self) -> None:
        num_steps = int(CONTROL_HZ * MOVE_SECONDS)
        dt = 1.0 / CONTROL_HZ
        start_pos = np.concatenate([self.arm_q, self.hand_q])
        for i in range(num_steps):
            if not rclpy.ok():
                sys.exit(0)
            start_time = time.time()
            alpha = (i + 1) / num_steps
            pos = start_pos + (self.target_pos - start_pos) * alpha
            self._publish_targets(pos)
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.get_logger().warn(
                    f"loop too slow, desired dt {dt:.4f} s, actual {elapsed:.4f} s"
                )

    def _publish_targets(self, pos: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()

        arm_msg = JointState()
        arm_msg.header.stamp = stamp
        arm_msg.name = self.arm_names
        arm_msg.position = pos[: self.num_arm].tolist()
        self.arm_pub.publish(arm_msg)

        hand_msg = JointState()
        hand_msg.header.stamp = stamp
        hand_msg.name = self.hand_names
        hand_msg.position = pos[self.num_arm :].tolist()
        self.hand_pub.publish(hand_msg)


def main() -> None:
    rclpy.init()
    node = HomeRobotNode()
    while not node.ready():
        node.get_logger().warn(
            "waiting for current joint states", throttle_duration_sec=1.0
        )
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info("moving to home pose")
    node.run()
    node.get_logger().info("reached home pose")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
