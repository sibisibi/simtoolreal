"""RobotEra XHand1 driver node, backed by the vendor xhand_controller SDK.

Imported at module scope so a missing SDK crashes at startup instead of
falling back to a stub. Verified against xhand_controller 1.1.8
(cp310, wheel from xhand1_delivery_with_tactile/SDK/Python/xhand_control_sdk_py).

FingerState_t and FingerCommand_t identify a joint by a bare integer id, the
compiled SDK exposes no per joint name to query at runtime. SDK_JOINT_ORDER
below is inferred, not read from the device, from the vendor's own example
(xhand_control_example.py), which documents fingertip tactile sensor ids
{2, 5, 7, 9, 11}. Those ids only make sense as the last joint of each finger
if the 12 ids run thumb(3) index(3) middle(2) ring(2) pinky(2) in that order,
proximal to distal within each finger, which is exactly the actuated hand
joint layout and count in this project's fr3_xhand.urdf. This has not been
confirmed against a live hand, verify direction and zero of each joint before
first motion.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from xhand_controller import xhand_control

from .contract import Contract

SDK_JOINT_ORDER = [
    "thumb_joint0",
    "thumb_joint1",
    "thumb_joint2",
    "index_joint0",
    "index_joint1",
    "index_joint2",
    "middle_joint0",
    "middle_joint1",
    "ring_joint0",
    "ring_joint1",
    "pinky_joint0",
    "pinky_joint1",
]

# Matches the vendor example's own defaults, xhand_control_sdk_py/xhand_control_example.py.
COMMAND_KP = 100
COMMAND_KI = 0
COMMAND_KD = 0
COMMAND_TOR_MAX = 300
COMMAND_MODE_POSITION = 3

STATE_RATE_HZ = 100.0


class HandNode(Node):
    def __init__(self) -> None:
        super().__init__("hand_node")
        contract = Contract()
        self.hand_names = contract.joint_order[contract.num_arm :]
        assert set(self.hand_names) == set(SDK_JOINT_ORDER), (
            f"contract hand joints {sorted(self.hand_names)} do not match "
            f"the known xhand SDK joint order {sorted(SDK_JOINT_ORDER)}"
        )
        # sdk_id_of_hand_idx[i] is the SDK finger id for contract hand joint i.
        self.sdk_id_of_hand_idx = np.array(
            [SDK_JOINT_ORDER.index(name) for name in self.hand_names]
        )

        self.declare_parameter("device", "")
        self.declare_parameter("baud_rate", 3_000_000)
        device = self.get_parameter("device").value
        baud_rate = self.get_parameter("baud_rate").value
        assert device, "device parameter (serial port) is required"

        self.device = xhand_control.XHandControl()
        rsp = self.device.open_serial(device, baud_rate)
        assert rsp.error_code == 0, f"open_serial failed: {rsp.error_message}"
        hands = self.device.list_hands_id()
        assert hands, "no xhand devices enumerated after open_serial"
        self.hand_id = hands[0]

        self.command = xhand_control.HandCommand_t()
        for sdk_id in range(len(SDK_JOINT_ORDER)):
            fc = self.command.finger_command[sdk_id]
            fc.id = sdk_id
            fc.kp = COMMAND_KP
            fc.ki = COMMAND_KI
            fc.kd = COMMAND_KD
            fc.tor_max = COMMAND_TOR_MAX
            fc.mode = COMMAND_MODE_POSITION

        self.prev_q: np.ndarray | None = None
        self.prev_t: float | None = None

        qos = QoSProfile(depth=1)
        self.create_subscription(JointState, "/xhand/joint_target", self._on_target, qos)
        self.state_pub = self.create_publisher(JointState, "/xhand/joint_states", qos)
        self.create_timer(1.0 / STATE_RATE_HZ, self._publish_state)

    def _on_target(self, msg: JointState) -> None:
        assert list(msg.name) == self.hand_names, (
            f"joint_target names {list(msg.name)} do not match hand order {self.hand_names}"
        )
        q = np.array(msg.position)
        for hand_idx, sdk_id in enumerate(self.sdk_id_of_hand_idx):
            self.command.finger_command[sdk_id].position = float(q[hand_idx])
        err = self.device.send_command(self.hand_id, self.command)
        assert err.error_code == 0, f"send_command failed: {err.error_message}"

    def _publish_state(self) -> None:
        err, state = self.device.read_state(self.hand_id, True)
        assert err.error_code == 0, f"read_state failed: {err.error_message}"
        sdk_positions = np.array(
            [state.finger_state[sdk_id].position for sdk_id in range(len(SDK_JOINT_ORDER))]
        )
        q = sdk_positions[self.sdk_id_of_hand_idx]

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_q is not None:
            qd = (q - self.prev_q) / (now - self.prev_t)
        else:
            qd = np.zeros_like(q)
        self.prev_q = q
        self.prev_t = now

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.hand_names
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        self.state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    rclpy.spin(HandNode())


if __name__ == "__main__":
    main()
