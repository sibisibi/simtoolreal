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
joint layout and count in this project's fr3_xhand.urdf.

Confirmed against LeFranX, which drives this hand. Its per id joint limits in
xhand_config.py match our URDF position by position, and id 3 spans plus or
minus 10 degrees where every other id spans about 110, so the single abduction
joint pins the ordering. Its live limits raise four mins from 0 to 5 degrees
against mechanical clogging, and its commented original matches ours exactly.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
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

# One RS485 exchange carries all 12 joints and all 5 sensors and takes 12 ms,
# so the hand tops out at 83 Hz. User_Manual_v1.3 says so, and the Python, C++,
# ROS1 and ROS2 API references repeat it. Polling faster asks for an exchange
# the bus cannot deliver, which shows up as a CRC error rather than a refusal.
# 60 Hz matches the policy's own rate and leaves margin under the ceiling.
#
# That ceiling counts every blocking exchange, not just reads. send_command is
# one, and read_state with force_update is another, so doing both every cycle
# is 24 ms of bus inside a 16.7 ms period and the bus falls over. The vendor's
# own RS485 test sends a command and then reads with force_update false, and
# their example says to force the update only when send_command is not used.
# send_command refreshes the device state on its way through, so the forced
# read is only needed while nothing is commanding.
HAND_MAX_RATE_HZ = 83.0
STATE_RATE_HZ = 60.0
assert STATE_RATE_HZ <= HAND_MAX_RATE_HZ, "state rate exceeds the hand's bus cycle"

# How long to keep asking the hand for one clean read before giving up at
# startup. A patience bound, chosen, not a vendor figure.
READY_TIMEOUT_S = 10.0

# "Current exceeds the set threshold, triggering overcurrent warning
# (exceeding 500ms)". The hand raises this when a finger has been stalled for
# half a second, which is what closing on a rigid object looks like, and it is
# advisory. Measured by holding every finger stalled at 90 percent of range for
# eight seconds with the gains this node sends: of the 92 cycles after the
# first warning, send_command returned success on 90 and read_state on all 92,
# the hand tracked throughout, and it opened again on 148 of 148 commands.
# Treating it as fatal ended a rollout mid grasp. Every other non-zero code
# still stops the node.
OVERCURRENT_CODE = 1501035
OVERCURRENT_LOG_EVERY_S = 1.0

# A single corrupt exchange is not a failure. The bus produces them
# intermittently, the startup wait already showed one clearing on the next
# attempt, and the policy publishes a fresh target 16.7 ms later regardless.
# What actually matters is going quiet for longer than the consumer tolerates,
# and that bound already exists, policy_node stops when hand state ages past
# STALE_S. Failing at the same bound means the hand is never the thing that
# leaves the policy working from a stale pose, and no new threshold is invented
# to sit alongside one that is already contracted.
COMMS_TIMEOUT_S = 0.25


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
        self.commanded_since_read = False
        self.overcurrent_count = 0
        self.overcurrent_last_log = 0.0
        self.bad_exchange_count = 0
        self.last_good_exchange = time.monotonic()

        # read_state already carries the tactile sensors in the same exchange the
        # joints ride on, so publishing them costs no extra bus time. The shape
        # is whatever the hand reports on the first read, fixed after that. This
        # hand answers with 5 sensors of 120 taxels and 20 temperatures each.
        self.shape: tuple[int, int, int] | None = None

        self._wait_until_ready()

        qos = QoSProfile(depth=1)
        self.create_subscription(JointState, "/xhand/joint_target", self._on_target, qos)
        self.state_pub = self.create_publisher(JointState, "/xhand/joint_states", qos)
        self.tactile_pub = self.create_publisher(
            Float64MultiArray, "/xhand/tactile", qos
        )
        self.create_timer(1.0 / STATE_RATE_HZ, self._publish_state)

    def _check(self, err, call: str) -> bool:
        """Say whether this exchange produced usable data, and stop if the hand
        has gone quiet for longer than the policy tolerates.

        Overcurrent is not a failed exchange. The hand answers normally through
        it and the state is good, so it counts as success and is reported
        separately. A rollout that spends its time stalled is worth knowing
        about even though nothing is broken.
        """
        now = time.monotonic()
        if err.error_code == 0 or err.error_code == OVERCURRENT_CODE:
            self.last_good_exchange = now
            if err.error_code == OVERCURRENT_CODE:
                self.overcurrent_count += 1
                if now - self.overcurrent_last_log >= OVERCURRENT_LOG_EVERY_S:
                    self.overcurrent_last_log = now
                    self.get_logger().warn(
                        f"hand overcurrent, a finger is stalled, "
                        f"{self.overcurrent_count} so far this run"
                    )
            return True

        self.bad_exchange_count += 1
        quiet = now - self.last_good_exchange
        if quiet > COMMS_TIMEOUT_S:
            raise AssertionError(
                f"{call} failed and the hand has been quiet {quiet:.3f} s, "
                f"past the {COMMS_TIMEOUT_S} s the policy tolerates, "
                f"last error {err.error_message!r}"
            )
        self.get_logger().warn(
            f"{call} returned {err.error_message!r}, skipping this cycle, "
            f"{self.bad_exchange_count} so far this run"
        )
        return False

    def _wait_until_ready(self) -> None:
        """Poll until the hand answers one clean read, then let the timer start.

        The first read after open_serial sometimes comes back a CRC error even
        though the hand is healthy, and the device probes 300/300 at 60 Hz a
        moment later. Starting the 60 Hz timer into that window kills the node
        on its first tick and takes the run with it, which happened three times
        in one session.

        READY_TIMEOUT_S is a patience bound, not a figure from the vendor. It
        is long enough that a hand which is merely still waking up gets there,
        and short enough that a hand which is genuinely absent fails while
        someone is still watching. A hand that never answers still raises.
        """
        deadline = time.monotonic() + READY_TIMEOUT_S
        attempts, last = 0, ""
        while time.monotonic() < deadline:
            err, _ = self.device.read_state(self.hand_id, True)
            attempts += 1
            if err.error_code == 0:
                if attempts > 1:
                    self.get_logger().warn(
                        f"hand answered on attempt {attempts}, last error {last!r}"
                    )
                return
            last = err.error_message
            time.sleep(1.0 / STATE_RATE_HZ)
        raise AssertionError(
            f"hand never answered a clean read in {READY_TIMEOUT_S} s, "
            f"{attempts} attempts, last error {last!r}"
        )

    def _on_target(self, msg: JointState) -> None:
        assert list(msg.name) == self.hand_names, (
            f"joint_target names {list(msg.name)} do not match hand order {self.hand_names}"
        )
        q = np.array(msg.position)
        for hand_idx, sdk_id in enumerate(self.sdk_id_of_hand_idx):
            self.command.finger_command[sdk_id].position = float(q[hand_idx])
        err = self.device.send_command(self.hand_id, self.command)
        # A command that did not land leaves the device state unrefreshed, so
        # the next read has to force its own update rather than trust a cache
        # that nothing wrote.
        self.commanded_since_read = self._check(err, "send_command")

    def _publish_state(self) -> None:
        # Force a refresh only when no command has been through since the last
        # read, otherwise send_command has already brought the state up to date
        # and a second exchange overruns the bus.
        force_update = not self.commanded_since_read
        err, state = self.device.read_state(self.hand_id, force_update)
        self.commanded_since_read = False
        if not self._check(err, "read_state"):
            # The payload may be corrupt, so publish nothing rather than a pose
            # the hand never reported. The consumer's staleness check covers a
            # gap this short, and _check raises if the gaps stop being short.
            return
        sdk_positions = np.array(
            [state.finger_state[sdk_id].position for sdk_id in range(len(SDK_JOINT_ORDER))]
        )
        # The SDK reports torque as an integer, and JointState.effort is float64.
        sdk_torques = np.array(
            [state.finger_state[sdk_id].torque for sdk_id in range(len(SDK_JOINT_ORDER))],
            dtype=float,
        )
        q = sdk_positions[self.sdk_id_of_hand_idx]
        tau = sdk_torques[self.sdk_id_of_hand_idx]

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_q is not None:
            qd = (q - self.prev_q) / (now - self.prev_t)
        else:
            qd = np.zeros_like(q)
        self.prev_q = q
        self.prev_t = now

        stamp = self.get_clock().now().to_msg()
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = self.hand_names
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        msg.effort = tau.tolist()
        self.state_pub.publish(msg)
        self._publish_tactile(state)

    def _publish_tactile(self, state) -> None:
        """One row per fingertip sensor, in the sensor_data order the SDK returns.

        The vendor example names {2, 5, 7, 9, 11} as the fingertip sensor ids,
        the distal joint of each finger, so the rows run thumb to pinky in the
        same finger order SDK_JOINT_ORDER uses.

        Each row is the resolved force, the two temperatures, then the raw taxel
        grid flattened. The layout carries the shape so a bag reader does not
        have to know the taxel count in advance.
        """
        sensors = state.sensor_data
        rows = []
        for sensor in sensors:
            if self.shape is None:
                self.shape = (len(sensors), len(sensor.raw_force), len(sensor.temperature))
                self.get_logger().info(
                    f"tactile, {self.shape[0]} sensors, {self.shape[1]} taxels, "
                    f"{self.shape[2]} temperatures each"
                )
            assert (len(sensors), len(sensor.raw_force), len(sensor.temperature)) == self.shape, (
                f"tactile shape changed from {self.shape}"
            )
            # Every field is cast. calc_force, calc_temperature and the taxels
            # all come back as integers, and Float64MultiArray takes float64.
            rows.append(
                [
                    float(sensor.calc_force.fx),
                    float(sensor.calc_force.fy),
                    float(sensor.calc_force.fz),
                    float(sensor.calc_temperature),
                ]
                + [float(t) for t in sensor.temperature]
                + [float(v) for f in sensor.raw_force for v in (f.fx, f.fy, f.fz)]
            )

        num_sensors, num_taxels, num_temps = self.shape
        stride = 4 + num_temps + 3 * num_taxels
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label="sensor", size=num_sensors, stride=num_sensors * stride
            ),
            MultiArrayDimension(
                label=f"calc_fxyz,calc_temp,temp[{num_temps}],raw_fxyz[{num_taxels}]",
                size=stride,
                stride=stride,
            ),
        ]
        msg.data = [v for row in rows for v in row]
        self.tactile_pub.publish(msg)


def main() -> None:
    rclpy.init()
    rclpy.spin(HandNode())


if __name__ == "__main__":
    main()
