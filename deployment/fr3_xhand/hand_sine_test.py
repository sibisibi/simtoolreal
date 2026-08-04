"""Free-air flexion sweeps for the hand smoke test, one joint at a time.

The first phase publishes the measured pose, so the target path is exercised
with nothing commanded to move. Each joint then flexes and returns under a
raised cosine, which never goes below the starting angle.

One sided on purpose. The fingers rest near their zero bound, a symmetric sine
would ride into it, and LeFranX carries a note about mechanical clogging at
that end. Our limits match the vendor and sim and stay unchanged, this test
simply has no reason to visit the bound.

Targets go out at 60 Hz on /xhand/joint_target, the same topic and message the
policy node uses. The hand tops out at 83 Hz for a whole hand exchange.

    .venv_deploy/bin/python3 deployment/fr3_xhand/hand_sine_test.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "deployment/fr3_xhand/contract/a4h1.json"
RATE_HZ = 60.0
LIMIT_MARGIN_RAD = 0.05


class HandSineTest(Node):
    def __init__(self, args) -> None:
        super().__init__("hand_sine_test")
        contract = json.loads(CONTRACT_PATH.read_text())
        n_arm = int(contract["num_arm_joints"])
        self.names = contract["joint_order"][n_arm:]
        self.lower = np.array(contract["joint_lower"][n_arm:])
        self.upper = np.array(contract["joint_upper"][n_arm:])
        self.args = args

        self.measured: dict[str, float] | None = None
        self.create_subscription(JointState, "/xhand/joint_states", self._on_state, 10)
        self.pub = self.create_publisher(JointState, "/xhand/joint_target", 1)

    def _on_state(self, msg: JointState) -> None:
        self.measured = dict(zip(msg.name, msg.position))

    def wait_for_state(self, timeout_s: float = 10.0) -> np.ndarray:
        end = time.time() + timeout_s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.measured and all(n in self.measured for n in self.names):
                return np.array([self.measured[n] for n in self.names])
        raise RuntimeError("no /xhand/joint_states, is hand_node running")

    def publish(self, q: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = q.tolist()
        self.pub.publish(msg)

    def hold(self, q0: np.ndarray, seconds: float) -> float:
        return self._drive(lambda t: q0.copy(), seconds)

    def flex(self, q0: np.ndarray, joint: int, seconds: float) -> float:
        headroom = self.upper[joint] - LIMIT_MARGIN_RAD - q0[joint]
        reach = min(self.args.amplitude, headroom)
        assert reach > 0.02, (
            f"{self.names[joint]} starts at {q0[joint]:.3f} with only "
            f"{headroom:.3f} rad of headroom below its upper limit"
        )
        if reach < self.args.amplitude:
            self.get_logger().warn(f"{self.names[joint]} limited to {reach:.3f} rad")

        def target(t: float) -> np.ndarray:
            # One raised cosine per cycle, so the flexion reaches the full reach
            # every cycle and returns to the start. Multiplying a wave by a
            # separate envelope put the envelope's peak on a wave zero, which
            # capped the motion near 60 percent of what was asked for.
            wave = 0.5 * (1.0 - math.cos(2.0 * math.pi * self.args.frequency * t))
            q = q0.copy()
            q[joint] = q0[joint] + reach * wave
            return q

        return self._drive(target, seconds)

    def _drive(self, target_fn, seconds: float) -> float:
        period = 1.0 / RATE_HZ
        start = time.perf_counter()
        worst, next_tick = 0.0, start
        while True:
            t = time.perf_counter() - start
            if t >= seconds:
                return worst
            q = np.clip(target_fn(t), self.lower, self.upper)
            self.publish(q)
            rclpy.spin_once(self, timeout_sec=0.0)
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                worst = max(worst, -sleep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplitude", type=float, default=0.30, help="rad of flexion")
    parser.add_argument("--frequency", type=float, default=0.25, help="Hz")
    parser.add_argument("--seconds_per_joint", type=float, default=8.0)
    parser.add_argument("--hold_seconds", type=float, default=3.0)
    parser.add_argument("--joints", default="", help="comma separated 1-based, default all")
    args = parser.parse_args()

    rclpy.init()
    node = HandSineTest(args)
    q0 = node.wait_for_state()
    node.get_logger().info(f"start pose rad {np.round(q0, 4).tolist()}")

    lag = node.hold(q0, args.hold_seconds)
    node.get_logger().info(f"held the measured pose, worst lag {lag*1e3:.1f} ms")

    which = (
        [int(v) - 1 for v in args.joints.split(",")] if args.joints
        else list(range(len(node.names)))
    )
    for j in which:
        lag = node.flex(q0, j, args.seconds_per_joint)
        node.get_logger().info(f"{node.names[j]} flexed, worst lag {lag*1e3:.1f} ms")

    node.hold(q0, args.hold_seconds)
    node.get_logger().info("back at the start pose")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
