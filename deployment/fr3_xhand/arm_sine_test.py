"""Free-air joint sines for the arm smoke test, one joint at a time.

The first phase publishes the measured pose, so the target path is exercised
with nothing commanded to move. Only then does each joint take a turn, with a
raised cosine envelope so every sweep starts and ends at zero displacement.

Targets go out at the policy's 60 Hz on /fr3/joint_target, the same topic and
message the policy node uses, so this tests the real path rather than a stand
in. Names must be fr3_joint1 to fr3_joint7 or the controller faults and stops.

Run with the arm already up, `real.launch.py arm_only:=true`, which is also
recording the bag this reads back.

    .venv_deploy/bin/python3 deployment/fr3_xhand/arm_sine_test.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "deployment/fr3_xhand/contract/a4h1.json"
RATE_HZ = 60.0
# The controller brakes on targets older than 200 ms, so a dropped cycle is
# safe, but a loop that cannot hold 60 Hz is not worth trusting on hardware.
MAX_PERIOD_S = 0.05
LIMIT_MARGIN_RAD = 0.10


class SineTest(Node):
    def __init__(self, args) -> None:
        super().__init__("arm_sine_test")
        contract = json.loads(CONTRACT_PATH.read_text())
        self.n = int(contract["num_arm_joints"])
        self.names = [f"fr3_joint{i}" for i in range(1, self.n + 1)]
        self.lower = contract["joint_lower"][: self.n]
        self.upper = contract["joint_upper"][: self.n]
        self.args = args

        self.measured: dict[str, float] | None = None
        self.create_subscription(JointState, "/joint_states", self._on_state, 10)
        self.pub = self.create_publisher(JointState, "/fr3/joint_target", 1)

    def _on_state(self, msg: JointState) -> None:
        self.measured = dict(zip(msg.name, msg.position))

    def wait_for_state(self, timeout_s: float = 10.0) -> list[float]:
        end = time.time() + timeout_s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.measured and all(n in self.measured for n in self.names):
                return [self.measured[n] for n in self.names]
        raise RuntimeError("no /joint_states carrying the arm joints")

    def publish(self, q: list[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = q
        self.pub.publish(msg)

    def hold(self, q0: list[float], seconds: float) -> float:
        return self._drive(lambda t: list(q0), seconds)

    def sweep(self, q0: list[float], joint: int, seconds: float) -> float:
        amp, freq = self.args.amplitude, self.args.frequency
        lo = self.lower[joint] + LIMIT_MARGIN_RAD
        hi = self.upper[joint] - LIMIT_MARGIN_RAD
        assert lo <= q0[joint] <= hi, (
            f"{self.names[joint]} starts at {q0[joint]:.3f}, outside "
            f"[{lo:.3f}, {hi:.3f}] once the margin is applied"
        )
        reach = min(amp, hi - q0[joint], q0[joint] - lo)
        if reach < amp:
            self.get_logger().warn(
                f"{self.names[joint]} limited to {reach:.3f} rad by its joint limits"
            )

        def target(t: float) -> list[float]:
            # Raised cosine envelope, zero displacement at both ends.
            envelope = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / seconds))
            q = list(q0)
            q[joint] = q0[joint] + reach * envelope * math.sin(2.0 * math.pi * freq * t)
            return q

        return self._drive(target, seconds)

    def _drive(self, target_fn, seconds: float) -> float:
        period = 1.0 / RATE_HZ
        start = time.perf_counter()
        worst = 0.0
        next_tick = start
        while True:
            now = time.perf_counter()
            t = now - start
            if t >= seconds:
                return worst
            self.publish(target_fn(t))
            rclpy.spin_once(self, timeout_sec=0.0)
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                worst = max(worst, -sleep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplitude", type=float, default=0.05, help="rad, peak")
    parser.add_argument("--frequency", type=float, default=0.25, help="Hz")
    parser.add_argument("--seconds_per_joint", type=float, default=8.0)
    parser.add_argument("--hold_seconds", type=float, default=3.0)
    parser.add_argument("--joints", default="", help="comma separated 1-based, default all")
    args = parser.parse_args()

    rclpy.init()
    node = SineTest(args)
    q0 = node.wait_for_state()
    node.get_logger().info(f"start pose rad {[round(v, 4) for v in q0]}")

    lag = node.hold(q0, args.hold_seconds)
    node.get_logger().info(f"held the measured pose for {args.hold_seconds:.0f} s, worst lag {lag*1e3:.1f} ms")

    which = (
        [int(v) - 1 for v in args.joints.split(",")] if args.joints
        else list(range(node.n))
    )
    for j in which:
        lag = node.sweep(q0, j, args.seconds_per_joint)
        node.get_logger().info(f"{node.names[j]} swept, worst lag {lag*1e3:.1f} ms")

    lag = node.hold(q0, args.hold_seconds)
    node.get_logger().info(f"back at the start pose, worst lag {lag*1e3:.1f} ms")
    assert lag < MAX_PERIOD_S, f"loop fell behind by {lag*1e3:.1f} ms"
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
