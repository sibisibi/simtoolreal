"""60 Hz policy loop for fr3-xhand, rclpy port of deployment/rl_policy_node.py.

Ported semantics, wait for every input, warm up holding the measured pose
while the LSTM runs, reset the RNN, then run the policy. Dropped upstream
warts, breakpoint() on control paths, per-node object scales, ignored
timestamps. Any stale input or a target far from the measured arm raises,
the process dies, and the arm controller's own staleness watchdog holds.
"""

from __future__ import annotations

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState

from deployment.rl_player import RlPlayer

from .contract import Contract, Fk
from .obs_action import build_observation, compute_targets
from .object_spec import load_object_spec

STALE_S = 0.25  # trained object-pose delay tolerance is about 0.167 s
MAX_ARM_TARGET_DIFF_DEG = 10.0
WARMUP_STEPS = 100


def _stamp_to_s(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _pose_to_xyz_xyzw(msg: PoseStamped) -> np.ndarray:
    p, o = msg.pose.position, msg.pose.orientation
    return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        self.declare_parameter("object_spec", "")
        spec_path = self.get_parameter("object_spec").value
        assert spec_path, "object_spec parameter is required"
        spec = load_object_spec(spec_path)
        self.object_scales = np.array(spec["object_scales"], dtype=np.float64)

        self.contract = Contract()
        self.fk = Fk(self.contract)
        c = self.contract
        self.arm_names = c.joint_order[: c.num_arm]
        self.hand_names = c.joint_order[c.num_arm :]

        self.player = RlPlayer(
            num_observations=c.num_observations,
            num_actions=c.num_actions,
            config_path=str(c.train_yaml_path),
            checkpoint_path=str(c.checkpoint_path),
            device="cpu",
            num_envs=1,
        )

        qos = QoSProfile(depth=1)
        self._msgs: dict[str, object] = {}
        for topic in ("/fr3/joint_states", "/xhand/joint_states"):
            self.create_subscription(JointState, topic, self._store(topic), qos)
        for topic in ("/robot_frame/current_object_pose", "/robot_frame/goal_object_pose"):
            self.create_subscription(PoseStamped, topic, self._store(topic), qos)

        self.pub_arm = self.create_publisher(JointState, "/fr3/joint_target", qos)
        self.pub_hand = self.create_publisher(JointState, "/xhand/joint_target", qos)

        self.prev_targets: np.ndarray | None = None
        self.warmup_left = WARMUP_STEPS
        self.timer = self.create_timer(c.control_dt, self._tick)

    def _store(self, topic):
        def cb(msg):
            self._msgs[topic] = msg
        return cb

    def _joints_canonical(self, msg: JointState, names: list[str]):
        idx = [list(msg.name).index(n) for n in names]
        assert len(msg.velocity) == len(msg.position), (
            f"joint state without velocities on {msg.header.frame_id or 'unknown'}"
        )
        pos = np.array(msg.position)[idx]
        vel = np.array(msg.velocity)[idx]
        return pos, vel

    def _assert_fresh(self, topic) -> None:
        msg = self._msgs[topic]
        age = _stamp_to_s(self.get_clock().now().to_msg()) - _stamp_to_s(msg.header.stamp)
        assert age < STALE_S, f"{topic} is stale, {age:.3f} s old"

    def _publish(self, targets: np.ndarray) -> None:
        c = self.contract
        stamp = self.get_clock().now().to_msg()
        arm = JointState()
        arm.header.stamp = stamp
        arm.name = self.arm_names
        arm.position = targets[: c.num_arm].tolist()
        hand = JointState()
        hand.header.stamp = stamp
        hand.name = self.hand_names
        hand.position = targets[c.num_arm :].tolist()
        self.pub_arm.publish(arm)
        self.pub_hand.publish(hand)

    def _tick(self) -> None:
        c = self.contract
        needed = (
            "/fr3/joint_states",
            "/xhand/joint_states",
            "/robot_frame/current_object_pose",
            "/robot_frame/goal_object_pose",
        )
        if any(t not in self._msgs for t in needed):
            return
        for t in needed:
            self._assert_fresh(t)

        q_arm, qd_arm = self._joints_canonical(self._msgs["/fr3/joint_states"], self.arm_names)
        q_hand, qd_hand = self._joints_canonical(self._msgs["/xhand/joint_states"], self.hand_names)
        q = np.concatenate([q_arm, q_hand])
        qd = np.concatenate([qd_arm, qd_hand])

        if self.warmup_left > 0:
            # Hold the measured pose while the LSTM sees real observations,
            # then start from a cleared RNN exactly like sim reset.
            self.prev_targets = q.copy()
            obs = self._observation(q, qd)
            self.player.get_normalized_action(obs, deterministic_actions=True)
            self._publish(np.clip(q, c.joint_lower, c.joint_upper))
            self.warmup_left -= 1
            if self.warmup_left == 0:
                self.player.player.init_rnn()
                self.get_logger().info("warmup done, policy active")
            return

        obs = self._observation(q, qd)
        action = self.player.get_normalized_action(obs, deterministic_actions=True)
        targets = compute_targets(c, action[0].detach().cpu().numpy(), self.prev_targets)

        arm_diff_deg = np.rad2deg(np.abs(targets[: c.num_arm] - q[: c.num_arm]).max())
        assert arm_diff_deg < MAX_ARM_TARGET_DIFF_DEG, (
            f"arm target {arm_diff_deg:.1f} deg from measured, aborting"
        )

        self.prev_targets = targets
        self._publish(targets)

    def _observation(self, q: np.ndarray, qd: np.ndarray) -> torch.Tensor:
        obs = build_observation(
            self.contract,
            self.fk,
            q,
            qd,
            self.prev_targets,
            _pose_to_xyz_xyzw(self._msgs["/robot_frame/current_object_pose"]),
            _pose_to_xyz_xyzw(self._msgs["/robot_frame/goal_object_pose"]),
            self.object_scales,
        )
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    rclpy.init()
    rclpy.spin(PolicyNode())


if __name__ == "__main__":
    main()
