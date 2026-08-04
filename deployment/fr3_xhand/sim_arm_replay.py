"""Replay the real arm's joint targets in sim and compare the tracking error.

The sine sweep on hardware produces a target stream and a measured response.
On its own the error means nothing, since a PD at kp 400 always lags. Driving
the same targets through the same actuator model in sim gives the comparison
that does mean something, because sim is what the policy was trained against.

The metric is the trajectory error from Tune to Learn, sim against real over
position and velocity together. It is reported, not thresholded, because the
number that matters is whether the policy transfers, and no line drawn here
predicts that.

Targets come from the bag the arm session recorded, so both sides see an
identical sequence rather than a regenerated one. bag_to_targets.py does the
reading, because rosbag2_py belongs to the system ROS python.

    /usr/bin/python3 deployment/fr3_xhand/bag_to_targets.py \
        --bag <rosbag dir> --out arm_targets.npz
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
        deployment/fr3_xhand/sim_arm_replay.py --targets arm_targets.npz --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
SETTLE_STEPS = 120  # two seconds at 60 Hz before anything is measured


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, help="npz from bag_to_targets.py")
    parser.add_argument("--out", required=True)
    return parser


def _launch():
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return AppLauncher(args).app, args


_app, _args = _launch()


def main() -> None:
    args = _args
    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    loaded = np.load(args.targets)
    targets, real = loaded["targets"], loaded["measured"]
    print(f"[replay] {len(targets)} targets from {args.targets}")

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1
    # The comparison is about the actuator, so every source of divergence that
    # is not the actuator gets switched off.
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)
    rs = cfg.reset
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    # A reset teleports the arm and reads as a huge tracking error. Episodes are
    # ten seconds and the sweep is over sixty, so without this the comparison
    # measures resets rather than the actuator.
    cfg.episode_length_s = 1.0e6

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    env.reset()

    # Start the sim arm where the real arm started, so the inertia the PD works
    # against is the same. The hand holds its reset pose throughout.
    q = inner.robot.data.joint_pos.clone()
    q[0, : len(ARM_NAMES)] = torch.tensor(targets[0], device=inner.device, dtype=q.dtype)
    inner.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    inner._cur_targets[:] = q
    inner._prev_targets = q.clone()

    # Settle before measuring, so the teleport into the start pose is not counted.
    for _ in range(SETTLE_STEPS):
        inner._replay_target_lab_order = inner._cur_targets.clone()
        env.step(torch.zeros((1, cfg.action_space), device=inner.device))

    measured, measured_vel, resets = [], [], 0
    prev_len = int(inner.episode_length_buf[0].item())
    zero_action = torch.zeros((1, cfg.action_space), device=inner.device)
    for row in targets:
        tgt = inner._cur_targets.clone()
        tgt[0, : len(ARM_NAMES)] = torch.tensor(row, device=inner.device, dtype=tgt.dtype)
        inner._replay_target_lab_order = tgt
        env.step(zero_action)
        now_len = int(inner.episode_length_buf[0].item())
        if now_len < prev_len:
            resets += 1
        prev_len = now_len
        measured.append(inner.robot.data.joint_pos[0, : len(ARM_NAMES)].cpu().numpy().copy())
        measured_vel.append(inner.robot.data.joint_vel[0, : len(ARM_NAMES)].cpu().numpy().copy())
    measured = np.stack(measured)
    measured_vel = np.stack(measured_vel)
    assert resets == 0, f"the env reset {resets} times mid replay, the comparison is void"
    print(f"[replay] no resets over {len(targets)} steps")

    real_vel = loaded["measured_vel"]
    out = {"joints": {}, "num_targets": int(len(targets))}
    print(f"{'joint':>10} {'pos rms deg':>12} {'vel rms deg/s':>14} {'traj err':>10}")
    total = 0.0
    for j, name in enumerate(ARM_NAMES):
        moving = np.abs(targets[:, j] - targets[0, j]) > 1e-4
        if moving.sum() < 10:
            continue
        dq = measured[moving, j] - real[moving, j]
        dv = measured_vel[moving, j] - real_vel[moving, j]
        # Tune to Learn sums the squared position and velocity discrepancy.
        err = float((dq ** 2).mean() + (dv ** 2).mean())
        total += err
        out["joints"][name] = {
            "pos_rms_deg": float(np.degrees(np.sqrt((dq ** 2).mean()))),
            "vel_rms_deg_s": float(np.degrees(np.sqrt((dv ** 2).mean()))),
            "traj_err": err,
        }
        print(f"{name:>10} {out['joints'][name]['pos_rms_deg']:12.4f} "
              f"{out['joints'][name]['vel_rms_deg_s']:14.4f} {err:10.6f}")
    out["traj_err_total"] = total
    print(f"\n[replay] trajectory error summed over joints {total:.6f}, sim against real")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[replay] wrote {args.out}")

    env.close()
    _app.close()


if __name__ == "__main__":
    main()
