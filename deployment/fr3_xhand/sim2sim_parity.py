"""Gate 1, deploy math against the live Isaac Sim env.

Runs the checkpoint in the loop with observations built by the DEPLOY math
(fr3_xhand_nodes.obs_action) instead of the env's own tensor, and at every
step diffs both obs and action targets field by field against the env. DR and
reset noise are off, so the two paths must agree to numerical precision, FK
fields to millimeters (yourdfpy URDF versus baked USD body poses).

    .venv_isaacsim/bin/python deployment/fr3_xhand/sim2sim_parity.py --episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "deployment/fr3_xhand/ws/src/fr3_xhand_nodes"))

# Per-field absolute tolerances. Exact fields are pure reindexing of env
# state, FK fields compare two kinematics implementations, quats compare up
# to double cover.
FIELD_TOL = {
    "joint_pos": 1e-5,
    "joint_vel": 1e-5,
    "prev_action_targets": 1e-5,
    "object_scales": 1e-5,
    "keypoints_rel_goal": 1e-4,
    "palm_pos": 2e-3,
    "fingertip_pos_rel_palm": 2e-3,
    "keypoints_rel_palm": 2e-3,
    "palm_rot": 1e-3,
    "object_rot": 1e-4,
}
QUAT_FIELDS = {"palm_rot", "object_rot"}


def _field_slices(contract):
    sizes = {
        "joint_pos": contract.num_joints,
        "joint_vel": contract.num_joints,
        "prev_action_targets": contract.num_joints,
        "palm_pos": 3,
        "palm_rot": 4,
        "object_rot": 4,
        "fingertip_pos_rel_palm": 15,
        "keypoints_rel_palm": 12,
        "keypoints_rel_goal": 12,
        "object_scales": 3,
    }
    slices, off = {}, 0
    for f in contract.obs_list:
        slices[f] = slice(off, off + sizes[f])
        off += sizes[f]
    assert off == contract.num_observations
    return slices


def _world_pose_to_base(t_world_base: np.ndarray, pos_w, quat_wxyz) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    r_wb = t_world_base[:3, :3]
    w, x, y, z = quat_wxyz
    r_obj_w = Rotation.from_quat([x, y, z, w]).as_matrix()
    pos_b = r_wb.T @ (np.asarray(pos_w) - t_world_base[:3, 3])
    quat_b_xyzw = Rotation.from_matrix(r_wb.T @ r_obj_w).as_quat()
    return np.concatenate([pos_b, quat_b_xyzw])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--rl_device", default="cuda")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer
    from fr3_xhand_nodes.contract import Contract, Fk
    from fr3_xhand_nodes.obs_action import build_observation, compute_targets

    contract = Contract()
    fk = Fk(contract)
    slices = _field_slices(contract)

    cfg = SimToolRealEnvCfg()
    cfg.robot = contract.raw["robot_profile"]
    cfg.scene.num_envs = 1
    cfg.assets.num_assets_per_type = 1

    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    rs = cfg.reset
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    perm = inner._perm_lab_to_canon.cpu().numpy()
    t_wb = contract.t_world_base

    player = RlPlayer(
        num_observations=contract.num_observations,
        num_actions=contract.num_actions,
        config_path=str(contract.train_yaml_path),
        checkpoint_path=str(contract.checkpoint_path),
        device=args.rl_device,
        num_envs=1,
    )

    worst = {f: 0.0 for f in contract.obs_list}
    worst_target = 0.0
    total_hits = 0
    episodes_with_lift = 0

    for ep in range(args.episodes):
        player.player.init_rnn()
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(torch.zeros(1, contract.num_actions, device=inner.device))
        lifted_this_ep = False

        for step in range(args.steps):
            env_origin = inner.scene.env_origins[0].cpu().numpy()

            q = inner.robot.data.joint_pos[0].cpu().numpy()[perm]
            qd = inner.robot.data.joint_vel[0].cpu().numpy()[perm]
            prev_targets = inner._prev_targets[0].cpu().numpy()[perm]
            object_pose_base = _world_pose_to_base(
                t_wb,
                inner.object.data.root_pos_w[0].cpu().numpy() - env_origin,
                inner.object.data.root_quat_w[0].cpu().numpy(),
            )
            goal_pose_base = _world_pose_to_base(
                t_wb,
                inner.goal_viz.data.root_pos_w[0].cpu().numpy() - env_origin,
                inner.goal_viz.data.root_quat_w[0].cpu().numpy(),
            )
            scales = inner._object_scale_per_env[0].cpu().numpy()

            deploy_obs = build_observation(
                contract, fk, q, qd, prev_targets,
                object_pose_base, goal_pose_base, scales,
            )
            env_obs = obs["policy"][0].cpu().numpy()

            for f, sl in slices.items():
                a, b = deploy_obs[sl], env_obs[sl]
                if f in QUAT_FIELDS:
                    diff = min(np.abs(a - b).max(), np.abs(a + b).max())
                else:
                    diff = np.abs(a - b).max()
                worst[f] = max(worst[f], float(diff))

            action = player.get_normalized_action(
                torch.tensor(deploy_obs, dtype=torch.float32, device=args.rl_device).unsqueeze(0),
                deterministic_actions=True,
            )
            action_np = action[0].detach().cpu().numpy()

            deploy_targets = compute_targets(contract, action_np, prev_targets)
            obs, _, terminated, truncated, _ = env.step(action.to(inner.device))

            # On a reset step _reset_idx overwrites the target buffers with
            # the reset pose inside env.step, so the pipeline output is gone.
            if not (terminated.any() or truncated.any()):
                env_targets = inner._cur_targets[0].cpu().numpy()[perm]
                worst_target = max(worst_target, float(np.abs(deploy_targets - env_targets).max()))

            total_hits += int(inner._is_success.sum().item())
            lifted_this_ep = lifted_this_ep or bool(inner._lifted_object.any().item())

            if terminated.any() or truncated.any():
                break

        episodes_with_lift += lifted_this_ep
        print(f"[parity] episode {ep + 1}/{args.episodes} done, hits so far {total_hits}")

    print(f"[parity] worst target diff {worst_target:.2e}")
    for f in contract.obs_list:
        print(f"[parity] worst obs diff {f:24s} {worst[f]:.2e} (tol {FIELD_TOL[f]:.0e})")

    assert worst_target < 1e-6, f"target math drifted, {worst_target}"
    for f in contract.obs_list:
        assert worst[f] < FIELD_TOL[f], f"obs field {f} drifted, {worst[f]}"
    assert total_hits > 0, "policy reached no goal through the deploy math"

    print(f"[parity] OK, goal hits {total_hits}, episodes with lift {episodes_with_lift}")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
