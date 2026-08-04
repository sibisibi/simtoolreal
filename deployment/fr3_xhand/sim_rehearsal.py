"""Rehearse the checkpoint on our object and record how close it gets.

eval_isaacsim.py reports goals reached, which is pass or fail at a 1 cm
tolerance. That number cannot tell a weak checkpoint from a broken object spec,
because both score zero. This logs the env's own success metric every step,
``_keypoints_max_dist``, so the distance the policy closes is visible.

The env config mirrors eval_isaacsim.py so the two runs stay comparable.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
        deployment/fr3_xhand/sim_rehearsal.py \
        --object_name davian_handle_eraser \
        --task_name wipe_up_down --category eraser --num_episodes 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
# Training sampled goals 10 cm out and scored them at 7.5 cm, tightening toward
# 1 cm. Both bars are drawn so a rollout can be read against either.
TRAIN_TOLERANCE_M = 0.075
TARGET_TOLERANCE_M = 0.01


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--config_path", default=str(REPO_ROOT / "deployment/fr3_xhand/contract/a4h1_train.yaml"))
    parser.add_argument("--checkpoint_path", default=str(REPO_ROOT / "runs/a4h1_20260731_085255/0_simtoolreal_sapg/last/model.pth"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--z_offset", type=float, default=0.03)
    parser.add_argument("--rl_device", default="cuda")
    return parser


def _launch_app():
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return AppLauncher(args).app, args


_app, _args = _launch_app()


def main() -> None:
    args = _args
    import gymnasium as gym
    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    from deployment.rl_player import RlPlayer
    from dextoolbench.objects import NAME_TO_OBJECT

    obj = NAME_TO_OBJECT[args.object_name]
    traj_path = (
        REPO_ROOT / "dextoolbench/trajectories" / args.category / args.object_name
        / f"{args.task_name}.json"
    )
    assert traj_path.exists(), f"trajectory not found, {traj_path}"
    traj = json.loads(traj_path.read_text())
    traj["start_pose"][2] += args.z_offset
    n_goals = len(traj["goals"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    isaac_traj = out_dir / "trajectory_isaac_format.json"
    isaac_traj.write_text(json.dumps({
        "pos": [[g[:3] for g in traj["goals"]]],
        "quat_wxyz": [[[g[6], g[3], g[4], g[5]] for g in traj["goals"]]],
    }))

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = 1
    cfg.assets.object_urdf = str(obj.decomposed_urdf_path)
    cfg.assets.object_scale = tuple(obj.scale)
    # The task's own environment URDF, which carries our real bench rather than
    # the benchmark's narrow table.
    cfg.assets.table_urdf = str(
        REPO_ROOT / "assets/urdf/dextoolbench/environments"
        / args.category / args.object_name / f"{args.task_name}.urdf"
    )

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z = 0.38
    rs.table_reset_z_range = 0.0
    rs.start_arm_higher = True
    sp = traj["start_pose"]
    rs.fixed_start_pose = (sp[0], sp[1], sp[2], sp[6], sp[3], sp[4], sp[5])
    rs.fixed_trajectory_file = str(isaac_traj)

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

    term = cfg.termination
    term.eval_success_tolerance = TARGET_TOLERANCE_M
    term.success_steps = 1
    term.max_consecutive_successes = n_goals

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    player = RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=cfg.action_space,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.rl_device,
        num_envs=1,
    )

    episodes = []
    for ep in range(args.num_episodes):
        player.player.init_rnn()
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(torch.zeros((1, cfg.action_space), device=inner.device))

        track = {"keypoint_dist": [], "object_z": [], "fingertip_min": [], "successes": []}
        done = False
        while not done:
            action = player.get_normalized_action(
                obs["policy"].to(args.rl_device), deterministic_actions=True
            )
            obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
            track["keypoint_dist"].append(float(inner._keypoints_max_dist[0].item()))
            track["object_z"].append(float(inner.object.data.root_pos_w[0, 2].item()))
            track["fingertip_min"].append(float(inner._curr_fingertip_distances[0].min().item()))
            track["successes"].append(int(inner._successes[0].item()))
            done = bool(terminated[0].item() or truncated[0].item())

        d = np.array(track["keypoint_dist"])
        z = np.array(track["object_z"])
        ft = np.array(track["fingertip_min"])
        episodes.append({
            "steps": len(d),
            "keypoint_dist_start": float(d[0]),
            "keypoint_dist_min": float(d.min()),
            "keypoint_dist_end": float(d[-1]),
            "object_lift_m": float(z.max() - z[0]),
            "fingertip_min_m": float(ft.min()),
            "goals_reached": int(inner._prev_episode_successes[0].item()),
            "track": track,
        })
        e = episodes[-1]
        print(
            f"[rehearsal] ep {ep + 1}/{args.num_episodes}, {e['steps']} steps, "
            f"keypoint dist {e['keypoint_dist_start']:.3f} to {e['keypoint_dist_min']:.3f} m, "
            f"lift {e['object_lift_m'] * 100:.1f} cm, "
            f"closest fingertip {e['fingertip_min_m'] * 100:.1f} cm, "
            f"goals {e['goals_reached']}/{n_goals}"
        )

    mins = np.array([e["keypoint_dist_min"] for e in episodes])
    summary = {
        "object_name": args.object_name,
        "task_name": args.task_name,
        "num_goals": n_goals,
        "keypoint_dist_min_mean": float(mins.mean()),
        "reached_train_tolerance": bool((mins <= TRAIN_TOLERANCE_M).any()),
        "reached_target_tolerance": bool((mins <= TARGET_TOLERANCE_M).any()),
        "episodes": episodes,
    }
    (out_dir / "rehearsal.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[rehearsal] closest approach {mins.mean():.3f} m, "
        f"training bar {TRAIN_TOLERANCE_M} m reached {summary['reached_train_tolerance']}, "
        f"target bar {TARGET_TOLERANCE_M} m reached {summary['reached_target_tolerance']}"
    )
    print(f"[rehearsal] wrote {out_dir / 'rehearsal.json'}")

    env.close()
    _app.close()


if __name__ == "__main__":
    main()
