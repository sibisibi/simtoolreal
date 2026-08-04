"""Export the deploy contract from a live SimToolReal Isaac Sim env.

The deploy stack must reproduce the orderings Isaac Lab produces at runtime,
the canonical-to-Lab joint permutation and the fingertip body order from
find_bodies(). Those exist only inside a constructed env, so the contract is
read from a live env, never assembled from config files by hand.

The run was trained with env.robot=fr3-xhand-a4h1, a registry key that no
longer exists. The user folded that exact setup into the fr3-xhand profile on
main, so the profile key is an explicit argument here and every contract field
the run config also stores is asserted against it below.

    .venv_isaacsim/bin/python deployment/fr3_xhand/export_contract.py \
        --run runs/a4h1_20260731_085255 --robot fr3-xhand
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import yaml

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_contract(env, robot_key: str, run_dir: Path) -> dict:
    from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import KEYPOINT_CORNERS

    cfg = env.cfg
    robot = cfg.robot
    run_cfg = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
    env_run_cfg = run_cfg["env"]

    # The saved hydra config is authoritative for what the checkpoint saw.
    # Any mismatch with the live cfg means the codebase drifted from the run.
    assert env_run_cfg["action"]["dof_speed_scale"] == cfg.action.dof_speed_scale
    assert env_run_cfg["action"]["arm_moving_average"] == cfg.action.arm_moving_average
    assert env_run_cfg["action"]["hand_moving_average"] == cfg.action.hand_moving_average
    assert tuple(env_run_cfg["obs"]["obs_list"]) == tuple(cfg.obs.obs_list)
    assert env_run_cfg["obs"]["clamp_abs_observations"] == cfg.obs.clamp_abs_observations
    assert env_run_cfg["reward"]["keypoint_scale"] == cfg.reward.keypoint_scale
    assert env_run_cfg["reward"]["object_base_size"] == cfg.reward.object_base_size
    # The saved yaml truncates the 1/120 float literal.
    assert math.isclose(env_run_cfg["sim"]["dt"], cfg.sim.dt, rel_tol=1e-9)
    assert env_run_cfg["decimation"] == cfg.decimation

    lab_joint_names = list(env.robot.data.joint_names)
    canon = list(robot.joint_order)
    n_arm = len(env._arm_joint_ids)
    # The deploy split slices canonical index n_arm, valid only if canonical
    # order is exactly arm joints then hand joints.
    assert canon[:n_arm] == [n for n in canon if re.fullmatch(robot.arm_joint_regex, n)]
    # apply_action_pipeline slices actions[:, :n_arm] in Lab order but writes
    # through _arm_joint_ids, consistent only when Lab order is also arm-first.
    assert list(env._arm_joint_ids) == list(range(n_arm))
    assert list(env._hand_joint_ids) == list(range(n_arm, len(canon)))

    default_pos_canon = (
        env.robot.data.default_joint_pos[0, env._perm_lab_to_canon].tolist()
    )
    env_origin = env.scene.env_origins[0]
    base_pos = (env.robot.data.root_pos_w[0] - env_origin).tolist()
    base_quat_wxyz = env.robot.data.root_quat_w[0].tolist()

    return {
        "robot_profile": robot_key,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "checkpoint": str(
            (run_dir / "0_simtoolreal_sapg" / "last" / "model.pth").relative_to(REPO_ROOT)
        ),
        "urdf": robot.urdf,
        "num_observations": cfg.observation_space,
        "num_actions": cfg.action_space,
        "num_arm_joints": n_arm,
        "joint_order": canon,
        "lab_joint_order": lab_joint_names,
        "joint_lower": env._joint_lower_canon.tolist(),
        "joint_upper": env._joint_upper_canon.tolist(),
        "default_joint_pos": default_pos_canon,
        "palm_body": robot.palm_body,
        "palm_center_offset": list(robot.palm_center_offset),
        "fingertip_bodies_lab_order": list(env._fingertip_body_names),
        "fingertip_offset_by_body": {
            name: list(robot.fingertip_offset_by_body[name])
            for name in env._fingertip_body_names
        },
        "base_pos": base_pos,
        "base_quat_wxyz": base_quat_wxyz,
        # Nominal table surface height in the env frame, for the real-table
        # geometry check. The live table pose is reset-dependent, so the cfg
        # value is the honest reference.
        "table_reset_z": cfg.reset.table_reset_z,
        "arm_stiffness": dict(robot.arm_stiffness),
        "arm_damping": dict(robot.arm_damping),
        "arm_armature": dict(robot.arm_armature),
        "hand_stiffness": dict(robot.hand_stiffness),
        "hand_damping": dict(robot.hand_damping),
        "control_dt": cfg.decimation * cfg.sim.dt,
        "dof_speed_scale": cfg.action.dof_speed_scale,
        "arm_moving_average": cfg.action.arm_moving_average,
        "hand_moving_average": cfg.action.hand_moving_average,
        "obs_list": list(cfg.obs.obs_list),
        "clamp_abs_observations": cfg.obs.clamp_abs_observations,
        "keypoint_corners": [list(c) for c in KEYPOINT_CORNERS],
        "object_base_size": cfg.reward.object_base_size,
        "keypoint_scale": cfg.reward.keypoint_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/a4h1_20260731_085255")
    parser.add_argument("--robot", default="fr3-xhand")
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "deployment/fr3_xhand/contract")
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401  registers gym envs
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    run_dir = REPO_ROOT / args.run
    out_dir = Path(args.out)

    cfg = SimToolRealEnvCfg()
    cfg.robot = args.robot
    cfg.scene.num_envs = 1
    cfg.assets.num_assets_per_type = 1

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    contract = build_contract(env.unwrapped, args.robot, run_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = out_dir / "a4h1.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    from omegaconf import OmegaConf

    # rl_player.read_cfg expects the rl_games block under a top-level train
    # key, the hydra run config stores it under agent, and that block holds
    # interpolations into env, so resolve against the full tree first.
    full = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    resolved = OmegaConf.to_container(full, resolve=True)
    train_path = out_dir / "a4h1_train.yaml"
    train_path.write_text(yaml.safe_dump({"train": resolved["agent"]}))

    print(f"wrote {contract_path}")
    print(f"wrote {train_path}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
