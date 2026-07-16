"""State allocation and reset helpers for SimToolReal."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import random_orientation

from .action_utils import sample_log_uniform
from .goal_sampling import sample_absolute_goal_pose, sample_delta_goal_pose
from .obs_utils import KEYPOINT_CORNERS, NUM_FINGERTIPS

RESET_MODE_NAMES = {0: "default", 1: "near_object", 2: "grasped_air", 3: "grasped_goal"}

def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer + index cache used by the hooks.

    Called once from ``__init__`` after ``super().__init__`` (which runs
    ``_setup_scene`` and makes ``env.robot``/``env.object``/``env.goal_viz``
    available).
    """
    dr = env.cfg.domain_randomization
    rew = env.cfg.reward

    # --- Joint/body id caches ---
    env._arm_joint_ids = env.robot.find_joints(env.cfg.robot.arm_joint_regex)[0]      # 7
    env._hand_joint_ids = env.robot.find_joints(env.cfg.robot.hand_joint_regex)[0]     # 22
    env._palm_body_id = env.robot.find_bodies(env.cfg.robot.palm_body)[0][0]
    _ft_ids, _ft_names = env.robot.find_bodies(env.cfg.robot.fingertip_body_regex)
    env._fingertip_body_ids = _ft_ids  # 5
    env._fingertip_body_names = list(_ft_names)  # 009: same order as ids

    # Convert between Lab parser order and canonical policy order.
    lab_names = list(env.robot.data.joint_names)
    env._perm_canon_to_lab = torch.tensor(
        [env.cfg.robot.joint_order.index(n) for n in lab_names],
        device=env.device, dtype=torch.long,
    )
    env._perm_lab_to_canon = torch.tensor(
        [lab_names.index(n) for n in env.cfg.robot.joint_order],
        device=env.device, dtype=torch.long,
    )

    limits = env.robot.data.joint_pos_limits  # (N, num_joints, 2), Lab order

    # Canonical-order limits for normalizing joint_pos observations.
    env._joint_lower_canon = limits[0, :, 0][env._perm_lab_to_canon]  # (29,)
    env._joint_upper_canon = limits[0, :, 1][env._perm_lab_to_canon]  # (29,)

    # Lab-order limits for action target clamping.
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    # --- Action target buffers  ---
    action_space = env.cfg.action_space
    env._cur_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._prev_targets = torch.zeros(env.num_envs, action_space, device=env.device)

    # --- Keypoint offsets in object local frame ---
    corners = torch.tensor(
        KEYPOINT_CORNERS, device=env.device, dtype=torch.float32
    )  # (4, 3)

    env._keypoint_offsets = (
        corners.unsqueeze(0)
        * (env._object_scale_per_env * rew.object_base_size * rew.keypoint_scale * 0.5).unsqueeze(1)
    )  # (N, 4, 3)

    # Fixed-size reward keypoints follow legacy: fixed_size * keypoint_scale / 2.
    env._keypoint_offsets_fixed = (
        corners
        * (0.5 * rew.keypoint_scale * torch.tensor(rew.fixed_size, device=env.device)).unsqueeze(0)
    ).unsqueeze(0).expand(env.num_envs, -1, -1).contiguous()

    # --- Per-env DR priors (re-sampled on reset; seeded once here) ---
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(
        env.num_envs, 3, device=env.device
    ).uniform_(lo, hi)

    # --- Reward / termination trackers  ---
    # whether the object is lifted
    env._lifted_object = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    # the maximum distance between the object and the goal since the last goal reset
    env._closest_keypoint_max_dist = torch.full(
        (env.num_envs,), -1.0, device=env.device
    )
    # the minimum distance between a fingertip and the object since the last goal reset
    env._closest_fingertip_dist = torch.full(
        (env.num_envs, NUM_FINGERTIPS), -1.0, device=env.device
    )
    # the number of succeesses in the current episode
    env._successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    # the number of consecutive steps that the object is near the goal
    env._near_goal_steps = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )

    # --- Tolerance curriculum state ---
    env._current_success_tolerance: float = env.cfg.termination.success_tolerance
    env._prev_episode_successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._frame_counter: int = 0
    env._last_curriculum_update: int = 0

    # --- Object lifted-reward reference z (updated on each _reset_object_pose) ---
    init_z = env.cfg.reset.table_reset_z + env.cfg.reset.table_object_z_offset
    env._object_init_z = torch.full(
        (env.num_envs,), init_z, device=env.device
    )

    # --- Per-env table surface z (randomized in _reset_table_pose) ---
    env._table_z_per_env = torch.full(
        (env.num_envs,), env.cfg.reset.table_reset_z, device=env.device
    )

    # --- Per-env reset-mode tag + windowed per-mode episode successes (019) ---
    env._reset_mode_per_env = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._mode_success_windows = {
        mode: deque(maxlen=2048) for mode in RESET_MODE_NAMES
    }

    # --- DR rolling buffers ---
    env._object_state_queue = torch.zeros(
        env.num_envs,
        max(1, dr.object_state_delay_max),
        13,  # pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        device=env.device,
    )
    env._obs_queue = torch.zeros(
        env.num_envs,
        max(1, dr.obs_delay_max),
        env.cfg.observation_space,
        device=env.device,
    )
    env._action_queue = torch.zeros(
        env.num_envs,
        max(1, dr.action_delay_max),
        action_space,
        device=env.device,
    )

    # --- Wrench DR state (Phase C/D) ---
    env._random_force_prob = sample_log_uniform(
        dr.force_prob_range, env.num_envs
    ).to(env.device)
    env._random_torque_prob = sample_log_uniform(
        dr.torque_prob_range, env.num_envs
    ).to(env.device)
    env._object_forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_torques = torch.zeros(env.num_envs, 1, 3, device=env.device)
    env._object_mass = env.object.data.default_mass[:, 0:1].to(env.device)  # (N, 1)

    # --- Step-shared caches populated by compute_intermediate_values (Phase F) ---
    env._keypoints_max_dist = torch.zeros(env.num_envs, device=env.device)
    env._curr_fingertip_distances = torch.zeros(
        env.num_envs, NUM_FINGERTIPS, device=env.device
    )
    env._near_goal = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._is_success = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # --- Fixed-trajectory pool (trajectory_count ablation) ---
    # Loaded once at init from the JSON file. Different runs share the same
    # file and pick the first N via cfg.reset.fixed_trajectory_count
    # (0 = take all). Empty filename = ablation disabled (baseline path).
    if env.cfg.reset.fixed_trajectory_file:
        path = Path(env.cfg.reset.fixed_trajectory_file)
        with open(path) as f:
            payload = json.load(f)
        all_pos = torch.tensor(payload["pos"], device=env.device, dtype=torch.float32)
        all_quat = torch.tensor(payload["quat_wxyz"], device=env.device, dtype=torch.float32)
        n_total = all_pos.shape[0]
        n_take = env.cfg.reset.fixed_trajectory_count or n_total
        if n_take > n_total:
            raise ValueError(
                f"fixed_trajectory_count={n_take} exceeds pool size {n_total} in {path}"
            )
        env._fixed_traj_pos = all_pos[:n_take].contiguous()    # (N, K, 3)
        env._fixed_traj_quat = all_quat[:n_take].contiguous()  # (N, K, 4)
        env._traj_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._traj_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # set the reward buffer to 0
    env.reward_buf = torch.zeros(env.num_envs, device=env.device)


def _randomize_robot_dof_state(env, env_ids: torch.Tensor) -> None:
    """Reset DOF state and seed previous targets from the reset pose."""
    cfg = env.cfg.reset
    default_pos = env.robot.data.default_joint_pos[env_ids]  # (n, num_dofs)
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]

    reset_scale = torch.zeros_like(default_pos)
    reset_scale[:, env._arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
    reset_scale[:, env._hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

    sampled_pos = lower + (upper - lower) * torch.rand_like(default_pos)
    joint_pos = torch.lerp(default_pos, sampled_pos, reset_scale).clamp(lower, upper)
    joint_vel = torch.empty_like(default_pos).uniform_(
        -cfg.reset_dof_vel_random_interval,
        cfg.reset_dof_vel_random_interval,
    )

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos


def _reset_table_pose(env, env_ids: torch.Tensor) -> None:
    """Randomize the table's pose per env and write the new transform.

    Z position is required (the policy uses `_table_z_per_env` to set the
    object init height). XY position and yaw are gated on
    `table_reset_xy_range_m` / `table_reset_yaw_range_deg` — default ranges
    are zero so this is a no-op for runs that don't opt in.
    """
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    dz = torch.empty(n, device=env.device).uniform_(
        -cfg.table_reset_z_range, cfg.table_reset_z_range
    )
    table_z = cfg.table_reset_z + dz
    env._table_z_per_env[env_ids] = table_z

    pos_local = torch.zeros(n, 3, device=env.device)
    pos_local[:, 2] = table_z

    # XY position noise — per-env independent uniform half-widths.
    xy_range = tuple(float(v) for v in cfg.table_reset_xy_range_m)
    if xy_range[0] > 0.0 or xy_range[1] > 0.0:
        rx = torch.empty(n, device=env.device).uniform_(-xy_range[0], xy_range[0])
        ry = torch.empty(n, device=env.device).uniform_(-xy_range[1], xy_range[1])
        pos_local[:, 0] = rx
        pos_local[:, 1] = ry

    # Yaw noise — sample uniform [-r, r] degrees, build a z-axis rotation quat.
    yaw_range_deg = float(cfg.table_reset_yaw_range_deg)
    if yaw_range_deg > 0.0:
        yaw_rad = (
            torch.empty(n, device=env.device).uniform_(-1.0, 1.0)
            * yaw_range_deg * (torch.pi / 180.0)
        )
        half = yaw_rad * 0.5
        w = torch.cos(half)
        z = torch.sin(half)
        quat = torch.stack([w, torch.zeros_like(w), torch.zeros_like(w), z], dim=-1)
    else:
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.table.write_root_pose_to_sim(pose, env_ids=env_ids)

def _reset_object_pose(env, env_ids: torch.Tensor) -> None:
    """Reset object pose and lifted-reward reference height."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_start_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_start_pose, device=env.device, dtype=torch.float32)
        pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        quat = fixed[3:].unsqueeze(0).expand(n, -1)
    else:
        noise = torch.empty(n, 3, device=env.device).uniform_(-1.0, 1.0)
        pos_local = torch.stack(
            (
                noise[:, 0] * cfg.reset_position_noise_x,
                noise[:, 1] * cfg.reset_position_noise_y,
                env._table_z_per_env[env_ids]
                + cfg.table_object_z_offset
                + noise[:, 2] * cfg.reset_position_noise_z,
            ),
            dim=-1,
        )
        quat = random_orientation(n, device=env.device)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )

    env._object_init_z[env_ids] = pos_local[:, 2]


def _reset_goal_pose(env, env_ids: torch.Tensor, mode: str) -> None:
    """Resample the goal pose and write it to GoalViz."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    if cfg.fixed_goal_pose is not None:
        fixed = torch.as_tensor(cfg.fixed_goal_pose, device=env.device, dtype=torch.float32)
        new_pos_local = fixed[:3].unsqueeze(0).expand(n, -1)
        new_quat = fixed[3:].unsqueeze(0).expand(n, -1)
        pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        return

    # Fixed-trajectory ablation: ignore goal_sampling_type, draw from the
    # pre-loaded pool. mode acts as the "hard reset vs intra-episode" signal:
    #   "absolute" → fresh episode → pick new traj_id, reset step to 0
    #   anything else → intra-episode goal-hit → advance step
    # The K-th success briefly calls this with step=K (out of bounds) before
    # max_consecutive_successes terminates the episode; the clamp keeps the
    # lookup safe — the goal we write is then immediately overwritten by the
    # full reset.
    if cfg.fixed_trajectory_file:
        K = env._fixed_traj_pos.shape[1]
        if mode == "absolute":
            n_traj = env._fixed_traj_pos.shape[0]
            env._traj_id[env_ids] = torch.randint(
                0, n_traj, (n,), device=env.device, dtype=torch.long,
            )
            env._traj_step[env_ids] = 0
        else:
            env._traj_step[env_ids] = env._traj_step[env_ids] + 1

        traj_id = env._traj_id[env_ids]
        step_clamped = env._traj_step[env_ids].clamp(max=K - 1)
        new_pos_local = env._fixed_traj_pos[traj_id, step_clamped]
        new_quat = env._fixed_traj_quat[traj_id, step_clamped]
        pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
        env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        return

    if mode == "delta":
        prev_pos_local = env.goal_viz.data.root_pos_w[env_ids] - env_origins
        prev_quat = env.goal_viz.data.root_quat_w[env_ids]
        new_pos_local, new_quat = sample_delta_goal_pose(
            prev_pos=prev_pos_local,
            prev_quat_wxyz=prev_quat,
            delta_distance=cfg.delta_goal_distance,
            delta_rotation_degrees=cfg.delta_rotation_degrees,
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
        )
    elif mode == "absolute":
        new_pos_local, new_quat = sample_absolute_goal_pose(
            mins=cfg.target_volume_mins,
            maxs=cfg.target_volume_maxs,
            scale=cfg.target_volume_region_scale,
            n_envs=n,
            device=env.device,
        )
    else:
        raise ValueError(f"unknown goal sampling mode: {mode}")

    pose = torch.cat([new_pos_local + env_origins, new_quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)


def _clear_goal_trackers(env, env_ids: torch.Tensor) -> None:
    env._closest_keypoint_max_dist[env_ids] = -1.0
    env._closest_fingertip_dist[env_ids] = -1.0
    env._near_goal_steps[env_ids] = 0


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """Clear per-goal trackers and sample the next goal."""
    _clear_goal_trackers(env, env_ids)
    _reset_goal_pose(env, env_ids, mode=env.cfg.reset.goal_sampling_type)


def _load_near_object_bank(env) -> None:
    """Load the 015 near-object reset bank and reorder joints to sim order."""
    bank = np.load(env.cfg.reset.reset_bank_path, allow_pickle=False)
    names = [str(n) for n in bank["joint_names"]]
    order = torch.tensor([names.index(n) for n in env.robot.data.joint_names], dtype=torch.long)
    env._nearobj_qpos = torch.as_tensor(bank["qpos"], dtype=torch.float32)[:, :, order].to(env.device)
    env._nearobj_objpose = torch.as_tensor(bank["objpose"], dtype=torch.float32).to(env.device)
    env._nearobj_count = torch.as_tensor(bank["count"], dtype=torch.long).to(env.device)
    n_assets = int(env._object_asset_index_per_env.max().item()) + 1
    assert env._nearobj_qpos.shape[0] >= n_assets, (env._nearobj_qpos.shape, n_assets)
    assert int(env._nearobj_count.sum().item()) > 0


def _load_grasped_bank(env) -> None:
    """Load the 019 grasped bank. objpose_abs is env-local ABSOLUTE z."""
    bank = np.load(env.cfg.reset.grasped_bank_path, allow_pickle=False)
    names = [str(n) for n in bank["joint_names"]]
    order = torch.tensor([names.index(n) for n in env.robot.data.joint_names], dtype=torch.long)
    env._grasped_qpos = torch.as_tensor(bank["qpos"], dtype=torch.float32)[:, :, order].to(env.device)
    env._grasped_qtarget = torch.as_tensor(bank["qtarget"], dtype=torch.float32)[:, :, order].to(env.device)
    env._grasped_objpose = torch.as_tensor(bank["objpose_abs"], dtype=torch.float32).to(env.device)
    env._grasped_count = torch.as_tensor(bank["count"], dtype=torch.long).to(env.device)
    n_assets = int(env._object_asset_index_per_env.max().item()) + 1
    assert env._grasped_qpos.shape[0] >= n_assets, (env._grasped_qpos.shape, n_assets)
    assert int(env._grasped_count.sum().item()) > 0


def _split_reset_modes(
    env, env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split resetting envs into (near-object, grasped-air, grasped-goal, default).

    Envs whose asset has no rows in the required bank fall back to default.
    """
    cfg = env.cfg.reset
    f_no = cfg.near_object_fraction
    f_ga = cfg.grasped_air_fraction
    f_gg = cfg.grasped_goal_fraction
    if f_no + f_ga + f_gg <= 0.0:
        e = env_ids.new_empty(0)
        return e, e, e, env_ids
    assert f_no + f_ga + f_gg <= 1.0 + 1e-6, (f_no, f_ga, f_gg)
    if f_no > 0.0 and not hasattr(env, "_nearobj_qpos"):
        _load_near_object_bank(env)
    if (f_ga > 0.0 or f_gg > 0.0) and not hasattr(env, "_grasped_qpos"):
        assert not cfg.fixed_trajectory_file, (
            "grasped resets are incompatible with the fixed-trajectory pool"
        )
        _load_grasped_bank(env)
    u = torch.rand(env_ids.numel(), device=env.device)
    asset = env._object_asset_index_per_env[env_ids]
    pick_no = u < f_no
    pick_ga = (u >= f_no) & (u < f_no + f_ga)
    pick_gg = (u >= f_no + f_ga) & (u < f_no + f_ga + f_gg)
    if f_no > 0.0:
        pick_no &= env._nearobj_count[asset] > 0
    if f_ga > 0.0 or f_gg > 0.0:
        has_g = env._grasped_count[asset] > 0
        pick_ga &= has_g
        pick_gg &= has_g
    default = ~(pick_no | pick_ga | pick_gg)
    return env_ids[pick_no], env_ids[pick_ga], env_ids[pick_gg], env_ids[default]


def _reset_near_object_state(env, env_ids: torch.Tensor) -> None:
    """Reset to a bank state, object resting on the table, hand at a pregrasp."""
    n = env_ids.numel()
    asset = env._object_asset_index_per_env[env_ids]
    row = (torch.rand(n, device=env.device) * env._nearobj_count[asset].to(torch.float32)).long()
    joint_pos = env._nearobj_qpos[asset, row]
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]
    joint_pos = joint_pos.clamp(lower, upper)
    env.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos

    objpose = env._nearobj_objpose[asset, row].clone()
    objpose[:, 2] += env._table_z_per_env[env_ids]
    pose = torch.cat([objpose[:, :3] + env.scene.env_origins[env_ids], objpose[:, 3:7]], dim=-1)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
    # 019: init_z matched to the default drop path so shaping and latch
    # thresholds are identically distributed across reset modes.
    cfg = env.cfg.reset
    noise_z = (
        torch.empty(n, device=env.device).uniform_(-1.0, 1.0)
        * cfg.reset_position_noise_z
    )
    env._object_init_z[env_ids] = (
        env._table_z_per_env[env_ids] + cfg.table_object_z_offset + noise_z
    )


def _reset_grasped_state(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Reset to a validated in-hand state, object in the air, hand holding it.

    Returns the env-local object pose written, the grasped-at-goal path
    samples its episode goal from it. _lifted_object pre-latching happens in
    reset_env_state after the tracker clear.
    """
    n = env_ids.numel()
    asset = env._object_asset_index_per_env[env_ids]
    row = (torch.rand(n, device=env.device) * env._grasped_count[asset].to(torch.float32)).long()
    joint_pos = env._grasped_qpos[asset, row]
    joint_target = env._grasped_qtarget[asset, row]
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]
    joint_pos = joint_pos.clamp(lower, upper)
    joint_target = joint_target.clamp(lower, upper)
    env.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    env._prev_targets[env_ids] = joint_target
    env._cur_targets[env_ids] = joint_target

    objpose = env._grasped_objpose[asset, row].clone()
    pose = torch.cat([objpose[:, :3] + env.scene.env_origins[env_ids], objpose[:, 3:7]], dim=-1)
    env.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    env.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)

    # 019: init_z uses the matched default expression, same as every mode.
    cfg = env.cfg.reset
    noise_z = (
        torch.empty(n, device=env.device).uniform_(-1.0, 1.0)
        * cfg.reset_position_noise_z
    )
    env._object_init_z[env_ids] = (
        env._table_z_per_env[env_ids] + cfg.table_object_z_offset + noise_z
    )
    return objpose


def _reset_goal_from_object(env, env_ids: torch.Tensor, objpose_local: torch.Tensor) -> None:
    """Grasped-at-goal, the episode goal is one delta step from the object."""
    cfg = env.cfg.reset
    new_pos, new_quat = sample_delta_goal_pose(
        prev_pos=objpose_local[:, :3],
        prev_quat_wxyz=objpose_local[:, 3:7],
        delta_distance=cfg.delta_goal_distance,
        delta_rotation_degrees=cfg.delta_rotation_degrees,
        mins=cfg.target_volume_mins,
        maxs=cfg.target_volume_maxs,
        scale=cfg.target_volume_region_scale,
    )
    pose = torch.cat([new_pos + env.scene.env_origins[env_ids], new_quat], dim=-1)
    env.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full per-env reset after ``super()._reset_idx``."""
    n = env_ids.numel()

    # Bank the finished episodes' successes under the mode they ran with (019).
    finished = env._successes[env_ids].detach().cpu()
    old_mode = env._reset_mode_per_env[env_ids].detach().cpu()
    for mode in RESET_MODE_NAMES:
        picked = finished[old_mode == mode]
        if picked.numel():
            env._mode_success_windows[mode].extend(picked.tolist())

    nearobj_ids, gair_ids, ggoal_ids, default_ids = _split_reset_modes(env, env_ids)
    _reset_table_pose(env, env_ids)
    if default_ids.numel() > 0:
        _randomize_robot_dof_state(env, default_ids)
        _reset_object_pose(env, default_ids)
    if nearobj_ids.numel() > 0:
        _reset_near_object_state(env, nearobj_ids)
    if gair_ids.numel() > 0:
        _reset_grasped_state(env, gair_ids)
    ggoal_pose_local = None
    if ggoal_ids.numel() > 0:
        ggoal_pose_local = _reset_grasped_state(env, ggoal_ids)
    env._reset_mode_per_env[env_ids] = 0
    env._reset_mode_per_env[nearobj_ids] = 1
    env._reset_mode_per_env[gair_ids] = 2
    env._reset_mode_per_env[ggoal_ids] = 3
    _reset_goal_pose(env, env_ids, mode="absolute")  # full reset → always absolute
    if ggoal_ids.numel() > 0:
        _reset_goal_from_object(env, ggoal_ids, ggoal_pose_local)

    env._prev_episode_successes[env_ids] = env._successes[env_ids]

    _clear_goal_trackers(env, env_ids)
    env._lifted_object[env_ids] = False
    # Grasped resets begin already holding the object, pre-latch so the
    # one-shot lift bonus cannot fire and keypoint progress pays from step 0.
    env._lifted_object[gair_ids] = True
    env._lifted_object[ggoal_ids] = True
    env._successes[env_ids] = 0

    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    for queue_name in ("_student_camera_queue", "_student_obs_queue"):
        if hasattr(env, queue_name):
            getattr(env, queue_name)[env_ids] = 0.0
    env._object_forces[env_ids] = 0.0
    env._object_torques[env_ids] = 0.0

    dr = env.cfg.domain_randomization
    env._random_force_prob[env_ids] = sample_log_uniform(
        dr.force_prob_range, n
    ).to(env.device)
    env._random_torque_prob[env_ids] = sample_log_uniform(
        dr.torque_prob_range, n
    ).to(env.device)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(
        n, 3, device=env.device
    ).uniform_(lo, hi)

    # Camera pose randomization (no-op when cfg.student_obs.use_camera_pose_rand is False).
    from .scene_utils import _apply_camera_pose_rand_at_reset
    _apply_camera_pose_rand_at_reset(env, env_ids)


__all__ = [
    "allocate_state_buffers",
    "reset_env_state",
    "reset_goal_trackers",
]
