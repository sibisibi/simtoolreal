"""Deploy-side observation and target math for fr3-xhand.

Mirrors the sim exactly, build_observations in obs_utils.py for the deployed
subset and apply_action_pipeline in action_utils.py including the arm
post-EMA clamp. Verified against the live env by sim2sim_parity.py.

Frames. q, qd, prev_targets are canonical order radians. Object and goal
poses arrive in the robot base frame as xyz plus xyzw. The policy consumes
sim world coordinates, so every pose is mapped through the contract's
T_world_base first.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import Contract, Fk


def _base_pose_to_world(t_world_base: np.ndarray, pose_xyz_xyzw: np.ndarray):
    rot = Rotation.from_quat(pose_xyz_xyzw[3:7]).as_matrix()
    pos_w = t_world_base[:3, :3] @ pose_xyz_xyzw[:3] + t_world_base[:3, 3]
    rot_w = t_world_base[:3, :3] @ rot
    return pos_w, rot_w


def _keypoints(pos_w: np.ndarray, rot_w: np.ndarray, kp_offsets: np.ndarray) -> np.ndarray:
    return pos_w[None, :] + kp_offsets @ rot_w.T


def build_observation(
    contract: Contract,
    fk: Fk,
    q: np.ndarray,
    qd: np.ndarray,
    prev_targets: np.ndarray,
    object_pose_base: np.ndarray,
    goal_pose_base: np.ndarray,
    object_scales: np.ndarray,
) -> np.ndarray:
    n = contract.num_joints
    assert q.shape == (n,) and qd.shape == (n,) and prev_targets.shape == (n,)
    assert object_pose_base.shape == (7,) and goal_pose_base.shape == (7,)
    assert object_scales.shape == (3,)

    joint_pos = 2.0 * (q - contract.joint_lower) / (
        contract.joint_upper - contract.joint_lower
    ) - 1.0

    t_wb = contract.t_world_base
    fk_dict = fk(q)

    t_palm = fk_dict[contract.palm_body]
    palm_rot_w = t_wb[:3, :3] @ t_palm[:3, :3]
    palm_body_pos_w = t_wb[:3, :3] @ t_palm[:3, 3] + t_wb[:3, 3]
    palm_pos = palm_body_pos_w + palm_rot_w @ contract.palm_center_offset
    palm_rot_xyzw = Rotation.from_matrix(palm_rot_w).as_quat()

    fingertips = np.empty((len(contract.fingertip_bodies), 3))
    for i, name in enumerate(contract.fingertip_bodies):
        t_ft = fk_dict[name]
        ft_rot_w = t_wb[:3, :3] @ t_ft[:3, :3]
        ft_pos_w = t_wb[:3, :3] @ t_ft[:3, 3] + t_wb[:3, 3]
        fingertips[i] = ft_pos_w + ft_rot_w @ contract.fingertip_offsets[i]
    fingertip_pos_rel_palm = fingertips - palm_pos[None, :]

    obj_pos_w, obj_rot_w = _base_pose_to_world(t_wb, object_pose_base)
    goal_pos_w, goal_rot_w = _base_pose_to_world(t_wb, goal_pose_base)
    obj_rot_xyzw = Rotation.from_matrix(obj_rot_w).as_quat()

    kp_offsets = contract.keypoint_corners * (object_scales * contract.keypoint_factor)
    obj_kp = _keypoints(obj_pos_w, obj_rot_w, kp_offsets)
    goal_kp = _keypoints(goal_pos_w, goal_rot_w, kp_offsets)

    fields = {
        "joint_pos": joint_pos,
        "joint_vel": qd,
        "prev_action_targets": prev_targets,
        "palm_pos": palm_pos,
        "palm_rot": palm_rot_xyzw,
        "object_rot": obj_rot_xyzw,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "keypoints_rel_palm": obj_kp - palm_pos[None, :],
        "keypoints_rel_goal": obj_kp - goal_kp,
        "object_scales": object_scales,
    }
    assert set(contract.obs_list) == set(fields)

    obs = np.concatenate([np.asarray(fields[f]).reshape(-1) for f in contract.obs_list])
    assert obs.shape == (contract.num_observations,)
    clip = contract.clamp_abs_observations
    return np.clip(obs, -clip, clip)


def compute_targets(
    contract: Contract, actions: np.ndarray, prev_targets: np.ndarray
) -> np.ndarray:
    n_arm = contract.num_arm
    assert actions.shape == (contract.num_actions,)
    assert prev_targets.shape == (contract.num_actions,)
    # rl_games clips with clip_actions 1.0 before the env sees the action.
    actions = np.clip(actions, -1.0, 1.0)

    lo, hi = contract.joint_lower, contract.joint_upper
    arm_raw = prev_targets[:n_arm] + contract.dof_speed_scale * contract.control_dt * actions[:n_arm]
    arm_raw = np.clip(arm_raw, lo[:n_arm], hi[:n_arm])
    arm = contract.arm_moving_average * arm_raw + (1.0 - contract.arm_moving_average) * prev_targets[:n_arm]
    arm = np.clip(arm, lo[:n_arm], hi[:n_arm])

    hand_raw = lo[n_arm:] + 0.5 * (actions[n_arm:] + 1.0) * (hi[n_arm:] - lo[n_arm:])
    hand = contract.hand_moving_average * hand_raw + (1.0 - contract.hand_moving_average) * prev_targets[n_arm:]
    hand = np.clip(hand, lo[n_arm:], hi[n_arm:])

    return np.concatenate([arm, hand])
