"""Observation assembly and step-shared geometry caches for SimToolReal."""

from __future__ import annotations

import math
import os

import torch

from isaaclab.utils.math import convert_quat, quat_apply, quat_from_angle_axis, quat_mul


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------


NUM_JOINTS: int = 19  # FR3 (7) + XHand1 right (12)
NUM_FINGERTIPS: int = 5
NUM_KEYPOINTS: int = 4

# Offset from `fr3_link7` (the wrist link the `palm` merged into) toward the
# grasp center, ~0.16 m along the flange axis (analogous to the original iiwa
# wrist offset). TODO(032): untuned for XHand1; refine against rendered frames.
PALM_CENTER_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.16)

# Fingertip-reward geometry. 009-pd-reward-sweep crosses this with the hand PD.
# distal (current): the *_link2 distal-phalanx origin, offset (0,0,0).
# pad: XHand's own fingertip pads, the per-finger *_joint3 origins from the
# verified v1.3 URDF. merge_fixed_joints folds each fixed *_tip into its
# *_link2 parent, so a constant local offset equals the true pad position
# exactly. Keyed by body name so the order always matches
# env._fingertip_body_names (== _fingertip_body_ids order). Selected by env
# var, default distal == current behavior.
FINGERTIP_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.0)
FINGERTIP_PAD_OFFSET_BY_BODY: dict[str, tuple[float, float, float]] = {
    "thumb_rota_link2": (0.0, 0.0502276499414863, 0.0),
    "index_rota_link2": (0.0, 0.0, 0.0422482924089424),
    "mid_link2": (0.0, 0.0, 0.042248),
    "ring_link2": (0.0, 0.0, 0.0422482924089404),
    "pinky_link2": (0.0, 0.0, 0.0422482924089405),
}
_FT_TARGET = os.environ.get("XHAND_FT_TARGET", "pad")
assert _FT_TARGET in ("distal", "pad"), (
    f"XHAND_FT_TARGET={_FT_TARGET!r} not in ('distal', 'pad')"
)

# Object-frame keypoint corners before scaling.
KEYPOINT_CORNERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (-1, -1, -1),
)

OBS_FIELD_SIZES: dict[str, int] = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "palm_vel": 6,
    "object_rot": 4,
    "object_vel": 6,
    "fingertip_pos_rel_palm": 3 * NUM_FINGERTIPS,  # 15
    "keypoints_rel_palm": 3 * NUM_KEYPOINTS,  # 12
    "keypoints_rel_goal": 3 * NUM_KEYPOINTS,  # 12
    "object_scales": 3,
    "closest_keypoint_max_dist": 1,
    "closest_fingertip_dist": NUM_FINGERTIPS,  # 5
    "lifted_object": 1,
    "progress": 1,
    "successes": 1,
    "reward": 1,
}


def compute_obs_dim(field_list) -> int:
    """Return total tensor dim for an ordered list of obs field names."""
    return sum(OBS_FIELD_SIZES[f] for f in field_list)


def _stack_obs_dict(obs_dict: dict[str, torch.Tensor], field_list) -> torch.Tensor:
    """Concatenate named tensors in config order."""
    return torch.cat(
        [obs_dict[f].reshape(obs_dict[f].shape[0], -1) for f in field_list],
        dim=-1,
    )


# ----------------------------------------------------------------------------
# Quaternion / keypoint helpers
# ----------------------------------------------------------------------------


def _perturb_quat(q_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """Apply random-axis rotation noise to wxyz quaternions."""
    n = q_wxyz.shape[0]
    axis = torch.nn.functional.normalize(
        torch.randn(n, 3, device=q_wxyz.device), dim=-1
    )
    angle = torch.empty(n, device=q_wxyz.device).uniform_(
        -max_deg, max_deg
    ) * (math.pi / 180.0)
    dq = quat_from_angle_axis(angle, axis)
    return quat_mul(dq, q_wxyz)


def _apply_local_offset(
    pos_w: torch.Tensor,
    rot_wxyz: torch.Tensor,
    offset,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    """Apply a local-frame offset to batched world poses.

    `offset` is either a shared (3,) vector applied to every pose, or a
    per-element (M, 3) array whose M matches the last batch dim (the 5
    fingertips). 009 uses the per-finger form for the pad target.
    """
    offset_t = torch.as_tensor(offset, device=pos_w.device, dtype=pos_w.dtype)
    if offset_t.ndim == 1:
        offset_t = offset_t.expand(*batch_shape, 3)
    elif offset_t.ndim == 2:
        assert offset_t.shape[0] == batch_shape[-1], (
            f"per-element offset {tuple(offset_t.shape)} vs batch {batch_shape}"
        )
        offset_t = offset_t.unsqueeze(0).expand(*batch_shape, 3)
    else:
        raise ValueError(f"offset must be (3,) or (M, 3), got {tuple(offset_t.shape)}")
    shifted = quat_apply(
        rot_wxyz.reshape(-1, 4), offset_t.reshape(-1, 3)
    ).reshape(*batch_shape, 3)
    return pos_w + shifted


def _fingertip_offset(env, device, dtype):
    """Per-finger local offset for the active fingertip-reward target.

    distal -> shared (0,0,0). pad -> (5,3) ordered to env._fingertip_body_names
    (== _fingertip_body_ids order). Built once and cached on the env.
    """
    cached = getattr(env, "_fingertip_offset_cached", None)
    if cached is not None and cached.device == device:
        return cached
    if _FT_TARGET == "pad":
        rows = []
        for name in env._fingertip_body_names:
            assert name in FINGERTIP_PAD_OFFSET_BY_BODY, (
                f"no pad offset for fingertip body {name!r}"
            )
            rows.append(FINGERTIP_PAD_OFFSET_BY_BODY[name])
        cached = torch.tensor(rows, device=device, dtype=dtype)  # (5, 3)
    else:
        cached = torch.as_tensor(FINGERTIP_OFFSET, device=device, dtype=dtype)
    env._fingertip_offset_cached = cached
    return cached


def _keypoints_world(
    center_pos: torch.Tensor,    # (N, 3)
    center_rot: torch.Tensor,    # (N, 4) wxyz
    kp_offsets: torch.Tensor,    # (N, K, 3)
) -> torch.Tensor:
    """Rotate + translate object-frame keypoints."""
    n_envs, k, _ = kp_offsets.shape
    rot_r = center_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)
    offsets_r = kp_offsets.reshape(-1, 3)
    return center_pos.unsqueeze(1) + quat_apply(rot_r, offsets_r).reshape(n_envs, k, 3)


def _episode_start(env) -> torch.Tensor:
    return (env.episode_length_buf == 0) & (env._successes == 0)


def _sample_delay(
    queue: torch.Tensor,
    values: torch.Tensor,
    env,
    flush: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push current values into a rolling queue and sample per-env delay."""
    if flush is not None and flush.any():
        queue[flush] = values[flush].unsqueeze(1).expand(-1, queue.shape[1], -1)

    queue = torch.roll(queue, shifts=1, dims=1)
    queue[:, 0, :] = values
    idx = torch.randint(0, queue.shape[1], (env.num_envs,), device=env.device)
    delayed = queue[torch.arange(env.num_envs, device=env.device), idx]
    return queue, delayed


def _canonical_joint_obs(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy-order joint pos, vel, and previous targets."""
    perm = env._perm_lab_to_canon
    joint_pos_raw = env.robot.data.joint_pos[:, perm]
    joint_pos = (
        2.0 * (joint_pos_raw - env._joint_lower_canon)
        / (env._joint_upper_canon - env._joint_lower_canon)
        - 1.0
    )
    return joint_pos, env.robot.data.joint_vel[:, perm], env._prev_targets[:, perm]


# ----------------------------------------------------------------------------
# Step-shared intermediate values (feeds _get_dones + _get_rewards)
# ----------------------------------------------------------------------------


def compute_intermediate_values(env) -> None:
    """Update shared geometric state for rewards and terminations."""
    from .reward_utils import update_near_goal_steps  # local import to avoid cycle

    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w
    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]
    # 009: shift to the active fingertip-reward target (distal origin or pad)
    # before measuring fingertip-object distance, the signal behind
    # fingertip_delta_rew and closest_fingertip_dist.
    ft_pos = _apply_local_offset(
        ft_state[:, :, 0:3],
        ft_state[:, :, 3:7],
        _fingertip_offset(env, ft_state.device, ft_state.dtype),
        (env.num_envs, NUM_FINGERTIPS),
    ) - env_origins.unsqueeze(1)
    env._curr_fingertip_distances = torch.norm(
        ft_pos - obj_pos.unsqueeze(1), dim=-1
    )  # (N, 5)

    if rew_cfg.fixed_size_keypoint_reward:
        kp_offsets = env._keypoint_offsets_fixed
    else:
        kp_offsets = env._keypoint_offsets

    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)

    env._keypoints_max_dist = torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values

    # Legacy -1 sentinel: first observed value becomes closest-so-far.
    sentinel = env._closest_keypoint_max_dist < 0.0
    env._closest_keypoint_max_dist = torch.where(
        sentinel, env._keypoints_max_dist, env._closest_keypoint_max_dist
    )
    sentinel_ft = env._closest_fingertip_dist < 0.0
    env._closest_fingertip_dist = torch.where(
        sentinel_ft, env._curr_fingertip_distances, env._closest_fingertip_dist
    )

    if hasattr(env, "_keypoint_success_tolerance_m"):
        tol = env._keypoint_success_tolerance_m()
    else:
        tol = env._current_success_tolerance * rew_cfg.keypoint_scale
    env._near_goal = env._keypoints_max_dist <= tol
    env._near_goal_steps = update_near_goal_steps(
        near_goal=env._near_goal,
        near_goal_steps=env._near_goal_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )
    env._is_success = env._near_goal_steps >= term_cfg.success_steps


# ----------------------------------------------------------------------------
# Observation builder (Phase D)
# ----------------------------------------------------------------------------


def _apply_object_state_dr(env, obj_pos, obj_rot, obj_linvel, obj_angvel):
    """Apply object-state delay and pose noise."""
    dr = env.cfg.domain_randomization
    state = torch.cat([obj_pos, obj_rot, obj_linvel, obj_angvel], dim=-1)
    env._object_state_queue, delayed = _sample_delay(
        env._object_state_queue, state, env, flush=_episode_start(env)
    )
    noisy_pos = delayed[:, 0:3] + torch.randn_like(delayed[:, 0:3]) * dr.object_state_xyz_noise_std
    noisy_rot = _perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
    noisy_vel = delayed[:, 7:13]
    return noisy_pos, noisy_rot, noisy_vel


def _apply_obs_delay(env, policy_tensor: torch.Tensor) -> torch.Tensor:
    """Apply per-env policy-observation delay."""
    env._obs_queue, delayed = _sample_delay(
        env._obs_queue, policy_tensor, env, flush=_episode_start(env)
    )
    return delayed


def build_observations(env) -> dict[str, torch.Tensor]:
    """Assemble actor-critic observations with obs-side DR."""
    dr = env.cfg.domain_randomization
    env_origins = env.scene.env_origins

    joint_pos, joint_vel, prev_targets_canon = _canonical_joint_obs(env)

    palm_state = env.robot.data.body_state_w[:, env._palm_body_id, :]  # (N, 13)
    palm_pos_w = palm_state[:, 0:3]
    palm_rot = palm_state[:, 3:7]  # wxyz (Isaac Lab convention)
    palm_vel = palm_state[:, 7:13]

    palm_center_pos_w = _apply_local_offset(
        palm_pos_w, palm_rot, PALM_CENTER_OFFSET, (env.num_envs,)
    )
    palm_pos = palm_center_pos_w - env_origins

    ft_state = env.robot.data.body_state_w[:, env._fingertip_body_ids, :]  # (N, 5, 13)
    ft_body_pos_w = ft_state[:, :, 0:3]
    ft_body_rot_w = ft_state[:, :, 3:7]  # wxyz

    ft_pos_w = _apply_local_offset(
        ft_body_pos_w,
        ft_body_rot_w,
        _fingertip_offset(env, ft_body_pos_w.device, ft_body_pos_w.dtype),
        (env.num_envs, NUM_FINGERTIPS),
    )

    obj_pos = env.object.data.root_pos_w - env_origins
    obj_rot = env.object.data.root_quat_w  # wxyz
    obj_linvel = env.object.data.root_lin_vel_w
    obj_angvel = env.object.data.root_ang_vel_w
    obj_vel = torch.cat([obj_linvel, obj_angvel], dim=-1)

    goal_pos = env.goal_viz.data.root_pos_w - env_origins
    goal_rot = env.goal_viz.data.root_quat_w  # wxyz

    if dr.use_object_state_delay_noise:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = _apply_object_state_dr(
            env, obj_pos, obj_rot, obj_linvel, obj_angvel
        )
    else:
        noisy_obj_pos, noisy_obj_rot, noisy_obj_vel = obj_pos, obj_rot, obj_vel

    kp_offsets = env._keypoint_offsets * env._object_scale_multiplier.unsqueeze(1)
    obj_kp = _keypoints_world(obj_pos, obj_rot, kp_offsets)
    goal_kp = _keypoints_world(goal_pos, goal_rot, kp_offsets)
    noisy_obj_kp = _keypoints_world(noisy_obj_pos, noisy_obj_rot, kp_offsets)

    # Optional per-env yaw noise on the observed goal (world +Z about goal_pos).
    goal_yaw_obs_noise = getattr(env, "goal_yaw_obs_noise", None)
    if goal_yaw_obs_noise is not None and torch.any(goal_yaw_obs_noise != 0):
        z_axis = torch.tensor(
            [0.0, 0.0, 1.0], device=goal_rot.device, dtype=goal_rot.dtype
        ).unsqueeze(0).expand(env.num_envs, -1)
        yaw_q = quat_from_angle_axis(goal_yaw_obs_noise, z_axis)
        noisy_goal_rot = quat_mul(yaw_q, goal_rot)
        noisy_goal_kp = _keypoints_world(goal_pos, noisy_goal_rot, kp_offsets)
    else:
        noisy_goal_kp = goal_kp

    keypoints_rel_palm_clean = obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_palm_noisy = noisy_obj_kp - palm_pos.unsqueeze(1)
    keypoints_rel_goal_clean = obj_kp - goal_kp
    keypoints_rel_goal_noisy = noisy_obj_kp - noisy_goal_kp

    fingertip_pos_rel_palm = (
        (ft_pos_w - env_origins.unsqueeze(1)) - palm_pos.unsqueeze(1)
    )  # (N, 5, 3)

    object_scales_obs = env._object_scale_per_env * env._object_scale_multiplier

    # Policy obs use legacy Isaac Gym xyzw; internal math stays wxyz.
    palm_rot_xyzw = convert_quat(palm_rot, to="xyzw")
    obj_rot_xyzw = convert_quat(obj_rot, to="xyzw")
    noisy_obj_rot_xyzw = convert_quat(noisy_obj_rot, to="xyzw")

    obs_clean: dict[str, torch.Tensor] = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets_canon,
        "palm_pos": palm_pos,
        "palm_rot": palm_rot_xyzw,
        "palm_vel": palm_vel,
        "object_rot": obj_rot_xyzw,
        "object_vel": obj_vel,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "keypoints_rel_palm": keypoints_rel_palm_clean,
        "keypoints_rel_goal": keypoints_rel_goal_clean,
        "object_scales": object_scales_obs,
        "closest_keypoint_max_dist": env._closest_keypoint_max_dist.unsqueeze(-1),
        "closest_fingertip_dist": env._closest_fingertip_dist,
        "lifted_object": env._lifted_object.float().unsqueeze(-1),
        "progress": torch.log(env.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
        "successes": torch.log(env._successes.float() + 1.0).unsqueeze(-1),
        "reward": (env.reward_buf * 0.01).unsqueeze(-1),
    }

    obs_noisy = dict(obs_clean)
    obs_noisy["object_rot"] = noisy_obj_rot_xyzw
    obs_noisy["object_vel"] = noisy_obj_vel
    obs_noisy["keypoints_rel_palm"] = keypoints_rel_palm_noisy
    obs_noisy["keypoints_rel_goal"] = keypoints_rel_goal_noisy
    if dr.joint_velocity_obs_noise_std > 0:
        obs_noisy["joint_vel"] = (
            joint_vel + torch.randn_like(joint_vel) * dr.joint_velocity_obs_noise_std
        )

    state_tensor = _stack_obs_dict(obs_clean, env.cfg.obs.state_list)
    policy_tensor = _stack_obs_dict(obs_noisy, env.cfg.obs.obs_list)

    if dr.use_obs_delay:
        policy_tensor = _apply_obs_delay(env, policy_tensor)

    clip = env.cfg.obs.clamp_abs_observations
    policy_tensor = policy_tensor.clamp(-clip, clip)
    state_tensor = state_tensor.clamp(-clip, clip)

    return {"policy": policy_tensor, "critic": state_tensor}


def _student_proprio_dict(env) -> dict[str, torch.Tensor]:
    joint_pos, joint_vel, prev_targets_canon = _canonical_joint_obs(env)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "prev_action_targets": prev_targets_canon,
    }


def _apply_student_tensor_delay(
    env,
    values: torch.Tensor,
    *,
    queue_attr: str,
    delay_max: int,
    enabled: bool,
) -> torch.Tensor:
    if not enabled or delay_max <= 0:
        return values

    flat_values = values.reshape(env.num_envs, -1)
    queue_len = max(1, int(delay_max))
    queue = getattr(env, queue_attr, None)
    expected_shape = (env.num_envs, queue_len, flat_values.shape[-1])
    if (
        queue is None
        or tuple(queue.shape) != expected_shape
        or queue.device != flat_values.device
        or queue.dtype != flat_values.dtype
    ):
        queue = flat_values.unsqueeze(1).expand(-1, queue_len, -1).clone()

    queue, delayed = _sample_delay(
        queue,
        flat_values,
        env,
        flush=_episode_start(env),
    )
    setattr(env, queue_attr, queue)
    return delayed.reshape_as(values)


def _apply_student_bundle_delay(
    env,
    student_obs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    cfg = env.cfg.student_obs
    if not cfg.use_student_obs_delay or cfg.student_obs_delay_max <= 0:
        return student_obs

    keys = [key for key in ("image", "proprio") if key in student_obs]
    flat_parts = [student_obs[key].reshape(env.num_envs, -1) for key in keys]
    sizes = [part.shape[-1] for part in flat_parts]
    packed = torch.cat(flat_parts, dim=-1)
    delayed = _apply_student_tensor_delay(
        env,
        packed,
        queue_attr="_student_obs_queue",
        delay_max=int(cfg.student_obs_delay_max),
        enabled=True,
    )

    out = dict(student_obs)
    offset = 0
    for key, size in zip(keys, sizes):
        tensor = student_obs[key]
        out[key] = delayed[:, offset : offset + size].reshape_as(tensor)
        offset += size
    return out


def build_student_observations(env) -> dict[str, torch.Tensor]:
    """Assemble optional image/proprio observations for distillation students."""
    cfg = env.cfg.student_obs
    if not cfg.enabled:
        raise RuntimeError(
            "cfg.student_obs.enabled is false; student observations are disabled."
        )

    proprio_fields = tuple(cfg.proprio_list)
    proprio_dict = _student_proprio_dict(env)
    unsupported = [field for field in proprio_fields if field not in proprio_dict]
    if unsupported:
        raise ValueError(
            f"Unsupported cfg.student_obs.proprio_list fields: {unsupported}. "
            f"Supported fields: {sorted(proprio_dict)}."
        )

    if proprio_fields:
        proprio = torch.cat(
            [proprio_dict[field].reshape(env.num_envs, -1) for field in proprio_fields],
            dim=-1,
        )
    else:
        proprio = torch.empty(env.num_envs, 0, device=env.device)

    student_obs = {"proprio": proprio}
    if cfg.image_enabled:
        from .scene_utils import read_student_camera_image

        image = read_student_camera_image(env)
        student_obs["image"] = _apply_student_tensor_delay(
            env,
            image,
            queue_attr="_student_camera_queue",
            delay_max=int(cfg.camera_delay_max),
            enabled=bool(cfg.use_camera_delay),
        )
    return _apply_student_bundle_delay(env, student_obs)


__all__ = [
    "NUM_JOINTS",
    "NUM_FINGERTIPS",
    "NUM_KEYPOINTS",
    "KEYPOINT_CORNERS",
    "OBS_FIELD_SIZES",
    "compute_obs_dim",
    "compute_intermediate_values",
    "build_observations",
    "build_student_observations",
]
