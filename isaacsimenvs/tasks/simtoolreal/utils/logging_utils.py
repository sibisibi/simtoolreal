"""Task metric publishing for SimToolReal."""

from __future__ import annotations

import torch


def log_step_metrics(env) -> None:
    """Publish step-level extras consumed by RL-Games observers."""
    term_cfg = env.cfg.termination
    if term_cfg.max_consecutive_successes > 0:
        all_goals_hit = env._successes >= term_cfg.max_consecutive_successes
    else:
        all_goals_hit = torch.zeros_like(env._successes, dtype=torch.bool)

    episode_final = {
        "successes": env._successes.float(),
        "all_goals_hit": all_goals_hit.float(),
    }
    episode_final.update(
        {
            f"done_{name}": value.float()
            for name, value in env._termination_reasons.items()
        }
    )

    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)

    # Per-reset-mode channels (019). Scalars flow through direct_info.
    from .reset_utils import RESET_MODE_NAMES

    names: list[str] = []
    values: list[torch.Tensor] = []
    for mode, mode_name in RESET_MODE_NAMES.items():
        mask = env._reset_mode_per_env == mode
        names.append(f"reset_mode/env_frac_{mode_name}")
        values.append(mask.float().mean())
        if bool(mask.any()):
            for term in (
                "lifting_rew", "lift_bonus_rew", "keypoint_rew",
                "bonus_rew", "total_reward",
            ):
                names.append(f"reset_mode/{term}_{mode_name}")
                values.append(env._reward_terms[term][mask].mean())
            names.append(f"reset_mode/lifted_frac_{mode_name}")
            values.append(env._lifted_object[mask].float().mean())
    for name, value in zip(names, torch.stack(values).detach().cpu().tolist()):
        env.extras[name] = float(value)
    for mode, mode_name in RESET_MODE_NAMES.items():
        window = env._mode_success_windows[mode]
        if window:
            env.extras[f"reset_mode/successes_{mode_name}"] = (
                float(sum(window)) / len(window)
            )


__all__ = ["log_step_metrics"]
