"""Convert a base-frame goal trajectory into the sim's world frame.

The demo pipeline writes goals in the robot base frame, which is what the deploy
nodes publish. DexToolBench trajectories are in the sim world frame instead, so
a rehearsal needs the poses pushed through the contract's base transform.

Writes the trajectory where eval_isaacsim.py looks for it, which lets that
evaluator run against our object without any change.

    .venv_isaacsim/bin/python deployment/fr3_xhand/goals_to_sim.py \
        --object_spec deployment/fr3_xhand/objects/davian_handle_eraser.json \
        --category eraser --task wipe_up_down
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "deployment/fr3_xhand/contract/a4h1.json"
# Where handle_eraser starts in the benchmark's own trajectories, used as a
# workspace placement reference and nothing else. A start far from it says the
# object sits outside the region the checkpoint saw in training.
REFERENCE_START = np.array([0.0184, 0.0196, 0.6188])


def world_from_base(contract: dict) -> np.ndarray:
    w, x, y, z = contract["base_quat_wxyz"]
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    t[:3, 3] = contract["base_pos"]
    return t


def pose_from_xyz_xyzw(p) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(p[3:]).as_matrix()
    t[:3, 3] = p[:3]
    return t


def pose_to_xyz_xyzw(t: np.ndarray) -> list[float]:
    return [*t[:3, 3].tolist(), *Rotation.from_matrix(t[:3, :3]).as_quat().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_spec", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    spec = json.loads(Path(args.object_spec).read_text())
    traj = json.loads((REPO_ROOT / spec["goal_trajectory"]).read_text())
    assert traj["frame"] == "robot_base", f"expected base-frame goals, got {traj['frame']}"

    contract = json.loads(CONTRACT_PATH.read_text())
    t_world_base = world_from_base(contract)

    start = t_world_base @ pose_from_xyz_xyzw(traj["start_pose"])
    goals = [t_world_base @ pose_from_xyz_xyzw(g) for g in traj["goals"]]

    out_dir = REPO_ROOT / "dextoolbench/trajectories" / args.category / spec["object_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}.json"
    out_path.write_text(json.dumps(
        {"start_pose": pose_to_xyz_xyzw(start), "goals": [pose_to_xyz_xyzw(g) for g in goals]},
        indent=4,
    ))

    heights = np.array([g[2, 3] for g in goals])
    offset = start[:3, 3] - REFERENCE_START
    print(f"[goals_to_sim] wrote {len(goals)} goals to {out_path}")
    print(f"[goals_to_sim] start world xyz {np.round(start[:3, 3], 4)}")
    print(f"[goals_to_sim] offset from the handle_eraser start {np.round(offset, 4)} m")
    print(f"[goals_to_sim] goal height world z, min {heights.min():.4f} max {heights.max():.4f}")


if __name__ == "__main__":
    main()
