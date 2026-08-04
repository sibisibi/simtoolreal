"""Rebuild the goal trajectory against a fresh eye-to-hand calibration.

Every base-frame number in the demo inherits the extrinsics that were current
when the tracking ran. A new solve changes them, and re-running FoundationPose
to find that out would waste minutes on an answer that is a matrix multiply.

Camera-frame poses are calibration independent, so the rebuild is

    poses_base = T_base_cam_new @ poses_cam

The canonical frame is a property of the mesh, not of the calibration, so its
transform is recovered from the existing pose pair and reused unchanged.

    /home/davian/anaconda3/envs/fp/bin/python \
        deployment/fr3_xhand/perception/recompose_goals.py \
        --demo deployment/fr3_xhand/demos/demo_20260803_081042
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_EXTRINSICS = Path(
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/calibration/"
    "results/extrinsics_347622076599_latest.npz"
)
GOAL_HZ = 3.0
CAPTURE_HZ = 30.0
LIFTOFF_M = 0.10


def pose_to_xyz_xyzw(t: np.ndarray) -> list[float]:
    return [*t[:3, 3].tolist(), *Rotation.from_matrix(t[:3, :3]).as_quat().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True)
    parser.add_argument("--extrinsics", default=str(LAB_EXTRINSICS))
    parser.add_argument("--liftoff_m", type=float, default=LIFTOFF_M)
    args = parser.parse_args()

    demo = Path(args.demo).resolve()
    poses_base_old = np.load(demo / "poses_base_30hz.npy")
    poses_canon_old = np.load(demo / "poses_base_30hz_canonical.npy")
    t_base_cam_new = np.load(args.extrinsics)["T_base_cam"]

    cam_path = demo / "poses_cam_30hz.npy"
    if cam_path.exists():
        poses_cam = np.load(cam_path)
    else:
        # Demos tracked before the camera-frame save existed. Undoing the
        # calibration that produced them is exact, so no re-track is needed.
        t_used = np.load(demo / "t_base_cam_used.npy")
        poses_cam = np.linalg.inv(t_used) @ poses_base_old
        np.save(cam_path, poses_cam)
        print(f"[recompose] backfilled {cam_path.name} from the calibration on record")

    # The canonical frame comes from the mesh, so the same transform applies to
    # any calibration. Recovering it keeps one definition rather than two.
    t_oc = np.linalg.inv(poses_base_old[0]) @ poses_canon_old[0]

    poses_base = t_base_cam_new @ poses_cam
    poses_canon = poses_base @ t_oc

    shift = np.linalg.norm(poses_base[:, :3, 3] - poses_base_old[:, :3, 3], axis=1)
    rel = np.matmul(np.transpose(poses_base_old[:, :3, :3], (0, 2, 1)), poses_base[:, :3, :3])
    turn = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    print(f"[recompose] calibration moved the poses {shift.mean() * 1000:.1f} mm mean, "
          f"{shift.max() * 1000:.1f} mm max, {turn.max():.2f} deg max")

    stride = int(round(CAPTURE_HZ / GOAL_HZ))
    goals = poses_canon[::stride]
    heights = goals[:, 2, 3]
    above = np.nonzero(heights - heights[0] > args.liftoff_m)[0]
    assert above.size > 0, (
        f"no goal rises {args.liftoff_m} m above the start height {heights[0]:.3f}, "
        "check the lift-off rule for this task"
    )
    goals = goals[above[0]:]

    np.save(demo / "poses_base_30hz.npy", poses_base)
    np.save(demo / "poses_base_30hz_canonical.npy", poses_canon)
    np.save(demo / "t_base_cam_used.npy", t_base_cam_new)

    traj_path = demo / "goal_trajectory.json"
    traj = json.loads(traj_path.read_text())
    traj["start_pose"] = pose_to_xyz_xyzw(poses_canon[0])
    traj["goals"] = [pose_to_xyz_xyzw(g) for g in goals]
    traj["liftoff_m"] = args.liftoff_m
    traj_path.write_text(json.dumps(traj, indent=2))
    print(f"[recompose] rewrote {len(goals)} goals in {traj_path}")
    print("[recompose] run goals_to_sim.py next to refresh the sim world copy")


if __name__ == "__main__":
    main()
