"""Track the demo object with FoundationPose and write the goal trajectory.

Consumes a recorded demo dir (rgb, depth, cam_K.txt, mask, picks.json) and a
metric mesh, registers at the trajectory start frame, tracks to the end frame,
transforms poses into the robot base frame with the lab calibration, then
downsamples to 3 Hz and truncates to the first goal above the table by the
paper's lift-off rule.

Pass the CANONICAL mesh, the one the live node and the sim load. FoundationPose
gives a different answer per mesh file, because it recentres on the axis aligned
bounding box and voxel downsamples on the mesh's own axes. Tracking the raw
reconstruction here and rotating the poses afterwards lands 8 mm and 7 deg away
from tracking the canonical mesh directly, on a stationary object.

Poses leave here in the ROBOT BASE frame and the canonical object frame, xyz
plus xyzw, matching what goal_node publishes on /robot_frame/goal_object_pose.

Runs in the fp conda env, FoundationPose from the lab checkout.

    /home/davian/anaconda3/envs/fp/bin/python \
        deployment/fr3_xhand/perception/extract_goals.py \
        --demo deployment/fr3_xhand/demos/demo_20260803_081042 \
        --mesh assets/urdf/davian/davian_handle_eraser/davian_handle_eraser.obj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker import ObjectTracker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_EXTRINSICS = Path(
    "/home/davian/byungkunlee/davian_robotics_real3d/deploy/calibration/results/"
    "extrinsics_347622076599_latest.npz"
)

GOAL_HZ = 3.0
CAPTURE_HZ = 30.0
# Paper rule, appendix E, lift-off truncation at 10 cm above the resting pose.
LIFTOFF_M = 0.10


def load_frame(demo: Path, idx: int):
    """Full resolution BGR and uint16 mm depth, exactly as the camera gives them."""
    rgb = cv2.imread(str(demo / "rgb" / f"{idx:06d}.png"))
    depth = cv2.imread(str(demo / "depth" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    assert rgb is not None and depth is not None, f"frame {idx} missing"
    return rgb, depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--register_iters", type=int, default=5)
    parser.add_argument("--track_iters", type=int, default=2)
    args = parser.parse_args()

    demo = Path(args.demo).resolve()
    picks = json.loads((demo / "picks.json").read_text())
    start = picks["trajectory_start_frame"]
    end = picks["trajectory_end_frame"]
    k = np.loadtxt(demo / "cam_K.txt")
    t_base_cam = np.load(LAB_EXTRINSICS)["T_base_cam"]

    tracker = ObjectTracker(args.mesh, k)

    # The mask was drawn on the sam3d frame, valid at start because the user
    # confirmed the object did not move between the two frames.
    mask = cv2.imread(str(demo / f"mask_{picks['sam3d_frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    rgb, depth = load_frame(demo, start)
    pose_cam = tracker.register(rgb, depth, mask, iteration=args.register_iters)

    poses_cam = [pose_cam]
    for idx in range(start + 1, end + 1):
        rgb, depth = load_frame(demo, idx)
        poses_cam.append(tracker.track(rgb, depth, iteration=args.track_iters))
        if (idx - start) % 150 == 0:
            print(f"[goals] tracked {idx - start} of {end - start}")

    poses_base = [t_base_cam @ np.asarray(p) for p in poses_cam]

    stride = int(round(CAPTURE_HZ / GOAL_HZ))
    goals = poses_base[::stride]

    heights = np.array([g[2, 3] for g in goals])
    table_z = heights[0]
    above = np.nonzero(heights - table_z > LIFTOFF_M)[0]
    assert above.size > 0, (
        f"no goal rises {LIFTOFF_M} m above the start height {table_z:.3f}, "
        "check tracking or the lift-off rule for this task"
    )
    goals = goals[above[0]:]

    def pose_to_xyz_xyzw(t):
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(t[:3, :3]).as_quat()
        return [*t[:3, 3].tolist(), *q.tolist()]

    out = {
        "frame": "robot_base",
        "source_demo": str(demo.relative_to(REPO_ROOT)),
        "goal_hz": GOAL_HZ,
        "liftoff_m": LIFTOFF_M,
        "start_pose": pose_to_xyz_xyzw(poses_base[0]),
        "goals": [pose_to_xyz_xyzw(g) for g in goals],
    }
    traj_path = demo / "goal_trajectory.json"
    traj_path.write_text(json.dumps(out, indent=2))
    print(f"[goals] wrote {len(goals)} goals to {traj_path}")

    np.save(demo / "poses_base_30hz.npy", np.stack(poses_base))
    print("[goals] full 30 Hz base-frame poses saved for inspection")

    # Camera-frame poses do not depend on the calibration, so a fresh solve
    # rebuilds everything downstream with recompose_goals.py instead of another
    # tracking pass. The calibration used is saved beside them, because the lab
    # tool overwrites its latest file in place.
    np.save(demo / "poses_cam_30hz.npy", np.stack([np.asarray(p) for p in poses_cam]))
    np.save(demo / "t_base_cam_used.npy", t_base_cam)
    print("[goals] camera-frame poses and the calibration used saved for recompose")


if __name__ == "__main__":
    main()
