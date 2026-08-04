"""Track the demo object with FoundationPose and write the goal trajectory.

Consumes a recorded demo dir (rgb, depth, cam_K.txt, mask, picks.json) and a
metric mesh, registers at the trajectory start frame, tracks to the end frame,
transforms poses into the robot base frame with the lab calibration, then
downsamples to 3 Hz and truncates to the first goal above the table by the
paper's lift-off rule.

Poses leave here in the ROBOT BASE frame, xyz plus xyzw, matching what
goal_node publishes on /robot_frame/goal_object_pose.

Runs in the fp conda env, FoundationPose from the lab checkout.

    /home/davian/anaconda3/envs/fp/bin/python \
        deployment/fr3_xhand/perception/extract_goals.py \
        --demo deployment/fr3_xhand/demos/demo_20260803_081042 \
        --mesh deployment/fr3_xhand/demos/demo_20260803_081042/sam3d_output/mesh_scaled.obj
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, "/home/davian/kinamkim/fp/FoundationPose")

REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_EXTRINSICS = Path(
    "/home/davian/byungkunlee/davian_robotics_real3d/deploy/calibration/results/"
    "extrinsics_347622076599_latest.npz"
)

GOAL_HZ = 3.0
CAPTURE_HZ = 30.0
LIFTOFF_M = 0.10  # paper rule, trajectory starts at the first goal 10 cm up
# The 252-candidate registration overflows the free VRAM at full resolution,
# half resolution stays far inside the trained 1 cm pose noise budget.
DOWNSCALE = 2


def load_frame(demo: Path, idx: int):
    rgb = cv2.imread(str(demo / "rgb" / f"{idx:06d}.png"))
    depth = cv2.imread(str(demo / "depth" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
    assert rgb is not None and depth is not None, f"frame {idx} missing"
    h, w = depth.shape
    rgb = cv2.resize(rgb, (w // DOWNSCALE, h // DOWNSCALE), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, (w // DOWNSCALE, h // DOWNSCALE), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth.astype(np.float64) / 1000.0


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
    k[:2] /= DOWNSCALE
    t_base_cam = np.load(LAB_EXTRINSICS)["T_base_cam"]

    mesh = trimesh.load(args.mesh, force="mesh")
    print(f"[goals] mesh {args.mesh}, extents m {np.round(mesh.extents, 4)}")

    # Registration rasterizes 252 pose candidates, the full-detail mesh blows
    # the VRAM budget. Track on a decimated copy, pose accuracy does not need
    # bristle geometry.
    import open3d as o3d

    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3 = o3.simplify_quadric_decimation(target_number_of_triangles=20000)
    mesh = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles))
    print(f"[goals] decimated to {len(mesh.faces)} faces, extents m {np.round(mesh.extents, 4)}")

    import torch  # noqa: F401  FoundationPose imports expect torch first
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=0,
        # The upstream default points at the author's home directory.
        debug_dir="/tmp/fp_debug",
    )

    # The mask was drawn on the sam3d frame, valid at start because the user
    # confirmed the object did not move between the two frames.
    mask = cv2.imread(str(demo / f"mask_{picks['sam3d_frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    mask = cv2.resize(
        mask, (mask.shape[1] // DOWNSCALE, mask.shape[0] // DOWNSCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    rgb, depth = load_frame(demo, start)
    pose_cam = est.register(
        K=k, rgb=rgb, depth=depth, ob_mask=mask.astype(bool),
        iteration=args.register_iters,
    )

    poses_cam = [pose_cam]
    for idx in range(start + 1, end + 1):
        rgb, depth = load_frame(demo, idx)
        poses_cam.append(est.track_one(rgb=rgb, depth=depth, K=k, iteration=args.track_iters))
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


if __name__ == "__main__":
    main()
