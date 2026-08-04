"""Gate for the live perception node, replay a demo through the node's own path.

The node and the offline goal extraction share tracker.py, so this replays the
recorded frames through the shared tracker and the node's message packing, then
compares against the poses the goal trajectory was built from.

Two modes, chosen by which mesh is passed.

  mesh_scaled.obj     compares to poses_base_30hz.npy, guards the refactor
  mesh_canonical.obj  compares to poses_base_30hz_canonical.npy, the frame the
                      node publishes, tolerance is the trained 1 cm and 5 deg

Tracking is a recursive filter and FoundationPose is not bit reproducible, so
two identical runs already disagree. Measured over this demo, run against run
came to 0.66 mm and 0.68 deg at worst, and the strict tolerance sits three
times above that floor. A preprocessing slip, a wrong intrinsic scale or a
swapped colour order, misses by far more than the margin.

Also reports per-frame tracking latency, which is the compute half of the
capture-to-publish budget the live node has to hold under 100 ms.

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davian/anaconda3/envs/fp/lib \
        /home/davian/anaconda3/envs/fp/bin/python \
        deployment/fr3_xhand/perception/replay_parity.py \
        --demo deployment/fr3_xhand/demos/demo_20260803_081042 \
        --mesh .../sam3d_output/mesh_canonical.obj
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from builtin_interfaces.msg import Time
from scipy.spatial.transform import Rotation

PERCEPTION = Path(__file__).resolve().parent
REPO_ROOT = PERCEPTION.parents[2]
sys.path.insert(0, str(PERCEPTION))
sys.path.insert(0, str(REPO_ROOT / "deployment/fr3_xhand/ws/src/fr3_xhand_nodes"))

from tracker import ObjectTracker  # noqa: E402
from fr3_xhand_nodes.perception_node import LAB_EXTRINSICS, pose_to_msg  # noqa: E402

TOLERANCES = {
    "mesh_scaled.obj": ("poses_base_30hz.npy", 2e-3, 2.0),
    "mesh_canonical.obj": ("poses_base_30hz_canonical.npy", 1e-2, 5.0),
    # The asset mesh is the canonical one decimated to the benchmark's density,
    # and it is what the object spec points the live node at.
    "davian_handle_eraser.obj": ("poses_base_30hz_canonical.npy", 1e-2, 5.0),
}
LATENCY_BUDGET_MS = 100.0


def msg_to_pose(msg) -> np.ndarray:
    t = np.eye(4)
    q = msg.pose.orientation
    t[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    t[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    return t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out_poses", default="", help="save the replayed poses for a run to run check")
    args = parser.parse_args()

    demo = Path(args.demo).resolve()
    reference_name, tol_m, tol_deg = TOLERANCES[Path(args.mesh).name]
    reference = np.load(demo / reference_name)
    picks = json.loads((demo / "picks.json").read_text())
    start, end = picks["trajectory_start_frame"], picks["trajectory_end_frame"]
    assert len(reference) == end - start + 1, (
        f"{reference_name} holds {len(reference)} poses, the demo spans {end - start + 1}"
    )

    k = np.loadtxt(demo / "cam_K.txt")
    t_base_cam = np.load(LAB_EXTRINSICS)["T_base_cam"]
    tracker = ObjectTracker(args.mesh, k)

    def frame(idx):
        rgb = cv2.imread(str(demo / "rgb" / f"{idx:06d}.png"))
        depth = cv2.imread(str(demo / "depth" / f"{idx:06d}.png"), cv2.IMREAD_UNCHANGED)
        assert rgb is not None and depth is not None, f"frame {idx} missing"
        return rgb, depth

    mask = cv2.imread(str(demo / f"mask_{picks['sam3d_frame']:06d}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    rgb, depth = frame(start)
    poses = [t_base_cam @ tracker.register(rgb, depth, mask)]

    latencies = []
    for idx in range(start + 1, end + 1):
        rgb, depth = frame(idx)
        t0 = time.perf_counter()
        pose_base = t_base_cam @ tracker.track(rgb, depth)
        latencies.append((time.perf_counter() - t0) * 1e3)
        # Round tripping through the message catches a packing or convention slip.
        poses.append(msg_to_pose(pose_to_msg(pose_base, Time())))
        if (idx - start) % 150 == 0:
            print(f"[parity] tracked {idx - start} of {end - start}")

    poses = np.stack(poses)
    if args.out_poses:
        np.save(args.out_poses, poses)
    trans = np.linalg.norm(poses[:, :3, 3] - reference[:, :3, 3], axis=1)
    rel = np.matmul(np.transpose(reference[:, :3, :3], (0, 2, 1)), poses[:, :3, :3])
    rot_deg = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    lat = np.array(latencies)

    print(f"\n[parity] reference {reference_name}, {len(poses)} poses")
    print(f"[parity] translation m, mean {trans.mean():.6f}, max {trans.max():.6f}, tol {tol_m}")
    print(f"[parity] rotation deg, mean {rot_deg.mean():.4f}, max {rot_deg.max():.4f}, tol {tol_deg}")
    print(
        f"[parity] track latency ms, p50 {np.percentile(lat, 50):.1f}, "
        f"p95 {np.percentile(lat, 95):.1f}, max {lat.max():.1f}, budget {LATENCY_BUDGET_MS}"
    )

    assert trans.max() < tol_m, f"translation error {trans.max():.6f} m exceeds {tol_m}"
    assert rot_deg.max() < tol_deg, f"rotation error {rot_deg.max():.4f} deg exceeds {tol_deg}"
    assert np.percentile(lat, 95) < LATENCY_BUDGET_MS, "p95 track latency exceeds the budget"
    print("[parity] PASS")


if __name__ == "__main__":
    main()
