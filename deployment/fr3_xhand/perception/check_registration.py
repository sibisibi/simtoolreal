"""Draw the registered pose over the init frame so the lock can be eyeballed.

A registration that latches onto the wrong face is invisible in the numbers and
obvious in a picture, and it stays invisible until the arm moves. Run this after
init_scene.py and look at the overlay before starting the loop.

Also prints the table surface height implied by the pose, which is the sim
geometry check, the contract puts the surface level with the robot base.

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davian/anaconda3/envs/fp/lib \
        /home/davian/anaconda3/envs/fp/bin/python \
        deployment/fr3_xhand/perception/check_registration.py \
        --init deployment/fr3_xhand/init/davian_handle_eraser \
        --object_spec deployment/fr3_xhand/objects/davian_handle_eraser.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

PERCEPTION = Path(__file__).resolve().parent
REPO_ROOT = PERCEPTION.parents[2]
sys.path.insert(0, str(PERCEPTION))
from tracker import ObjectTracker  # noqa: E402

LAB_EXTRINSICS = (
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/calibration/"
    "results/extrinsics_347622076599_latest.npz"
)
EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]
AXIS_M = 0.06


def project(points_cam: np.ndarray, k: np.ndarray) -> np.ndarray:
    uv = (k @ points_cam.T).T
    return (uv[:, :2] / uv[:, 2:3]).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True)
    parser.add_argument("--object_spec", required=True)
    # A candidate calibration is worth checking before it replaces the one in
    # use, and the implied table height is the check, so it has to be possible
    # to point this at a file other than the live one.
    parser.add_argument("--extrinsics", default=LAB_EXTRINSICS)
    args = parser.parse_args()

    init = Path(args.init).resolve()
    spec = json.loads(Path(args.object_spec).read_text())
    mesh_path = REPO_ROOT / spec["mesh"]

    k = np.loadtxt(init / "cam_K.txt")
    rgb = cv2.imread(str(init / "rgb.png"))
    depth = cv2.imread(str(init / "depth.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(init / "mask.png"), cv2.IMREAD_GRAYSCALE)
    assert rgb is not None and depth is not None and mask is not None, f"{init} is incomplete"

    tracker = ObjectTracker(str(mesh_path), k)
    pose_cam = tracker.register(rgb, depth, mask)
    t_base_cam = np.load(args.extrinsics)["T_base_cam"]
    pose_base = t_base_cam @ pose_cam

    mesh = trimesh.load(mesh_path, force="mesh")
    lo, hi = np.asarray(mesh.vertices).min(axis=0), np.asarray(mesh.vertices).max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    corners_cam = (pose_cam[:3, :3] @ corners.T).T + pose_cam[:3, 3]
    uv = project(corners_cam, k)
    for a, b in EDGES:
        cv2.line(rgb, tuple(uv[a]), tuple(uv[b]), (0, 255, 0), 2)

    origin_uv = project(pose_cam[:3, 3][None], k)[0]
    for axis, color in zip(np.eye(3) * AXIS_M, ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        tip_cam = pose_cam[:3, :3] @ axis + pose_cam[:3, 3]
        cv2.line(rgb, tuple(origin_uv), tuple(project(tip_cam[None], k)[0]), color, 3)

    out = init / "registration_overlay.jpg"
    x, y = uv[:, 0], uv[:, 1]
    pad = 120
    crop = rgb[max(0, y.min() - pad):y.max() + pad, max(0, x.min() - pad):x.max() + pad]
    cv2.imwrite(str(out), crop)

    surface_z = pose_base[2, 3] - float(-np.asarray(mesh.vertices)[:, 2].min())
    print(f"[check] pose base xyz {np.round(pose_base[:3, 3], 4)}")
    print(f"[check] implied table surface at base z {surface_z:+.4f} m, the contract puts it at 0")
    print(f"[check] wrote {out}")


if __name__ == "__main__":
    main()
