"""Render the tracked object trajectory as a bbox overlay video.

Consumes poses_base_30hz.npy from extract_goals.py, converts back to camera
frame with the lab calibration, projects the mesh bounding box onto each demo
frame, and encodes an h264 mp4 for the dashboard.

    /home/davian/anaconda3/envs/fp/bin/python \
        deployment/fr3_xhand/perception/render_track_video.py \
        --demo deployment/fr3_xhand/demos/demo_20260803_081042 \
        --mesh deployment/fr3_xhand/demos/demo_20260803_081042/sam3d_output/mesh_scaled.obj
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

LAB_EXTRINSICS = Path(
    "/home/davian/byungkunlee/davian_robotics_real3d/deploy/calibration/results/"
    "extrinsics_347622076599_latest.npz"
)

EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True)
    parser.add_argument("--mesh", required=True)
    args = parser.parse_args()

    demo = Path(args.demo)
    picks = json.loads((demo / "picks.json").read_text())
    start, end = picks["trajectory_start_frame"], picks["trajectory_end_frame"]
    k = np.loadtxt(demo / "cam_K.txt")
    t_base_cam = np.load(LAB_EXTRINSICS)["T_base_cam"]
    t_cam_base = np.linalg.inv(t_base_cam)

    poses_base = np.load(demo / "poses_base_30hz.npy")
    assert len(poses_base) == end - start + 1, (len(poses_base), start, end)

    mesh = trimesh.load(args.mesh, force="mesh")
    lo, hi = mesh.bounds
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    with tempfile.TemporaryDirectory() as tmp:
        for i, t_base_obj in enumerate(poses_base):
            idx = start + i
            img = cv2.imread(str(demo / "rgb" / f"{idx:06d}.png"))
            t_cam_obj = t_cam_base @ t_base_obj
            pts_cam = corners @ t_cam_obj[:3, :3].T + t_cam_obj[:3, 3]
            uv = (pts_cam @ k.T)
            uv = uv[:, :2] / uv[:, 2:3]
            uv = uv.astype(int)
            for a, b in EDGES:
                cv2.line(img, tuple(uv[a]), tuple(uv[b]), (0, 200, 255), 2)
            origin = t_cam_obj[:3, 3]
            for axis, color in zip(np.eye(3) * 0.05, ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                tip = origin + t_cam_obj[:3, :3] @ axis
                pair = np.stack([origin, tip]) @ k.T
                pair = (pair[:, :2] / pair[:, 2:3]).astype(int)
                cv2.line(img, tuple(pair[0]), tuple(pair[1]), color, 3)
            cv2.putText(img, f"frame {idx}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{tmp}/{i:06d}.png", img)

        out = demo / "trajectory_overlay.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30",
             "-i", f"{tmp}/%06d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "26", str(out)],
            check=True,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
