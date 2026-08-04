"""FoundationPose wrapper shared by the offline goal extraction and the live node.

The two paths have to see identical preprocessing, so the mesh decimation and
the estimator settings live here instead of in each caller.
Poses come back in the camera frame, expressed in the mesh file's own origin,
because register and track_one both post-multiply the centering transform away.

Runs in the fp conda env, with the CUDA runtime libs appended to the loader
path. Appended, not replacing, because the ROS entries ahead of them are what
let rclpy import in this env.

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davian/anaconda3/envs/fp/lib \
        /home/davian/anaconda3/envs/fp/bin/python ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

FP_ROOT = "/home/davian/kinamkim/fp/FoundationPose"

# Registration rasterizes every candidate, and the full-detail mesh blows the
# VRAM budget. Pose accuracy does not need bristle geometry.
TARGET_FACES = 20000
REGISTER_ITERS = 5
TRACK_ITERS = 2


def load_mesh(mesh_path: str, target_faces: int = TARGET_FACES) -> trimesh.Trimesh:
    import open3d as o3d

    mesh = trimesh.load(mesh_path, force="mesh")
    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    o3 = o3.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    out = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles))
    print(f"[tracker] {Path(mesh_path).name}, {len(out.faces)} faces, extents m {np.round(out.extents, 4)}")
    return out


class ObjectTracker:
    """Registers once against a mask, then tracks frame to frame."""

    def __init__(
        self,
        mesh_path: str,
        cam_k: np.ndarray,
        target_faces: int = TARGET_FACES,
        debug_dir: str = "/tmp/fp_debug",
    ) -> None:
        self.k = np.array(cam_k, dtype=np.float64)
        self.mesh = load_mesh(mesh_path, target_faces)

        sys.path.insert(0, FP_ROOT)
        import torch  # noqa: F401  FoundationPose imports expect torch first
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr

        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=0,
            # The upstream default points at the author's home directory.
            debug_dir=debug_dir,
        )
        self.registered = False

    def _prepare(self, rgb_bgr: np.ndarray, depth_mm: np.ndarray):
        return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), depth_mm.astype(np.float64) / 1000.0

    def register(
        self, rgb_bgr: np.ndarray, depth_mm: np.ndarray, mask: np.ndarray,
        iteration: int = REGISTER_ITERS,
    ) -> np.ndarray:
        rgb, depth = self._prepare(rgb_bgr, depth_mm)
        pose = self.est.register(
            K=self.k, rgb=rgb, depth=depth, ob_mask=mask.astype(bool), iteration=iteration
        )
        self.registered = True
        return np.asarray(pose, dtype=np.float64)

    def track(
        self, rgb_bgr: np.ndarray, depth_mm: np.ndarray, iteration: int = TRACK_ITERS,
    ) -> np.ndarray:
        assert self.registered, "track called before register"
        rgb, depth = self._prepare(rgb_bgr, depth_mm)
        pose = self.est.track_one(rgb=rgb, depth=depth, K=self.k, iteration=iteration)
        return np.asarray(pose, dtype=np.float64)
