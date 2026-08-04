"""Annotate the brush with the DexToolBench grasp box convention.

Builds the canonical object frame, origin at the handle arch centroid, x along
the arch bar, z up away from the bristles, following the convention verified
on the authors' blue_brush asset, origin in the handle, x toward the head,
scale equal to grasp box extent over 0.06.

Outputs, mesh_canonical.obj, canonicalized poses and goal trajectory, the
object spec JSON, and a side by side render of the authors' annotation and
ours for the dashboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
# The reference is handle_eraser, the benchmark's own loop-handle tool, user
# confirmed as the topological twin. Its origin sits at the arch bar, its box
# runs bar length in x, 1.59 times bar width in y spanning the grip zone, and
# 0.75 times bar height in z, ratios measured against its top 20 mm slab of
# 136.7 by 30.2 by 20.0 mm with scale (2.25, 0.8, 0.25).
REF_OBJ = REPO_ROOT / "assets/urdf/dextoolbench/eraser/handle_eraser/handle_eraser.obj"
REF_SCALE = np.array([2.25, 0.8, 0.25])
REF_BOX_TO_SLAB = np.array([135.0 / 136.7, 48.0 / 30.2, 15.0 / 20.0])
KEYPOINT_TO_METER = 0.03  # half extent per unit scale, 0.5 * 0.04 * 1.5
ARCH_SLAB_M = 0.024
GOAL_HZ = 3.0
CAPTURE_HZ = 30.0
LIFTOFF_M = 0.10

EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]


def box_corners(half: np.ndarray) -> np.ndarray:
    return np.array(
        [[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])]
    )


def draw_mesh_with_box(ax, mesh, half_extents, center, title):
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], cmap="viridis", linewidth=0, alpha=0.55, shade=True)
    c = box_corners(half_extents) + center
    for a, b in EDGES:
        ax.plot(*zip(c[a], c[b]), color="crimson", linewidth=2)
    for axis, color in zip(np.eye(3) * half_extents.max() * 1.4, ("red", "green", "blue")):
        ax.plot(*zip(center, center + axis), color=color, linewidth=2)
    ax.set_box_aspect(np.array(mesh.extents))
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    ax.view_init(elev=18, azim=-55)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True)
    parser.add_argument("--out_views", required=True)
    args = parser.parse_args()
    demo = Path(args.demo).resolve()

    mesh = trimesh.load(demo / "sam3d_output/mesh_scaled.obj", force="mesh")
    poses = np.load(demo / "poses_base_30hz.npy")

    # The tracked rest pose says which mesh axis pointed at the sky.
    r_rest = poses[0][:3, :3]
    up_alignment = r_rest.T @ np.array([0.0, 0.0, 1.0])
    up_idx = int(np.argmax(np.abs(up_alignment)))
    up_sign = float(np.sign(up_alignment[up_idx]))
    v = np.asarray(mesh.vertices)
    up_coord = v[:, up_idx] * up_sign
    slab = v[up_coord > up_coord.max() - ARCH_SLAB_M]
    assert len(slab) > 100, f"arch slab has only {len(slab)} vertices"

    lo, hi = slab.min(axis=0), slab.max(axis=0)
    center_old = (lo + hi) / 2.0
    extents_old = hi - lo

    lateral = [i for i in range(3) if i != up_idx]
    x_idx = lateral[int(np.argmax(extents_old[lateral]))]
    z_axis = np.zeros(3)
    z_axis[up_idx] = up_sign
    x_axis = np.zeros(3)
    x_axis[x_idx] = 1.0
    y_axis = np.cross(z_axis, x_axis)
    r_oc = np.stack([x_axis, y_axis, z_axis], axis=1)
    t_oc = np.eye(4)
    t_oc[:3, :3] = r_oc
    t_oc[:3, 3] = center_old

    arch_slab = np.abs(r_oc.T @ extents_old)
    arch_extents = arch_slab * REF_BOX_TO_SLAB
    scales = arch_extents / (2.0 * KEYPOINT_TO_METER)
    print(f"[annotate] arch slab m {np.round(arch_slab, 4)}, box m {np.round(arch_extents, 4)}, scales {np.round(scales, 3)}")

    mesh_canon = mesh.copy()
    mesh_canon.vertices = (np.asarray(mesh.vertices) - t_oc[:3, 3]) @ r_oc
    mesh_canon.export(demo / "sam3d_output/mesh_canonical.obj")

    poses_canon = np.array([p @ t_oc for p in poses])
    np.save(demo / "poses_base_30hz_canonical.npy", poses_canon)

    stride = int(round(CAPTURE_HZ / GOAL_HZ))
    goals = poses_canon[::stride]
    heights = goals[:, 2, 3]
    above = np.nonzero(heights - heights[0] > LIFTOFF_M)[0]
    assert above.size > 0
    goals = goals[above[0]:]

    def pose_to_xyz_xyzw(t):
        q = Rotation.from_matrix(t[:3, :3]).as_quat()
        return [*t[:3, 3].tolist(), *q.tolist()]

    traj = {
        "frame": "robot_base",
        "source_demo": str(demo.relative_to(REPO_ROOT)),
        "goal_hz": GOAL_HZ,
        "liftoff_m": LIFTOFF_M,
        "start_pose": pose_to_xyz_xyzw(poses_canon[0]),
        "goals": [pose_to_xyz_xyzw(g) for g in goals],
    }
    (demo / "goal_trajectory.json").write_text(json.dumps(traj, indent=2))
    print(f"[annotate] rewrote goal_trajectory.json with {len(goals)} canonical goals")

    spec = {
        "object_name": "davian_blue_scrub_brush",
        "mesh": str((demo / "sam3d_output/mesh_canonical.obj").relative_to(REPO_ROOT)),
        "object_scales": [round(float(s), 4) for s in scales],
        "goal_trajectory": str((demo / "goal_trajectory.json").relative_to(REPO_ROOT)),
        "annotation": "grasp box is the loop handle top arch, x along the arch bar, z up, "
        "box ratios mirrored from dextoolbench handle_eraser, the benchmark's own "
        "loop-handle tool, user-confirmed topological twin",
    }
    spec_path = REPO_ROOT / "deployment/fr3_xhand/objects/davian_blue_scrub_brush.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"[annotate] wrote {spec_path}")

    ref = trimesh.load(REF_OBJ, force="mesh")
    fig = plt.figure(figsize=(11, 4.2), dpi=110)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_mesh_with_box(
        ax1, ref, REF_SCALE * KEYPOINT_TO_METER, np.zeros(3),
        "authors' handle_eraser, loop handle, box 135 x 48 x 15 mm at the arch",
    )
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_mesh_with_box(
        ax2, mesh_canon, arch_extents / 2.0, np.zeros(3),
        f"our scrub brush, loop handle, box {' x '.join(str(int(e * 1000)) for e in arch_extents)} mm, same ratios",
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(args.out_views, pil_kwargs={"quality": 88})
    print(f"[annotate] wrote {args.out_views}")


if __name__ == "__main__":
    main()
