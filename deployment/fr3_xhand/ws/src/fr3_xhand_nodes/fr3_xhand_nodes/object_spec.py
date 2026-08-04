"""Loader for the per-object spec JSON consumed by the fr3-xhand deploy nodes.

Schema, object_name str, mesh str, object_scales three floats, goal_trajectory
a path to a dextoolbench-format trajectory JSON.

goal_trajectory holds ROBOT BASE frame poses, which is what goal_node
publishes. The sim world frame copy under dextoolbench/trajectories belongs to
the Isaac tools and goals_to_sim.py writes it, so the two must not be swapped.

Path fields may be absolute or repo relative, and both come back absolute. The
repo root comes from the spec's own location rather than this file's, because
this file moves into the colcon install space at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = ("object_name", "mesh", "object_scales", "goal_trajectory")
PATH_KEYS = ("mesh", "goal_trajectory")
# Specs live at <repo>/deployment/fr3_xhand/objects/<name>.json.
SPEC_DEPTH_BELOW_ROOT = 4


def repo_root_of(spec_path: Path) -> Path:
    root = spec_path.resolve().parents[SPEC_DEPTH_BELOW_ROOT - 1]
    assert (root / "deployment/fr3_xhand/objects").is_dir(), (
        f"object spec {spec_path} does not sit under deployment/fr3_xhand/objects"
    )
    return root


def load_object_spec(path: str) -> dict:
    spec_path = Path(path)
    spec = json.loads(spec_path.read_text())
    for key in REQUIRED_KEYS:
        assert key in spec, f"object spec {path} missing key {key}"
    assert len(spec["object_scales"]) == 3, (
        f"object spec {path} object_scales has length "
        f"{len(spec['object_scales'])}, expected 3"
    )

    root = repo_root_of(spec_path)
    for key in PATH_KEYS:
        resolved = Path(spec[key])
        if not resolved.is_absolute():
            resolved = root / resolved
        assert resolved.exists(), f"object spec {path} points {key} at a missing file, {resolved}"
        spec[key] = str(resolved)
    return spec
