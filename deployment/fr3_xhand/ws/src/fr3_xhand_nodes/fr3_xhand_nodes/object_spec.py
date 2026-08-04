"""Loader for the per-object spec JSON consumed by the fr3-xhand deploy nodes.

Schema, object_name str, mesh str, object_scales three floats, goal_trajectory
a path to a dextoolbench-format trajectory JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = ("object_name", "mesh", "object_scales", "goal_trajectory")


def load_object_spec(path: str) -> dict:
    spec = json.loads(Path(path).read_text())
    for key in REQUIRED_KEYS:
        assert key in spec, f"object spec {path} missing key {key}"
    assert len(spec["object_scales"]) == 3, (
        f"object spec {path} object_scales has length "
        f"{len(spec['object_scales'])}, expected 3"
    )
    return spec
