"""Tier-2 test: the committed deploy contract matches a fresh export.

The deploy stack trusts deployment/fr3_xhand/contract/a4h1.json as its single
source of truth. This re-exports the contract from a live env in a subprocess
and asserts byte equality, so any registry or env change that shifts the
contract fails here instead of silently skewing the real robot.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_deploy_contract.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = REPO_ROOT / "deployment/fr3_xhand/contract"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "deployment/fr3_xhand/export_contract.py"),
                "--out", tmp,
            ],
            check=True,
        )
        for name in ("a4h1.json", "a4h1_train.yaml"):
            committed = (CONTRACT_DIR / name).read_bytes()
            fresh = (Path(tmp) / name).read_bytes()
            assert committed == fresh, (
                f"{name} drifted from a fresh export, re-run export_contract.py "
                "and review the diff before committing"
            )

    print("[test] deploy contract matches fresh export OK")


if __name__ == "__main__":
    main()
