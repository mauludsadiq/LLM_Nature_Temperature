from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_smoke_v1_emits_pass_and_writes_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "_smoke_v1"
    cmd = [
        sys.executable, "-u", "-m", "scripts", "run_smoke_v1",
        "--device", "auto",
        "--out", str(out_dir),
    ]

    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "ignore:NotOpenSSLWarning"

    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )

    assert r.returncode == 0, r.stdout
    assert "PASS_SMOKE_V1" in r.stdout, r.stdout
    assert (out_dir / "margin" / "spec.json").exists()
    assert (out_dir / "margin" / "grid.json").exists()
    assert (out_dir / "margin" / "boundary.json").exists()
    assert (out_dir / "margin" / "manifest.json").exists()
    assert (out_dir / "phase" / "spec.json").exists()
    assert (out_dir / "phase" / "grid.json").exists()
    assert (out_dir / "phase" / "hazard_surface.json").exists()
    assert (out_dir / "phase" / "boundary_surface.json").exists()
    assert (out_dir / "phase" / "interaction_fit.json").exists()
    assert (out_dir / "phase" / "manifest.json").exists()
