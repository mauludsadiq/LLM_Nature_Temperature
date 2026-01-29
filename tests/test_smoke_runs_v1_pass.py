from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

def _device_flag() -> str:
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _run(cmd: list[str]) -> str:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    out = p.stdout
    assert p.returncode == 0, out
    return out

def test_smoke_margin_cliff_v1_writes_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "margin"
    dev = _device_flag()
    txt = _run([
        sys.executable, "-u", "-m", "scripts", "run_margin_cliff_v1",
        "--model", "gpt2",
        "--device", dev,
        "--seed", "0",
        "--T", "0.9",
        "--H", "4",
        "--tau", "0.90",
        "--trials", "1",
        "--context-fracs", "0.1,0.5",
        "--deltas", "0.0,0.5",
        "--out", str(out_dir),
    ])
    need = ["spec.json","grid.json","boundary.json","manifest.json"]
    for n in need:
        assert (out_dir / n).exists(), f"missing {n}\n{txt}"

def test_smoke_phase_surface_v1_writes_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "phase"
    dev = _device_flag()
    txt = _run([
        sys.executable, "-u", "-m", "scripts", "run_phase_surface_v1",
        "--model", "gpt2",
        "--device", dev,
        "--seed", "0",
        "--H", "4",
        "--tau", "0.90",
        "--trials", "1",
        "--temps", "0.7,1.0",
        "--context-fracs", "0.1,0.5",
        "--deltas", "0.0,0.5",
        "--out", str(out_dir),
    ])
    need = ["spec.json","grid.json","hazard_surface.json","boundary_surface.json","interaction_fit.json","manifest.json"]
    for n in need:
        assert (out_dir / n).exists(), f"missing {n}\n{txt}"
