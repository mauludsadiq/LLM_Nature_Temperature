from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def test_storyboard_v1_smoke_writes_files() -> None:
    out = Path("out/_pytest_storyboard_v1")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "scripts", "run_storyboard_v1",
        "--device", "auto",
        "--model", "gpt2",
        "--seed", "0",
        "--T", "0.9",
        "--turns", "6",
        "--max-new-tokens", "64",
        "--out", str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert "PASS_STORYBOARD_V1" in (r.stdout + r.stderr)

    need = [
        out / "spec.json",
        out / "with" / "storyboard.md",
        out / "without" / "storyboard.md",
        out / "with" / "turns.json",
        out / "without" / "turns.json",
    ]
    for p in need:
        assert p.exists() and p.stat().st_size > 0, f"missing/empty: {p}"
