from __future__ import annotations

from pathlib import Path

def _find_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(15):
        if (cur / "scripts").exists() and (cur / "tests").exists():
            return cur
        cur = cur.parent
    return start.resolve()

ROOT = _find_root(Path(__file__))

def test_repo_layout_smoke() -> None:
    assert (ROOT / "scripts").exists()
    assert (ROOT / "tests").exists()
    assert (ROOT / "pyproject.toml").exists() or (ROOT / "requirements.txt").exists()
