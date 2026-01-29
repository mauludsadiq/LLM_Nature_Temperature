from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import List

import torch


def _pick_device(arg: str) -> str:
    a = str(arg or "").strip().lower()
    if a and a != "auto":
        return a
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_module_main(mod_name: str, argv: List[str]) -> None:
    mod = importlib.import_module(f"scripts.{mod_name}")
    if not hasattr(mod, "main"):
        raise SystemExit(f"ERROR: scripts.{mod_name} has no main()")
    old_argv = sys.argv[:]
    try:
        sys.argv = ["python3 -m scripts " + mod_name] + argv
        mod.main()
    finally:
        sys.argv = old_argv


def _req(out_dir: Path, names: List[str]) -> None:
    missing = [n for n in names if not (out_dir / n).exists()]
    if missing:
        raise SystemExit(f"ERROR: missing files in {out_dir}: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--T", type=float, default=0.9)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--temps", default="0.7,1.0")
    ap.add_argument("--context-fracs", default="0.1,0.5")
    ap.add_argument("--deltas", default="0.0,0.5")
    ap.add_argument("--out", default="out/_smoke_v1")
    args = ap.parse_args()

    dev = _pick_device(args.device)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    margin_out = root / "margin"
    phase_out = root / "phase"
    margin_out.mkdir(parents=True, exist_ok=True)
    phase_out.mkdir(parents=True, exist_ok=True)

    print("[RUN] run_smoke_v1", flush=True)
    print("[NOTE] device =", dev, flush=True)

    margin_argv = [
        "--model", str(args.model),
        "--device", str(dev),
        "--seed", str(args.seed),
        "--T", str(args.T),
        "--H", str(args.H),
        "--tau", str(args.tau),
        "--trials", str(args.trials),
        "--context-fracs", str(args.context_fracs),
        "--deltas", str(args.deltas),
        "--out", str(margin_out),
    ]

    phase_argv = [
        "--model", str(args.model),
        "--device", str(dev),
        "--seed", str(args.seed),
        "--H", str(args.H),
        "--tau", str(args.tau),
        "--trials", str(args.trials),
        "--temps", str(args.temps),
        "--context-fracs", str(args.context_fracs),
        "--deltas", str(args.deltas),
        "--out", str(phase_out),
    ]

    _run_module_main("run_margin_cliff_v1", margin_argv)
    _run_module_main("run_phase_surface_v1", phase_argv)

    need_margin = ["spec.json", "grid.json", "boundary.json", "manifest.json"]
    need_phase = ["spec.json", "grid.json", "hazard_surface.json", "boundary_surface.json", "interaction_fit.json", "manifest.json"]

    _req(margin_out, need_margin)
    _req(phase_out, need_phase)

    print("PASS_SMOKE_V1", flush=True)
    for n in need_margin:
        q = margin_out / n
        print("WROTE:", str(q), "bytes", q.stat().st_size, flush=True)
    for n in need_phase:
        q = phase_out / n
        print("WROTE:", str(q), "bytes", q.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
