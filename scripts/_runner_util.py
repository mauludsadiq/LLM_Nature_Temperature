from __future__ import annotations

from pathlib import Path
from typing import Iterable

def is_mps_device(device: str) -> bool:
    d = str(device).strip().lower()
    return d in ("mps", "apple", "metal") or d.startswith("mps")

def force_no_kv_cache_on_mps(args) -> None:
    dev = str(getattr(args, "device", "")).strip().lower()
    if is_mps_device(dev) and hasattr(args, "no_kv_cache") and not getattr(args, "no_kv_cache"):
        print("[NOTE] MPS detected: forcing --no-kv-cache to avoid KV-cache stall.", flush=True)
        args.no_kv_cache = True

def print_write_report(tag: str, out_dir: Path, files: Iterable[str]) -> None:
    need = list(files)
    ok = all((out_dir / n).exists() for n in need)
    print(f"PASS_{tag}_WROTE_JSON" if ok else f"FAIL_{tag}_MISSING_FILES", flush=True)
    for n in need:
        q = out_dir / n
        print("WROTE:", str(q), "bytes", (q.stat().st_size if q.exists() else -1), flush=True)
