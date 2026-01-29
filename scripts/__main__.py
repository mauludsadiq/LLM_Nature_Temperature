from __future__ import annotations

import importlib
import sys

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 -m scripts <module_name> [args...]")
    mod_name = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    mod = importlib.import_module(f"scripts.{mod_name}")
    if not hasattr(mod, "main"):
        raise SystemExit(f"scripts.{mod_name} has no main()")
    mod.main()

if __name__ == "__main__":
    main()
