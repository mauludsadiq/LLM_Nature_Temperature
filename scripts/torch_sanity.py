from __future__ import annotations
import torch

def main() -> None:
    print("torch:", torch.__version__)
    print("mps_available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    x = torch.randn(1024, 1024)
    y = x @ x
    print("matmul_ok:", float(y[0,0]))

if __name__ == "__main__":
    main()
