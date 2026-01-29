from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from scripts._runner_util import force_no_kv_cache_on_mps, print_write_report
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _get_n_max(model: Any) -> int:
    cfg = getattr(model, "config", None)
    for k in ("n_positions", "max_position_embeddings"):
        v = getattr(cfg, k, None)
        if isinstance(v, int) and v > 0:
            return v
    return 1024


def _set_det(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


@dataclass(frozen=True)
class SpecMarginCliffV1:
    exp_id: str
    exp_version: str
    model_name: str
    device: str
    seed: int
    temperature: float
    horizon_steps: int
    tau_survive: float
    n_trials: int
    context_fracs: List[float]
    deltas: List[float]
    base_prompt: str
    filler_text: str
    token_correct: str
    token_distract: str
    use_kv_cache: bool


def _pad_to_len(base_ids: List[int], n_target: int, filler_ids: List[int]) -> List[int]:
    if n_target <= len(base_ids):
        return base_ids[:n_target]
    out = list(base_ids)
    while len(out) < n_target:
        take = min(len(filler_ids), n_target - len(out))
        out.extend(filler_ids[:take])
        if len(filler_ids) == 0:
            break
    return out


def _sample_two(lc: float, ld: float, temperature: float, rng: np.random.Generator) -> int:
    t = max(1e-8, float(temperature))
    a = lc / t
    b = ld / t
    m = max(a, b)
    pa = math.exp(a - m)
    pb = math.exp(b - m)
    z = pa + pb
    pa /= z
    u = float(rng.random())
    return 0 if u < pa else 1


@torch.no_grad()
def _trial(
    model: Any,
    device: str,
    input_ids0: torch.Tensor,
    id_correct: int,
    id_distract: int,
    delta: float,
    temperature: float,
    horizon_steps: int,
    rng: np.random.Generator,
    use_kv_cache: bool,
) -> Tuple[bool, int, float]:
    input_ids = input_ids0
    past = None
    margins: List[float] = []

    for step in range(1, horizon_steps + 1):
        out = model(input_ids=input_ids, use_cache=use_kv_cache, past_key_values=past)
        logits = out.logits
        if use_kv_cache:
            past = out.past_key_values

        last = logits[0, -1, :]
        lc = float(last[id_correct].item()) + float(delta)
        ld = float(last[id_distract].item())
        margins.append(lc - ld)

        pick = _sample_two(lc, ld, temperature=temperature, rng=rng)
        chosen = id_correct if pick == 0 else id_distract
        if chosen != id_correct:
            return (False, step, float(np.mean(margins)) if margins else 0.0)

        nxt = torch.tensor([[chosen]], device=device, dtype=torch.long)
        if use_kv_cache:
            input_ids = nxt
        else:
            input_ids = torch.cat([input_ids, nxt], dim=1)

    return (True, horizon_steps + 1, float(np.mean(margins)) if margins else 0.0)


def run(spec: SpecMarginCliffV1, out_dir: Path) -> Dict[str, Any]:
    _set_det(spec.seed)

    model = AutoModelForCausalLM.from_pretrained(spec.model_name)
    model.eval()
    model.to(spec.device)

    tok = AutoTokenizer.from_pretrained(spec.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    n_max = _get_n_max(model)
    max_prompt_len = n_max - 2

    ids_c = tok.encode(spec.token_correct, add_special_tokens=False)
    ids_d = tok.encode(spec.token_distract, add_special_tokens=False)
    if len(ids_c) != 1 or len(ids_d) != 1:
        raise ValueError(f"Tokens must map to exactly 1 token each. correct={ids_c} distract={ids_d}")
    id_correct, id_distract = ids_c[0], ids_d[0]

    base_ids = tok.encode(spec.base_prompt, add_special_tokens=False)
    filler_ids = tok.encode(spec.filler_text, add_special_tokens=False)
    if len(filler_ids) == 0:
        raise ValueError("filler_text tokenized to empty")

    prompt_by_C: Dict[float, List[int]] = {}
    ctx_len_by_C: Dict[float, int] = {}
    for C in spec.context_fracs:
        c = float(C)
        n_target = int(round(c * max_prompt_len))
        n_target = max(8, min(max_prompt_len, n_target))
        ctx_len_by_C[c] = n_target
        prompt_by_C[c] = _pad_to_len(base_ids, n_target, filler_ids)

    grid: List[Dict[str, Any]] = []
    boundary: List[Dict[str, Any]] = []

    for C in spec.context_fracs:
        c = float(C)
        input0 = torch.tensor([prompt_by_C[c]], device=spec.device, dtype=torch.long)

        survive_curve: List[Tuple[float, float]] = []

        for delta in spec.deltas:
            d = float(delta)
            survived = 0
            collapse_steps: List[int] = []
            mean_margins: List[float] = []

            cell_seed = int(_sha256_hex(f"{spec.seed}|{c}|{d}")[:8], 16)
            rng = np.random.default_rng(cell_seed)

            for _ in range(spec.n_trials):
                ok, step, mean_margin = _trial(
                    model=model,
                    device=spec.device,
                    input_ids0=input0,
                    id_correct=id_correct,
                    id_distract=id_distract,
                    delta=d,
                    temperature=spec.temperature,
                    horizon_steps=spec.horizon_steps,
                    rng=rng,
                    use_kv_cache=spec.use_kv_cache,
                )
                if ok:
                    survived += 1
                else:
                    collapse_steps.append(step)
                mean_margins.append(mean_margin)

            p_survive = survived / float(spec.n_trials)
            mean_T = (
                (sum(collapse_steps) + (spec.horizon_steps + 1) * (spec.n_trials - len(collapse_steps)))
                / float(spec.n_trials)
            )
            hazard_r = 1.0 / float(mean_T)

            grid.append(
                {
                    "C": c,
                    "C_target_prompt_len": int(ctx_len_by_C[c]),
                    "delta": d,
                    "p_survive_H": float(p_survive),
                    "mean_T_censored": float(mean_T),
                    "hazard_r": float(hazard_r),
                    "mean_margin": float(np.mean(mean_margins)) if mean_margins else 0.0,
                }
            )
            survive_curve.append((d, p_survive))

        survive_curve.sort(key=lambda x: x[0])
        delta_c: Optional[float] = None
        for d, p in survive_curve:
            if p >= spec.tau_survive:
                delta_c = d
                break

        boundary.append({"C": c, "C_target_prompt_len": int(ctx_len_by_C[c]), "tau": spec.tau_survive, "delta_c": delta_c})

    out_dir.mkdir(parents=True, exist_ok=True)

    spec_obj = asdict(spec)
    spec_obj["n_max_positions"] = int(n_max)
    spec_obj["token_ids"] = {"correct": int(id_correct), "distract": int(id_distract)}

    spec_sha = _sha256_hex(_canon(spec_obj))
    grid_sha = _sha256_hex(_canon(grid))
    boundary_sha = _sha256_hex(_canon(boundary))

    _write_json(out_dir / "spec.json", spec_obj)
    _write_json(out_dir / "grid.json", grid)
    _write_json(out_dir / "boundary.json", boundary)

    manifest = {
        "exp_id": spec.exp_id,
        "exp_version": spec.exp_version,
        "spec_sha256": spec_sha,
        "grid_sha256": grid_sha,
        "boundary_sha256": boundary_sha,
        "status": "COMPLETE",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    print("[RUN] run_margin_cliff_v1", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=0.9)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.90)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--context-fracs", default="0.10,0.20,0.35,0.50,0.65,0.80,0.90,0.95")
    ap.add_argument("--deltas", default="-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--base-prompt", default="Answer with")
    ap.add_argument("--filler-text", default=" .")
    ap.add_argument("--token-correct", default=" yes")
    ap.add_argument("--token-distract", default=" no")
    ap.add_argument("--no-kv-cache", action="store_true")
    ap.add_argument("--out", default="out/margin_cliff_v1")
    args = ap.parse_args()
    force_no_kv_cache_on_mps(args)

    context_fracs = [float(x.strip()) for x in args.context_fracs.split(",") if x.strip()]
    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]

    spec = SpecMarginCliffV1(
        exp_id="margin_cliff_v1",
        exp_version="v1",
        model_name=args.model,
        device=args.device,
        seed=args.seed,
        temperature=args.T,
        horizon_steps=args.H,
        tau_survive=args.tau,
        n_trials=args.trials,
        context_fracs=context_fracs,
        deltas=deltas,
        base_prompt=args.base_prompt,
        filler_text=args.filler_text,
        token_correct=args.token_correct,
        token_distract=args.token_distract,
        use_kv_cache=not args.no_kv_cache,
    )
    run(spec, out_dir=Path(args.out))
    print_write_report("MARGIN_CLIFF", Path(args.out), [
        "spec.json",
        "grid.json",
        "boundary.json",
        "manifest.json",
    ])


if __name__ == "__main__":
    main()
