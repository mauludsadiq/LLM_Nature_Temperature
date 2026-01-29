# LLM Nature Temperature

This repository is a **deterministic instrumentation + certification framework** for **autoregressive stability** in language models.

It does **not** benchmark “task performance.” Instead, it measures when a model’s multi-step continuation undergoes a **stochastic phase transition** from stable continuation to runaway divergence as a function of two controlled stressors:

- **Temperature** \(T\): sampling entropy ("thermal agitation")
- **Context saturation** \(C\): \(C := N_{context}/N_{max}\), the fraction of the context window consumed ("structural pressure")

The repo produces **hash-anchored phase artifacts** (boundary curves/surfaces + hazard fits) that can be used as an **ABI stability envelope** (e.g., a `low_margin_gate_v1`) to prevent long-horizon degeneration.

## What this repo does (explicit definition)

Define a discrete-time hazard (failure rate) for a continuation step \(t\):

\[
 h_t := \Pr(\text{failure at step }t \mid \text{survived to }t)
\]

This repo measures \(h_t\) indirectly by running a calibrated forced-choice process and recording the time-to-failure distribution. It then publishes:

1) A **phase boundary** (Experiment B):

\[
\delta_c(C;\tau) := \min\{\delta : \hat p_{survive}(C,\delta;H) \ge \tau\}
\]

2) A **phase surface** (Experiment B ⊗ C):

\[
\delta_c(T,C;\tau) := \min\{\delta : \hat p_{survive}(T,C,\delta;H) \ge \tau\}
\]

3) A **hazard interaction fit** (Experiment C) on a selected slice (typically \(\delta=0\)):

\[
\log r(T,C) = \log k + \alpha\log T + \beta\log C
\]

where \(r(T,C)\) is a hazard proxy computed as \(r = 1/\widehat{\mathbb{E}}[T_{failure}]\) with censoring at horizon \(H\).

### The calibrated gauge: forced two-token continuation

Rather than asking the model a question, the instrument forces a two-token choice at every generation step:

- `token_correct` must be selected for survival
- `token_distract` ends the trial (failure event)

A **margin knob** \(\delta\) is applied as an explicit logit bias to the correct token:

\[
\ell_{correct} \leftarrow \ell_{correct} + \delta
\]

Interpretation: \(\delta\) measures “how much help” the model needs to remain stable at \((T,C)\).

## Outputs

After a run you will have:

- `out/phase_surface_v1/spec.json` — the full run specification (grid, tokens, horizon, seeds)
- `out/phase_surface_v1/grid.json` — full lattice over \((T,C,\delta)\) with survival + hazard summaries
- `out/phase_surface_v1/boundary_surface.json` — the phase surface \(\delta_c(T,C;\tau)\)
- `out/phase_surface_v1/hazard_surface.json` — hazard slice \(r(T,C;\delta=hazard\_delta)\)
- `out/phase_surface_v1/interaction_fit.json` — fitted \((\alpha,\beta,R^2)\)
- `out/phase_surface_v1/manifest.json` — SHA-256 digests for registry anchoring

The simpler 1D boundary run (Experiment B only) writes the analogous artifacts under `out/margin_cliff_v1/`.

## ABI usage (turn the artifact into a stability envelope)

Once you have `boundary_surface.json`, an ABI gate can enforce an operational stability envelope during generation:

1. Compute context saturation \(C(t)=N_{context}/N_{max}\)
2. Read/interpolate \(\delta_c(T,C;\tau)\)
3. Compute live margin \(\Delta_t = \ell_{top1}-\ell_{top2}\)
4. If \(\Delta_t < \delta_c(T,C)\), **reject** the step and trigger an intervention:

- contrastive decoding / margin boosting
- rollback + counterfactual resampling
- lower temperature adaptively
- recompute / disable KV cache (if drift dominates)

This converts “guessing when the model will fail” into **measured envelope control**.

---

# Quickstart (GPT-2 small)

## 1) Clone or copy locally

Clone:

```bash
git clone <YOUR_REMOTE_URL>
cd LLM_Nature_Temperature
```

Or create a local folder and copy the repo contents into it.

## 2) Create a virtual environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
```

## 3) Install dependencies

CPU-only (recommended for a deterministic reference run):

```bash
pip install -r requirements.txt
```

CUDA:

- Install a CUDA-enabled PyTorch build that matches your system (see PyTorch instructions), then:

```bash
pip install -r requirements.txt
```

## 4) Run Experiment B (margin boundary \(\delta_c(C)\))

```bash
python -m scripts.run_margin_cliff_v1 \
  --model gpt2 \
  --device cpu \
  --seed 0 \
  --T 0.9 \
  --H 128 \
  --tau 0.90 \
  --trials 200 \
  --out out/margin_cliff_v1
```

Artifacts will appear in `out/margin_cliff_v1/`.

## 5) Run Experiment B ⊗ C (phase surface \(\delta_c(T,C)\) + interaction fit)

```bash
python -m scripts.run_phase_surface_v1 \
  --model gpt2 \
  --device cpu \
  --seed 0 \
  --H 128 \
  --tau 0.90 \
  --trials 200 \
  --out out/phase_surface_v1
```

## 6) Run tests

Unit tests (fast, no model download required):

```bash
pytest -q
```

Integration tests (runs a tiny instrument sweep; downloads GPT-2 once):

```bash
RUN_INTEGRATION=1 pytest -q
```

---

# VS Code workflow

This repo includes a `.vscode/` folder with:

- `settings.json` — sensible defaults for Python in this repo
- `extensions.json` — recommended extensions
- `tasks.json` — one-command tasks to run the instrument
- `launch.json` — debug configs for the scripts

## Steps

1. Open the repo folder in VS Code.
2. `Cmd/Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.venv`.
3. Use the VS Code Task Runner:
   - `Terminal` → `Run Task` → **Run: Phase Surface (GPT-2 CPU)**

---

# Reproducibility notes

- CPU runs are the best reference point for deterministic behavior.
- GPU kernels can introduce nondeterminism even when “deterministic algorithms” is enabled.
- The scripts seed:
  - Python hashing
  - NumPy
  - Torch CPU + CUDA
  - and uses per-cell RNG seeding \(seed|T|C|\delta\) so the lattice is stable.

---

# License

MIT (see `LICENSE`).
