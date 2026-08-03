# ARC-AGI-3 Kaggle Starter — Sovereign Pure NumPy v0.41

This branch replaces the random starter policy with the ARC3 Sovereign competition runtime:

- Pure NumPy learned ensemble; no PyTorch, JAX, TensorFlow, vLLM, or Transformers at inference.
- Symbolic search, online program synthesis, click memory, calibrated novelty routing, and bounded AEGIS-RT causal adaptation.
- Learned models are proposal-only. Action legality, freshness, reset lifecycle, and transactional assimilation remain authoritative.
- The runtime and trained weights are embedded into the generated notebook with SHA-256 verification and guarded extraction.
- CPU-only Kaggle notebook, internet disabled, one BLAS thread for reproducible latency.

## Supported official contract

- Python 3.12+
- `arc-agi==0.9.9`
- `arcengine==0.9.3`
- `GameAction.ACTION1` through `ACTION7`; `RESET` remains lifecycle-only
- `ACTION6` uses validated `x`/`y` coordinates
- Official animation payloads are normalized by selecting the final visual frame

## Local verification

```bash
python3.12 -m venv .venv
.venv/bin/pip install "numpy>=1.26,<3" pytest
.venv/bin/python scripts/verify_sovereign.py
.venv/bin/pytest -q tests/test_sovereign_runtime.py
.venv/bin/python scripts/build_notebook.py
```

For real local games, keep the starter workflow:

```bash
make setup
make play-local GAME=ls20 STEPS=200
make notebook
```

## Kaggle submission

1. Review the Kaggle notebook ID in `notebooks/kernel-metadata.json`.
2. Keep internet disabled. GPU is not required.
3. Run `make submit` and wait for the commit run to complete.
4. On Kaggle, select `submission.parquet` and submit it to the competition rerun.

The generated notebook installs only official competition wheels from the mounted wheelhouse and verifies exact package versions before starting the agent.

## Data and model provenance

The full authored corpus contains 692,000 generated records across the base curriculum, quality overlay, frontier reasoning, hard mining, CEGIS closure, and AEGIS causal adaptation. Raw training HDF5 files are intentionally not loaded or embedded at competition time. This repository contains the frozen data lock, model/runtime source, and trained NumPy weights required for inference. Excluding unused training tensors reduces startup time, repository size, and submission failure surface without changing the policy.

## Evidence boundary

The runtime has passed deterministic local archive checks and synthetic/procedural tests. That is not a guarantee of a particular private leaderboard score. Competition readiness still requires a successful committed Kaggle rerun and the official score returned by the competition server.
