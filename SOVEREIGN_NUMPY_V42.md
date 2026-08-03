# Sovereign Pure NumPy v42

This branch replaces the stochastic starter with a deterministic, evidence-gated ARC-AGI-3 agent designed for offline Kaggle competition reruns.

## What changed

- Learns unknown action semantics from observed frame transitions.
- Uses the final frame of each animation sequence as the authoritative observation.
- Extracts connected components and tracks translated objects to identify the controllable actor.
- Learns traversable terrain from cells vacated and entered by the actor.
- Builds footprint-aware shortest paths to compact, globally rare goal candidates.
- Records state-action no-ops and avoids repeating known blocked transitions.
- Detects contradictory mature action effects and starts a fresh causal epoch for hidden control remaps.
- Ranks ACTION6 coordinates by rarity, compactness, prior success, and prior no-op evidence.
- Masks every proposal by the official available-action set and validates ACTION6 coordinates against the current grid.
- Uses deterministic tie-breaking; there is no random action fallback.

The NumPy policy is still a proposal layer. The official framework owns environment state and dispatch, and RESET is used only for NOT_PLAYED/GAME_OVER lifecycle states.

## Reproducible intelligence evidence

Run:

```bash
make setup
make verify-intelligence
```

The committed benchmark uses 200 fixed seeds per condition and compares against a deterministic random baseline:

| Condition | v42 completion | Random completion | Absolute gain | Illegal actions |
|---|---:|---:|---:|---:|
| Permuted-control navigation | 98.5% | 0.5% | +98.0 points | 0 |
| Hidden control remap | 94.0% | 1.5% | +92.5 points | 0 |
| Coordinate click with decoys | 100.0% | 10.5% | +89.5 points | 0 |

These are deterministic synthetic/procedural tests. They demonstrate a clear behavioral improvement over the repository's random baseline, but they are not an official Kaggle score and do not certify private-set performance.

## Submission profile

- Python 3.12
- `arc-agi==0.9.9`
- `arcengine==0.9.3`
- Pure NumPy CPU runtime
- Internet disabled
- GPU disabled
- BLAS/NumExpr thread count fixed to one
- `PYTHONHASHSEED=0`
- Source safety scan before notebook generation
- Intelligence tests must pass before `make notebook` or `make submit`

## Evidence files

- `evidence/intelligence_benchmark_v42.json`
- `scripts/benchmark_agent.py`
- `tests/test_intelligence.py`
- `.github/workflows/intelligence-gates.yml`

## Remaining external gates

The branch is locally and procedurally verified. Competition readiness remains unverified until a committed Kaggle competition rerun completes against the private gateway and produces a valid official score.
