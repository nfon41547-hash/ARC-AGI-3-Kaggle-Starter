# ARC3 Sovereign NumPy v0.43

## Competition architecture

The submission is a modular Pure NumPy agent. The notebook builder embeds four
runtime files and copies them into the official starter framework before the
competition process begins.

```text
Official frame
→ strict grid/action normalization
→ component and role hypotheses
→ contextual causal posterior
→ footprint-aware shortest-path search
→ counterfactual action scoring
→ legal action dispatch
→ next-frame verification
→ bounded change-point repair
```

The runtime performs no network I/O, model download, subprocess launch,
dynamic execution, or unrestricted online weight training.

## Intelligence changes

- Learns action semantics from verified transitions rather than action names.
- Keeps separate posteriors for visible structural modes.
- Repairs only the contradicted action/context posterior after a confident
  non-zero prediction error.
- Treats blocked movement as a map constraint instead of false remapping data.
- Infers moving actors from translated connected components when colors are
  unfamiliar.
- Plans with the complete multi-cell actor footprint.
- Retains verified semantics across `GAME_OVER` resets while clearing local
  cycles and click attempts.
- Ranks complex-action coordinates by rarity, salience, causal-change proximity,
  and negative click evidence.
- Bounds context, state-action, visit, outcome, and click memories.

## Controlled evidence

`tests/v43_regression.py` covers unknown action permutations, visible mode
switches, hidden remapping, multi-cell actors, reset retention, unfamiliar role
colors, click-only interfaces, and restricted legal-action masks. Results are
committed in `evidence/v43_intelligence_benchmark.json`.

This is controlled synthetic evidence only, not an official ARC-AGI-3 score.

## Submission profile

`ACCELERATOR = "cpu"` is intentional. The policy is Pure NumPy and has no GPU
kernel, so a GPU session adds quota and startup risk without increasing policy
capability. Internet remains disabled and BLAS thread counts are fixed to one.

Before a real Kaggle push, replace `REPLACE_WITH_YOUR_USERNAME` in
`notebooks/kernel-metadata.json`, then run:

```bash
make notebook
make verify-local
make submit
```

A successful CI benchmark does not establish private leaderboard performance.
Competition readiness still requires a committed Kaggle rerun and inspection of
its logs and generated `submission.parquet`.
