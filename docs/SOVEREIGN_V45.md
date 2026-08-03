# ARC3 Sovereign NumPy v0.45 — Data-Derived Real-Time Cognition

The competition hot path contains no public-game identifiers, per-game scripts, fixed semantic color palette, fixed actor/goal color, or fixed action-to-motion mapping. Runtime knowledge is constructed from the current authorized frame sequence and verified action outcomes.

## Visual pipeline

`FrameData / animation tensor → preserve every 2-D frame → infer background → segment value components → measure morphology/topology → temporal correspondence → persistent object identities → scene graph → rule induction`

A single frame yields visual identity and calibrated hypotheses, not fabricated functional certainty. Actor, goal, switch, hazard, or tool roles are earned from temporal and intervention evidence.

Set `ARC3_TRACE_GIF=/path/trace.gif` locally to generate a continuously refreshed GIF plus JSON sidecar. The left pane shows observed equality classes and the right pane shows learned track identities. GIF tracing is disabled in the competition notebook.

Rules are bounded distributions over translations, creation/removal, recoloring, resizing, shape changes, changed-cell mass, progress, terminal loss, and click outcomes. Blocked actions do not corrupt semantic mappings; strong contradictions create a new semantic epoch.

The offline corpus builder consumes real recording JSONL and optional literal rulebook data, strips solution scripts/source paths, and emits per-level scene/rule/provenance records plus a 16-family branching plan. Its exact schema acceptance passes for 25 games and 183 levels. The real frame-derived corpus remains blocked until raw authorized recordings are mounted; no placeholder is represented as real evidence.

## Repository packaging

`agent/sovereign_v45_runtime.zip` is deterministic auditable Python source, verified by SHA-256 before import. `verification_v45.zip` contains tests, the offline corpus builder, documentation, evidence, and inventory. CI expands it before running all gates. This packaging avoids notebook/source drift while preserving exact source bytes.

## Claim boundary

Controlled local tests are not an official ARC-AGI-3 score. Private Kaggle performance remains unverified.
