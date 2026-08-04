# Gemma 4 31B + RTX PRO 6000 Dataset Architecture

This branch switches the competition packaging plan from a CPU-only Pure NumPy submission to a GPU-backed Gemma 4 31B proposal system targeting NVIDIA RTX PRO 6000 Blackwell.

## Inputs

1. Official ARC-AGI-3 competition dataset and toolkit assets.
2. A private Kaggle Dataset containing Gemma 4 31B weights, processor/tokenizer files, license metadata, and exact file hashes.
3. A private Kaggle Dataset containing a tested Blackwell/SM120 offline wheelhouse for the exact Kaggle Python/CUDA image.
4. This repository source bundle.

## Authority boundary

Gemma is proposal-only. It may propose object labels, goal hypotheses, compact world rules, candidate programs, and tool calls. It cannot bypass official available actions, RESET lifecycle, transaction freshness, deterministic replay, legality checks, resource deadlines, or final output validation.

## Required model dataset layout

```text
arc3-gemma4-31b-model/
├── model/
│   ├── config.json
│   ├── generation_config.json
│   ├── model-*.safetensors
│   ├── model.safetensors.index.json
│   ├── processor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── MODEL_MANIFEST.json
├── LICENSE
├── SHA256SUMS
└── smoke_test_prompts.json
```

## Required SM120 wheelhouse layout

```text
arc3-blackwell-sm120-wheelhouse/
├── wheels/
│   ├── torch-*.whl
│   ├── torchvision-*.whl
│   ├── vllm-*.whl
│   ├── transformers-*.whl
│   ├── triton-*.whl
│   ├── flashinfer_python-*.whl
│   ├── safetensors-*.whl
│   ├── tokenizers-*.whl
│   ├── accelerate-*.whl
│   ├── msgspec-*.whl
│   └── official ARC wheels
├── WHEEL_MANIFEST.json
├── COMPATIBILITY.json
├── BUILD_PROVENANCE.json
├── SHA256SUMS
└── smoke_test.py
```

## Runtime rules

- Internet disabled.
- GPU required.
- Exact model and wheel hashes verified before installation or loading.
- GPU name, VRAM, CUDA runtime, compute capability, Torch CUDA tag, and package versions logged.
- Attention backend remains automatic. Do not force FlashInfer for Gemma 4; mixed head dimensions may require Triton attention.
- Start with conservative memory allocation and increase only after empirical Kaggle profiling.
- On model preflight failure, fail closed before opening the ARC competition scorecard.
- No placeholder submission and no fabricated fallback score.

## Initial serving profile

```json
{
  "dtype": "bfloat16",
  "gpu_memory_utilization": 0.82,
  "max_model_len": 32768,
  "max_num_seqs": 2,
  "max_num_batched_tokens": 4096,
  "enable_prefix_caching": true,
  "trust_remote_code": false,
  "attention_backend": "auto"
}
```

These are starting values, not certified final values. The final profile must be selected from cold-start, VRAM, latency, long-context, vision, structured-output, and multi-request tests on the actual Kaggle RTX PRO 6000 image.

## Acceptance gates

- Offline installation from mounted inputs only.
- GPU is Blackwell compute capability 12.x and exposes sufficient memory.
- All native imports pass without undefined symbols.
- Gemma 4 31B loads from local files only.
- Three consecutive text generations pass.
- One image-text request passes.
- 8K, 16K, and 32K prompt probes pass within the deadline and VRAM budget.
- Structured JSON generation is schema-valid and deterministic enough for the verifier.
- No forced FlashInfer path.
- ARC host ACTION7 and RESET lifecycle round-trip pass.
- Real submission.parquet is generated atomically from the committed run.

Official/private ARC-AGI-3 score remains unverified until a committed Kaggle rerun completes.
