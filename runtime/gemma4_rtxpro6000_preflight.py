"""Fail-closed offline preflight for Gemma 4 31B on RTX PRO 6000 Blackwell."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Iterable


def first_existing(candidates: Iterable[str]) -> Path:
    for raw in candidates:
        path = Path(raw)
        if path.is_dir():
            return path
    raise RuntimeError(f"none of the required dataset mounts exist: {list(candidates)!r}")


def verify_sha256_manifest(root: Path, filename: str = "SHA256SUMS") -> int:
    manifest = root / filename
    if not manifest.is_file():
        raise RuntimeError(f"missing integrity manifest: {manifest}")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = (root / relative).resolve()
        if root.resolve() not in path.parents and path != root.resolve():
            raise RuntimeError(f"unsafe manifest path: {relative}")
        if not path.is_file():
            raise RuntimeError(f"manifest file missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != digest.lower():
            raise RuntimeError(f"checksum mismatch: {relative}")
        checked += 1
    if checked == 0:
        raise RuntimeError(f"empty integrity manifest: {manifest}")
    return checked


def run(config_path: str | os.PathLike[str]) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model_dir = first_existing(config["model_mount_candidates"])
    wheelhouse = first_existing(config["wheelhouse_mount_candidates"])

    verify_sha256_manifest(model_dir.parent)
    verify_sha256_manifest(wheelhouse.parent)

    required_model_files = [
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    missing = [name for name in required_model_files if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Gemma model dataset is incomplete: {missing}")
    if not list(model_dir.glob("model-*.safetensors")):
        raise RuntimeError("Gemma model shards are missing")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.cuda.get_device_properties(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    total_gib = device.total_memory / (1024 ** 3)
    required = config["gpu"]
    if capability < tuple(required["minimum_compute_capability"]):
        raise RuntimeError(f"compute capability {capability} is below the required minimum")
    if total_gib < float(required["minimum_vram_gib"]):
        raise RuntimeError(f"GPU memory {total_gib:.2f} GiB is below the required minimum")
    if not any(token.lower() in device.name.lower() for token in required["name_contains"]):
        raise RuntimeError(f"unexpected GPU: {device.name}")

    packages = {}
    for package in ("torch", "torchvision", "vllm", "transformers", "triton", "safetensors"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required package is missing: {package}") from exc

    os.environ["ARC3_GEMMA_MODEL_DIR"] = str(model_dir)
    os.environ["ARC3_GEMMA_WHEELHOUSE"] = str(wheelhouse)
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)

    report = {
        "model_dir": str(model_dir),
        "wheelhouse": str(wheelhouse),
        "gpu_name": device.name,
        "gpu_memory_gib": round(total_gib, 3),
        "compute_capability": list(capability),
        "torch_cuda": torch.version.cuda,
        "packages": packages,
        "attention_backend": "auto",
        "status": "PASS",
    }
    return report


if __name__ == "__main__":
    config_path = os.environ.get("ARC3_GEMMA_CONFIG", "config/gemma4_rtxpro6000.json")
    print(json.dumps(run(config_path), indent=2, sort_keys=True))
