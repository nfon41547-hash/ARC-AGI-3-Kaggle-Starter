from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"


def test_agent_has_no_heavy_ml_imports() -> None:
    tree = ast.parse((AGENT / "my_agent.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not ({"torch", "tensorflow", "jax", "vllm", "transformers"} & imported)


def test_embedded_runtime_roundtrip() -> None:
    sys.path.insert(0, str(AGENT))
    from sovereign_payload import _RUNTIME_SHA256, _read_payload, ensure_runtime

    raw = _read_payload()
    assert hashlib.sha256(raw).hexdigest() == _RUNTIME_SHA256
    runtime = ensure_runtime()
    assert (runtime / "arc3compiler" / "numpy_brain.py").is_file()

    from arc3compiler.numpy_brain import PureNumpyCompetitionBrain

    assert PureNumpyCompetitionBrain() is not None
