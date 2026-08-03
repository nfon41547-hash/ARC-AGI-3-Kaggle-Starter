"""Strict local verifier for the embedded ARC3 Pure NumPy runtime."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent"
EXPECTED = "1fda15b9725e388f9d98f9643cd9e1897803de6267f8102ac1f6af7936b71026"
FORBIDDEN = {"torch", "tensorflow", "jax", "vllm", "transformers"}


def main() -> None:
    for path in (AGENT / "my_agent.py", AGENT / "sovereign_payload.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    tree = ast.parse((AGENT / "my_agent.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    bad = sorted(FORBIDDEN & imported)
    if bad:
        raise SystemExit(f"forbidden competition imports: {bad}")

    sys.path.insert(0, str(AGENT))
    from sovereign_payload import _read_payload, ensure_runtime

    raw = _read_payload()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise SystemExit("payload digest mismatch")
    runtime = ensure_runtime()

    from arc3compiler.numpy_brain import PureNumpyCompetitionBrain

    PureNumpyCompetitionBrain()
    print(f"PASS runtime={runtime} sha256={EXPECTED} bytes={len(raw)}")


if __name__ == "__main__":
    main()
