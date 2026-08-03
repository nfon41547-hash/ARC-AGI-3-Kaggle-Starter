from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_notebook.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("arc3_notebook_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builder_produces_cpu_offline_notebook() -> None:
    module = load_builder()
    notebook = module.build()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 5
    kaggle = notebook["metadata"]["kaggle"]
    assert kaggle["accelerator"] == "none"
    assert kaggle["isGpuEnabled"] is False
    assert kaggle["isInternetEnabled"] is False


def test_notebook_contains_strict_current_contract() -> None:
    module = load_builder()
    notebook = module.build()
    code = "\n".join(
        str(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    required = (
        '"arc-agi==0.9.9"',
        '"arcengine==0.9.3"',
        "--no-index",
        "--no-cache-dir",
        "KAGGLE_IS_COMPETITION_RERUN",
        "/tmp/my_agent.py",
        "OPERATION_MODE=competition",
        "PYTHONHASHSEED=0",
        "OMP_NUM_THREADS=1",
        "submission.parquet",
    )
    assert all(token in code for token in required)
    assert "/kaggle/working/my_agent.py" not in code


def test_notebook_json_roundtrip_and_single_output_contract() -> None:
    module = load_builder()
    notebook = module.build()
    encoded = json.dumps(notebook, sort_keys=True)
    assert json.loads(encoded) == notebook
    code = "\n".join(str(cell.get("source", "")) for cell in notebook["cells"])
    assert code.count("to_parquet(") == 1
    assert "'/kaggle/working/submission.parquet'" in code
    assert "%%writefile /tmp/my_agent.py" in code
    assert "%%writefile /kaggle/working/" not in code


def test_kernel_metadata_is_cpu_offline() -> None:
    metadata = json.loads((ROOT / "notebooks" / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["code_file"] == "submission.ipynb"
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-3"]
