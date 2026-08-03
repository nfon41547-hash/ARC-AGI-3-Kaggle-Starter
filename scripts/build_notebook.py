"""Build the canonical offline Kaggle notebook from ``agent/my_agent.py``.

The generated notebook follows Kaggle's two-phase code-competition pattern:
commit mode emits only a dummy ``submission.parquet``; competition rerun mode
copies the official framework, registers ``MyAgent``, and connects it to the
local gateway sidecar. The agent itself is Pure NumPy and CPU-only.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ACCELERATOR = "cpu"

_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def validate_agent_source(source: str) -> None:
    compile(source, str(AGENT_SRC), "exec")
    forbidden = (
        "import random",
        "requests.",
        "urllib",
        "socket",
        "subprocess",
        "os.system",
        "eval(",
        "exec(",
    )
    found = [token for token in forbidden if token in source]
    if found:
        raise SystemExit(f"agent source contains forbidden runtime tokens: {found}")
    if "class MyAgent" not in source or "SovereignNumpyCore" not in source:
        raise SystemExit("agent source is missing the canonical v42 classes")


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text(encoding="utf-8")
    validate_agent_source(agent_body)

    install_cell = code_cell(
        dedent(
            """\
            import os

            # Configure deterministic CPU execution before importing NumPy or
            # the official framework. The competition environment is offline.
            for key in (
                'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS',
            ):
                os.environ[key] = '1'
            os.environ['PYTHONHASHSEED'] = '0'
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['ARC3_POLICY_VERSION'] = 'sovereign-numpy-v42'

            !pip install --no-index --no-cache-dir \\
                --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\
                "arc-agi==0.9.9" "arcengine==0.9.3" python-dotenv

            import numpy as np
            assert np.__version__
            print('Pure NumPy runtime ready:', np.__version__)
            """
        )
    )

    # /tmp avoids exposing my_agent.py as a candidate Kaggle output file.
    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    run_cell = code_cell(
        dedent(
            """\
            import os

            if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                      --retry-max-time 600 http://gateway:8001/api/games

                !rm -rf /kaggle/working/ARC-AGI-3-Agents
                !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                       /kaggle/working/ARC-AGI-3-Agents
                !cp /tmp/my_agent.py \\
                    /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

                # Register only the dependencies required by this submission.
                with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                    f.write("""from typing import Type
            from dotenv import load_dotenv
            from .agent import Agent, Playback
            from .swarm import Swarm
            from .templates.random_agent import Random
            from .templates.my_agent import MyAgent

            load_dotenv()

            AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
                'random': Random,
                'myagent': MyAgent,
            }
            """)

                with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                    f.write("""SCHEME=http
            HOST=gateway
            PORT=8001
            ARC_API_KEY=test-key-123
            ARC_BASE_URL=http://gateway:8001/
            OPERATION_MODE=competition
            ENVIRONMENTS_DIR=
            RECORDINGS_DIR=/kaggle/working/server_recording
            """)

                !cd /kaggle/working/ARC-AGI-3-Agents && \\
                    PYTHONHASHSEED=0 \\
                    OMP_NUM_THREADS=1 \\
                    OPENBLAS_NUM_THREADS=1 \\
                    MKL_NUM_THREADS=1 \\
                    NUMEXPR_NUM_THREADS=1 \\
                    MPLBACKEND=agg \\
                    python main.py --agent myagent
            """
        )
    )

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'],
                )
                submission.to_parquet(
                    '/kaggle/working/submission.parquet',
                    index=False,
                )
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(f"Unknown ACCELERATOR={ACCELERATOR!r}")
    accel = _ACCELERATORS[ACCELERATOR]
    return {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC-AGI-3 — Sovereign Pure NumPy v42\n\n"
                "Generated from `agent/my_agent.py`. Edit source, run "
                "`make verify-intelligence`, then rebuild."
            ),
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(
        f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)} "
        f"(accelerator: {ACCELERATOR})"
    )
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        meta["enable_gpu"] = _ACCELERATORS[ACCELERATOR]["gpu"]
        meta["enable_internet"] = False
        meta["title"] = "ARC-AGI-3 Sovereign Pure NumPy v42"
        METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("[build_notebook] Synced CPU/offline kernel metadata")


if __name__ == "__main__":
    main()
