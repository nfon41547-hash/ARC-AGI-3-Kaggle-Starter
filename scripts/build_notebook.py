"""Build the CPU-only offline Kaggle notebook for Sovereign NumPy v0.45."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
AGENT_FILES = ("sovereign_v45_bundle.py", "my_agent.py")
RUNTIME_ZIP = AGENT_DIR / "sovereign_v45_runtime.zip"
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


def _write_cells() -> list[dict]:
    cells = []
    if not RUNTIME_ZIP.is_file():
        raise SystemExit(f"missing runtime bundle: {RUNTIME_ZIP}")
    encoded = base64.b85encode(RUNTIME_ZIP.read_bytes()).decode("ascii")
    cells.append(code_cell(
        "import base64\n"
        f"open('/tmp/{RUNTIME_ZIP.name}', 'wb').write(base64.b85decode({encoded!r}))"
    ))
    for filename in AGENT_FILES:
        path = AGENT_DIR / filename
        if not path.is_file():
            raise SystemExit(f"missing runtime source: {path}")
        cells.append(code_cell(f"%%writefile /tmp/{filename}\n" + path.read_text(encoding="utf-8")))
    return cells


def build() -> dict:
    install = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )
    init_content = dedent(
        """\
        from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent
        load_dotenv()
        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {'random': Random, 'myagent': MyAgent}
        """
    )
    env_content = dedent(
        """\
        SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        """
    )
    run_source = f"""import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents
    !cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    !cp /tmp/sovereign_v45_bundle.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/sovereign_v45_bundle.py
    !cp /tmp/sovereign_v45_runtime.zip /kaggle/working/ARC-AGI-3-Agents/agents/templates/sovereign_v45_runtime.zip
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as handle:
        handle.write({init_content!r})
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as handle:
        handle.write({env_content!r})
    !cd /kaggle/working/ARC-AGI-3-Agents && ARC3_TRACE_GIF='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=agg python main.py --agent myagent
"""
    run = code_cell(run_source)
    dummy = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'],
                )
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": "none",
                "isInternetEnabled": False,
                "isGpuEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — Sovereign NumPy v0.45\n\n"
                "Real-time frame-sequence perception and online rule induction. "
                "Public game solutions are not embedded in the competition runtime."
            ),
            install,
            *_write_cells(),
            run,
            dummy,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    else:
        metadata = {
            "id": "REPLACE_WITH_YOUR_USERNAME/arc3-sovereign-numpy-v45",
            "title": "ARC3 Sovereign NumPy v0.45",
            "code_file": "submission.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "competition_sources": ["arc-prize-2026-arc-agi-3"],
        }
    metadata.update({"enable_gpu": False, "enable_internet": False})
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[build_notebook] wrote {NOTEBOOK_PATH.relative_to(ROOT)} (cpu, internet-off)")


if __name__ == "__main__":
    main()
