"""Build the canonical Kaggle submission notebook from the sovereign agent."""
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
AGENT_DIR = ROOT / "agent"
AGENT_FILES = (
    "my_agent.py",
    "sovereign_payload.py",
    "sovereign_payload.part00.b85",
    "sovereign_payload.part01.b85",
    "sovereign_payload.part02.b85",
)
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


def _write_file_cell(name: str) -> dict:
    body = (AGENT_DIR / name).read_text(encoding="utf-8")
    return code_cell(f"%%writefile /tmp/arc3_agent/{name}\n" + body)


def build() -> dict:
    missing = [name for name in AGENT_FILES if not (AGENT_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"missing sovereign agent files: {missing}")

    install_cell = code_cell(dedent("""\
        import os, sys, subprocess
        wheelhouse = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-index', '--no-cache-dir',
            '--only-binary=:all:', '--find-links', wheelhouse,
            'arc-agi==0.9.9', 'arcengine==0.9.3', 'python-dotenv'
        ], check=True)
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    """))

    mkdir_cell = code_cell(
        "from pathlib import Path\n"
        "Path('/tmp/arc3_agent').mkdir(parents=True, exist_ok=True)"
    )

    run_cell = code_cell(dedent("""\
        import importlib.metadata
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path

        assert importlib.metadata.version('arc-agi') == '0.9.9'
        assert importlib.metadata.version('arcengine') == '0.9.3'

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            subprocess.run([
                'curl', '--fail', '--retry', '999', '--retry-all-errors',
                '--retry-delay', '5', '--retry-max-time', '600',
                'http://gateway:8001/api/games'
            ], check=True)

            framework_src = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents')
            framework_dst = Path('/kaggle/working/ARC-AGI-3-Agents')
            shutil.copytree(framework_src, framework_dst, dirs_exist_ok=True)
            templates = framework_dst / 'agents' / 'templates'
            templates.mkdir(parents=True, exist_ok=True)
            for source in Path('/tmp/arc3_agent').iterdir():
                if source.is_file():
                    shutil.copy2(source, templates / source.name)

            (framework_dst / 'agents' / '__init__.py').write_text(
                "from typing import Type\n"
                "from dotenv import load_dotenv\n"
                "from .agent import Agent, Playback\n"
                "from .swarm import Swarm\n"
                "from .templates.random_agent import Random\n"
                "from .templates.my_agent import MyAgent\n\n"
                "load_dotenv()\n\n"
                "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\n"
                "    'random': Random,\n"
                "    'myagent': MyAgent,\n"
                "}\n",
                encoding='utf-8',
            )
            (framework_dst / '.env').write_text(
                "SCHEME=http\nHOST=gateway\nPORT=8001\n"
                "ARC_API_KEY=test-key-123\nARC_BASE_URL=http://gateway:8001/\n"
                "OPERATION_MODE=online\nENVIRONMENTS_DIR=\n"
                "RECORDINGS_DIR=/kaggle/working/server_recording\n",
                encoding='utf-8',
            )
            env = os.environ.copy()
            env.update({
                'MPLBACKEND': 'agg',
                'OMP_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1',
                'PYTHONHASHSEED': '0',
            })
            subprocess.run(
                [sys.executable, 'main.py', '--agent', 'myagent'],
                cwd=framework_dst,
                env=env,
                check=True,
            )
    """))

    dummy_cell = code_cell(dedent("""\
        import os
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            import pandas as pd
            pd.DataFrame(
                data=[['1_0', '1', True, 1]],
                columns=['row_id', 'game_id', 'end_of_game', 'score'],
            ).to_parquet('/kaggle/working/submission.parquet', index=False)
    """))

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
                "# ARC3 Sovereign Pure NumPy v0.41\n\n"
                "Generated from versioned source; do not edit this notebook directly."
            ),
            install_cell,
            mkdir_cell,
            *[_write_file_cell(name) for name in AGENT_FILES],
            run_cell,
            dummy_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[build_notebook] wrote {NOTEBOOK_PATH.relative_to(ROOT)}")
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        metadata["enable_gpu"] = _ACCELERATORS[ACCELERATOR]["gpu"]
        metadata["enable_internet"] = False
        METADATA_PATH.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
