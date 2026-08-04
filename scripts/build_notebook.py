"""Build the offline GPU Kaggle notebook for Gemma 4 31B on RTX PRO 6000."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
AGENT_FILES = ("sovereign_v45_bundle.py", "gemma4_oracle.py", "my_agent.py")
RUNTIME_ZIP = AGENT_DIR / "sovereign_v45_runtime.zip"
CONFIG_PATH = ROOT / "config" / "gemma4_rtxpro6000.json"
PREFLIGHT_PATH = ROOT / "runtime" / "gemma4_rtxpro6000_preflight.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [], "execution_count": None, "source": source}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def writefile_cell(path: str, source: str) -> dict:
    return code_cell(f"%%writefile {path}\n" + source)


def embedded_files() -> list[dict]:
    if not RUNTIME_ZIP.is_file():
        raise SystemExit(f"missing runtime bundle: {RUNTIME_ZIP}")
    cells = [
        code_cell(
            "import base64\n"
            f"open('/tmp/{RUNTIME_ZIP.name}', 'wb').write(base64.b85decode({base64.b85encode(RUNTIME_ZIP.read_bytes()).decode('ascii')!r}))"
        )
    ]
    for filename in AGENT_FILES:
        path = AGENT_DIR / filename
        if not path.is_file():
            raise SystemExit(f"missing agent source: {path}")
        cells.append(writefile_cell(f"/tmp/{filename}", path.read_text(encoding="utf-8")))
    cells.append(writefile_cell("/tmp/gemma4_rtxpro6000.json", CONFIG_PATH.read_text(encoding="utf-8")))
    cells.append(writefile_cell("/tmp/gemma4_rtxpro6000_preflight.py", PREFLIGHT_PATH.read_text(encoding="utf-8")))
    return cells


def build() -> dict:
    install = code_cell(dedent("""\
        import json, os, subprocess, sys
        from pathlib import Path

        cfg = json.loads(Path('/tmp/gemma4_rtxpro6000.json').read_text())
        wheelhouse = next((Path(p) for p in cfg['wheelhouse_mount_candidates'] if Path(p).is_dir()), None)
        if wheelhouse is None:
            raise RuntimeError('Gemma/SM120 wheelhouse dataset is not mounted')
        official = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels')
        command = [
            sys.executable, '-m', 'pip', 'install', '--no-index',
            '--find-links', str(wheelhouse), '--find-links', str(official),
            'arc-agi', 'python-dotenv', 'torch', 'torchvision', 'vllm',
            'transformers', 'triton', 'safetensors', 'pillow',
        ]
        subprocess.check_call(command)
    """))

    preflight = code_cell(dedent("""\
        import importlib.util, json, os
        spec = importlib.util.spec_from_file_location('arc3_gemma_preflight', '/tmp/gemma4_rtxpro6000_preflight.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preflight = module.run('/tmp/gemma4_rtxpro6000.json')
        print(json.dumps(preflight, indent=2, sort_keys=True))
    """))

    start_server = code_cell(dedent("""\
        import os, subprocess, sys, time
        from urllib import request

        os.environ.pop('VLLM_ATTENTION_BACKEND', None)
        model_dir = preflight['model_dir']
        command = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', model_dir,
            '--served-model-name', 'arc3-gemma4-31b',
            '--dtype', 'bfloat16',
            '--gpu-memory-utilization', '0.82',
            '--max-model-len', '32768',
            '--max-num-seqs', '2',
            '--max-num-batched-tokens', '4096',
            '--enable-prefix-caching',
            '--host', '127.0.0.1', '--port', '8000',
        ]
        server_log = open('/kaggle/working/gemma4_server.log', 'w')
        gemma_server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if gemma_server.poll() is not None:
                raise RuntimeError('Gemma 4 server exited during startup')
            try:
                with request.urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            gemma_server.terminate()
            raise RuntimeError('Gemma 4 server readiness timeout')
        os.environ['ARC3_GEMMA_URL'] = 'http://127.0.0.1:8000'
        os.environ['ARC3_GEMMA_SERVED_NAME'] = 'arc3-gemma4-31b'
    """))

    init_content = dedent("""\
        from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent
        load_dotenv()
        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {'random': Random, 'myagent': MyAgent}
    """)
    env_content = dedent("""\
        SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
    """)
    run = code_cell(f"""import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print('Competition rerun is required; no placeholder submission is generated.')
else:
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents
    !cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    !cp /tmp/gemma4_oracle.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/gemma4_oracle.py
    !cp /tmp/sovereign_v45_bundle.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/sovereign_v45_bundle.py
    !cp /tmp/sovereign_v45_runtime.zip /kaggle/working/ARC-AGI-3-Agents/agents/templates/sovereign_v45_runtime.zip
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as handle:
        handle.write({init_content!r})
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as handle:
        handle.write({env_content!r})
    !cd /kaggle/working/ARC-AGI-3-Agents && ARC3_TRACE_GIF='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=agg python main.py --agent myagent
""")

    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python", "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": False, "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell("# ARC Prize 2026 — Gemma 4 31B + RTX PRO 6000\n\nOffline dataset-mounted model and SM120 runtime. The deterministic ARC core remains authoritative; Gemma is proposal-only."),
            *embedded_files(), install, preflight, start_server, run,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    metadata.update({
        "title": "ARC3 Gemma 4 31B RTX PRO 6000 v0.46",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
    })
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[build_notebook] wrote {NOTEBOOK_PATH.relative_to(ROOT)} (gpu, internet-off, Gemma 4 31B)")


if __name__ == "__main__":
    main()
