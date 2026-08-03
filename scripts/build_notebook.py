"""Build the offline Kaggle notebook from the modular v0.43 agent sources."""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

ACCELERATOR="cpu"
_ACCELERATORS={"cpu":{"name":"none","gpu":False},"t4":{"name":"nvidiaTeslaT4","gpu":True},"p100":{"name":"nvidiaTeslaP100","gpu":True},"rtx6000":{"name":"nvidiaRtx6000","gpu":True}}
ROOT=Path(__file__).resolve().parents[1]; AGENT_DIR=ROOT/'agent'; AGENT_FILES=('sovereign_v43_types.py','sovereign_v43_memory.py','sovereign_v43_core.py','my_agent.py')
NOTEBOOK_PATH=ROOT/'notebooks'/'submission.ipynb'; METADATA_PATH=ROOT/'notebooks'/'kernel-metadata.json'

def code_cell(source): return {'cell_type':'code','metadata':{'trusted':True},'outputs':[],'execution_count':None,'source':source}
def markdown_cell(source): return {'cell_type':'markdown','metadata':{},'source':source}
def _agent_write_cells():
    cells=[]
    for filename in AGENT_FILES:
        path=AGENT_DIR/filename
        if not path.is_file(): raise SystemExit(f'Could not find {path}')
        cells.append(code_cell(f'%%writefile /tmp/{filename}\n'+path.read_text(encoding='utf-8')))
    return cells

def build():
    if ACCELERATOR not in _ACCELERATORS: raise SystemExit(f'Unknown ACCELERATOR={ACCELERATOR!r}')
    accelerator=_ACCELERATORS[ACCELERATOR]
    install=code_cell("!pip install --no-index --find-links \\\n    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n    arc-agi python-dotenv")
    run=code_cell(dedent("""\
        import os
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents
            !cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
            !cp /tmp/sovereign_v43_types.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/sovereign_v43_types.py
            !cp /tmp/sovereign_v43_memory.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/sovereign_v43_memory.py
            !cp /tmp/sovereign_v43_core.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/sovereign_v43_core.py
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py','w') as handle:
                handle.write("""from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent
        load_dotenv()
        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {'random': Random, 'myagent': MyAgent}
        """)
            with open('/kaggle/working/ARC-AGI-3-Agents/.env','w') as handle:
                handle.write("""SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        """)
            !cd /kaggle/working/ARC-AGI-3-Agents && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=agg python main.py --agent myagent
        """))
    dummy=code_cell(dedent("""\
        import os
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            import pandas as pd
            submission=pd.DataFrame(data=[['1_0','1',True,1]],columns=['row_id','game_id','end_of_game','score'])
            submission.to_parquet('/kaggle/working/submission.parquet',index=False)
            submission.head()
        """))
    return {'metadata':{'kernelspec':{'language':'python','display_name':'Python 3','name':'python3'},'language_info':{'name':'python','mimetype':'text/x-python','file_extension':'.py','pygments_lexer':'ipython3'},'kaggle':{'accelerator':accelerator['name'],'isInternetEnabled':False,'isGpuEnabled':accelerator['gpu'],'language':'python','sourceType':'notebook'}},'nbformat_minor':4,'nbformat':4,'cells':[markdown_cell('# ARC Prize 2026 — ARC-AGI-3 Sovereign NumPy v0.43\n\nGenerated from verified modular agent sources.'),install,*_agent_write_cells(),run,dummy]}

def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True,exist_ok=True); NOTEBOOK_PATH.write_text(json.dumps(build(),indent=1),encoding='utf-8'); print(f'[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)} ({ACCELERATOR})')
    if METADATA_PATH.exists():
        metadata=json.loads(METADATA_PATH.read_text(encoding='utf-8')); metadata['enable_gpu']=_ACCELERATORS[ACCELERATOR]['gpu']; metadata['enable_internet']=False; METADATA_PATH.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__': main()
