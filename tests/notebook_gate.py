"""Validate the generated Kaggle notebook profile without executing Kaggle I/O."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'scripts'/'build_notebook.py'
spec=importlib.util.spec_from_file_location('builder',path)
module=importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
assert module.ACCELERATOR=='cpu'
notebook=module.build()
assert notebook['metadata']['kaggle']['isGpuEnabled'] is False
assert notebook['metadata']['kaggle']['isInternetEnabled'] is False
assert len(notebook['cells'])==8
source='\n'.join(''.join(cell.get('source','')) for cell in notebook['cells'])
for token in ('sovereign_numpy_v43','sovereign_v43_types.py','sovereign_v43_memory.py','sovereign_v43_core.py'):
    assert token in source,token
print({'status':'PASS','cells':len(notebook['cells']),'accelerator':'cpu','internet':False})
