"""Static and runtime quality gates for the competition agent."""
from __future__ import annotations
import ast, importlib.util, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; AGENT_DIR=ROOT/'agent'; RUNTIME_FILES=tuple(sorted(AGENT_DIR.glob('*.py'))); BENCHMARK=ROOT/'tests'/'v43_regression.py'
ALLOWED={'__future__','collections','dataclasses','hashlib','math','typing','numpy','arcengine','agents','sovereign_v43_types','sovereign_v43_memory','sovereign_v43_core'}
FORBIDDEN={'requests','urllib','socket','subprocess','os.system','torch','tensorflow','jax','vllm','transformers','pickle.loads','eval(','exec('}

def load_benchmark():
    sys.path.insert(0,str(BENCHMARK.parent)); spec=importlib.util.spec_from_file_location('quality_benchmark',BENCHMARK); module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader; sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

def run():
    sources={path:path.read_text(encoding='utf-8') for path in RUNTIME_FILES}; source='\n'.join(sources.values()); imports=set()
    for text in sources.values():
        tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(alias.name.split('.',1)[0] for alias in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split('.',1)[0])
    assert not sorted(imports-ALLOWED),sorted(imports-ALLOWED); assert not sorted(token for token in FORBIDDEN if token in source); assert len(source.encode())<64000
    benchmark=load_benchmark(); first=benchmark.run(); second=benchmark.run(); assert first==second
    memory=benchmark.module._SemanticMemory(max_modes=3)
    for index in range(12): memory.update(index.to_bytes(2,'little'),'ACTION1',benchmark.module._Outcome((1,0),1.0,False,True,False),None,0.0)
    assert len(memory.contexts)==3
    try: benchmark.module._to_grid([[16]])
    except ValueError: invalid=True
    else: invalid=False
    assert invalid
    return {'status':'PASS','runtime_files':[p.name for p in RUNTIME_FILES],'source_bytes':len(source.encode()),'source_lines':source.count('\n')+1,'imports':sorted(imports),'forbidden_dependencies':[],'deterministic_benchmark':True,'bounded_contexts':True,'invalid_grid_rejected':True,'official_score':'UNVERIFIED'}

if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
