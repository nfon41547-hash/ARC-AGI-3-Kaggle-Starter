"""Deterministic CPU latency benchmark for the v0.43 agent."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from time import perf_counter_ns
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; FIXTURE=ROOT/'tests'/'v43_fixture.py'; sys.path.insert(0,str(FIXTURE.parent))
spec=importlib.util.spec_from_file_location('performance_fixture',FIXTURE); fixture=importlib.util.module_from_spec(spec)
assert spec and spec.loader; sys.modules[spec.name]=fixture; spec.loader.exec_module(fixture)

def run(iterations=500):
    h=w=32; actor={(1,1)}; goal={(30,30)}; walls={(y,16) for y in range(2,30) if y not in (8,9,22,23)}; mapping=(2,0,3,1)
    agent=fixture.MyAgent(game_id='latency-v43'); frames=[]; samples=[]; illegal=0
    for _ in range(iterations):
        frame=fixture.FrameData(fixture.render(h,w,actor,goal,walls),fixture.GameState.RUNNING,0,fixture.ACTIONS); frames.append(frame); start=perf_counter_ns(); action=agent.choose_action(frames,frame); samples.append((perf_counter_ns()-start)/1e6)
        if action not in fixture.ACTIONS: illegal+=1; continue
        target=fixture.translate(actor,fixture.DIRS[mapping[fixture.ACTIONS.index(action)]])
        if fixture.can_place(target,h,w,walls): actor=target
        if actor&goal: actor={(1,1)}; frames.clear()
    values=np.asarray(samples,dtype=np.float64); report={'iterations':iterations,'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'p99_ms':float(np.percentile(values,99)),'max_ms':float(values.max()),'illegal_actions':illegal,'scope':'Local CPU synthetic 32x32 navigation; not Kaggle target hardware evidence.'}
    assert illegal==0 and report['p99_ms']<100.0; return report

if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
