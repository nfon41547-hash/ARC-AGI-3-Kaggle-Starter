"""Minimal deterministic ARC3 test harness for Sovereign NumPy v0.43."""
from __future__ import annotations
import importlib.util, random, sys, types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable
import numpy as np

class GameState(Enum):
    NOT_PLAYED=0; RUNNING=1; WIN=2; GAME_OVER=3
class _Action:
    def __init__(self,name): self.name=name; self.data=None; self.reasoning=None
    def set_data(self,data): self.data=dict(data)
    def __repr__(self): return self.name
class GameAction:
    RESET=_Action('RESET'); ACTION1=_Action('ACTION1'); ACTION2=_Action('ACTION2')
    ACTION3=_Action('ACTION3'); ACTION4=_Action('ACTION4'); ACTION5=_Action('ACTION5')
    ACTION6=_Action('ACTION6'); ACTION7=_Action('ACTION7')
@dataclass
class FrameData:
    frame:list; state:GameState; levels_completed:int; available_actions:tuple
class Agent:
    def __init__(self,*args,**kwargs): self.game_id=kwargs.get('game_id','synthetic'); self.arc_env=kwargs.get('arc_env')
    @property
    def name(self): return 'agent'
arcengine=types.ModuleType('arcengine'); arcengine.FrameData=FrameData; arcengine.GameAction=GameAction; arcengine.GameState=GameState
agents=types.ModuleType('agents'); agents_agent=types.ModuleType('agents.agent'); agents_agent.Agent=Agent
sys.modules['arcengine']=arcengine; sys.modules['agents']=agents; sys.modules['agents.agent']=agents_agent
AGENT_DIR=Path(__file__).resolve().parents[1]/'agent'; sys.path.insert(0,str(AGENT_DIR))
spec=importlib.util.spec_from_file_location('my_agent_v43',AGENT_DIR/'my_agent.py'); module=importlib.util.module_from_spec(spec)
assert spec and spec.loader; sys.modules[spec.name]=module; spec.loader.exec_module(module)
MyAgent=module.MyAgent
DIRS=((-1,0),(1,0),(0,-1),(0,1)); ACTIONS=(GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4)

def render(h,w,actor,goal,walls,marker=0,actor_color=1,goal_color=3):
    grid=np.zeros((h,w),np.uint8)
    for y,x in walls: grid[y,x]=2
    for y,x in goal: grid[y,x]=goal_color
    for y,x in actor: grid[y,x]=actor_color
    if marker: grid[0,w-1]=marker
    return [grid.tolist()]
def translate(cells,delta):
    dy,dx=delta; return {(y+dy,x+dx) for y,x in cells}
def can_place(cells,h,w,walls): return all(0<=y<h and 0<=x<w and (y,x) not in walls for y,x in cells)
def summary(rows):
    return {'solved':sum(r[0] for r in rows),'episodes':len(rows),'mean_steps':sum(r[1] for r in rows)/len(rows),'noops':sum(r[2] for r in rows),'illegal':sum(r[3] for r in rows)}

def navigation(seed,kind,policy='v43'):
    rng=random.Random(seed); h=w=9; walls={(y,x) for y in range(h) for x in range(w) if rng.random()<.10}
    actor={(1,1)}; goal={(7,7)}; walls-=actor|goal; m0=list(range(4)); rng.shuffle(m0); m1=list(range(4)); rng.shuffle(m1)
    if m1==m0: m1=m1[1:]+m1[:1]
    agent=MyAgent(game_id=f'{kind}-{seed}') if policy=='v43' else None; frames=[]; noops=illegal=0
    for step in range(80):
        mode=(step//8)%2 if kind=='visible_modes' else int(kind=='hidden_remap' and step>=12); mapping=m1 if mode else m0
        frame=FrameData(render(h,w,actor,goal,walls,9+mode if kind=='visible_modes' else 0),GameState.RUNNING,0,ACTIONS); frames.append(frame)
        action=ACTIONS[rng.randrange(4)] if policy=='random' else agent.choose_action(frames,frame)
        if action not in ACTIONS: illegal+=1; continue
        target=translate(actor,DIRS[mapping[ACTIONS.index(action)]])
        if can_place(target,h,w,walls): actor=target
        else: noops+=1
        if actor&goal: return True,step+1,noops,illegal
    return False,80,noops,illegal

def footprint(seed,policy='v43'):
    rng=random.Random(seed); h=w=11; actor={(1,1),(1,2),(2,1),(2,2)}; goal={(8,8),(8,9),(9,8),(9,9)}
    walls={(y,5) for y in range(h)}-{(3,5),(7,5),(8,5)}; mapping=list(range(4)); rng.shuffle(mapping)
    agent=MyAgent(game_id=f'foot-{seed}') if policy=='v43' else None; frames=[]; noops=illegal=0
    for step in range(100):
        frame=FrameData(render(h,w,actor,goal,walls),GameState.RUNNING,0,ACTIONS); frames.append(frame)
        action=ACTIONS[rng.randrange(4)] if policy=='random' else agent.choose_action(frames,frame)
        if action not in ACTIONS: illegal+=1; continue
        target=translate(actor,DIRS[mapping[ACTIONS.index(action)]])
        if can_place(target,h,w,walls): actor=target
        else: noops+=1
        if actor&goal: return True,step+1,noops,illegal
    return False,100,noops,illegal
