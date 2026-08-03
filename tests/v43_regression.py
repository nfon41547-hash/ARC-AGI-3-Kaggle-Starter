"""Focused v0.43 intelligence and legality regression suite."""
from __future__ import annotations
import json, random
from v43_fixture import *

def retention(seed,policy='v43'):
    rng=random.Random(seed); h=w=8; mapping=list(range(4)); rng.shuffle(mapping); agent=MyAgent(game_id=f'retain-{seed}') if policy=='v43' else None
    total=noops=illegal=0
    for life in range(3):
        actor={(1,1)}; goal={(6,6)}; walls={(2,2),(2,3),(3,2),(4,5),(5,5)}; frames=[]
        for _ in range(16+life*4):
            frame=FrameData(render(h,w,actor,goal,walls),GameState.RUNNING,0,ACTIONS); frames.append(frame)
            action=ACTIONS[rng.randrange(4)] if policy=='random' else agent.choose_action(frames,frame); total+=1
            if action not in ACTIONS: illegal+=1; continue
            target=translate(actor,DIRS[mapping[ACTIONS.index(action)]])
            if can_place(target,h,w,walls): actor=target
            else: noops+=1
            if actor&goal: return True,total,noops,illegal
        if policy=='v43':
            dead=FrameData(render(h,w,actor,goal,walls),GameState.GAME_OVER,0,ACTIONS); assert agent.choose_action(frames,dead) is GameAction.RESET
    return False,total,noops,illegal

def roles(seed):
    rng=random.Random(seed); h=w=8; actor={(1,1)}; goal={(6,6)}; walls={(3,2),(3,3),(4,4)}; colors=list(range(4,16)); rng.shuffle(colors)
    mapping=list(range(4)); rng.shuffle(mapping); agent=MyAgent(game_id=f'roles-{seed}'); frames=[]; noops=illegal=0
    for step in range(90):
        frame=FrameData(render(h,w,actor,goal,walls,actor_color=colors[0],goal_color=colors[1]),GameState.RUNNING,0,ACTIONS); frames.append(frame)
        action=agent.choose_action(frames,frame)
        if action not in ACTIONS: illegal+=1; continue
        target=translate(actor,DIRS[mapping[ACTIONS.index(action)]])
        if can_place(target,h,w,walls): actor=target
        else: noops+=1
        if actor&goal: return True,step+1,noops,illegal
    return False,90,noops,illegal

def restricted(seed):
    rng=random.Random(seed); h=w=7; actor={(1,1)}; goal={(5,5)}; mapping=list(range(4)); rng.shuffle(mapping); agent=MyAgent(game_id=f'legal-{seed}'); frames=[]; illegal=0
    for step in range(80):
        useful=1 if next(iter(actor))[0]<5 else 3; chosen=ACTIONS[mapping.index(useful)]; other=[a for a in ACTIONS if a is not chosen][step%3]; available=(chosen,other)
        frame=FrameData(render(h,w,actor,goal,set()),GameState.RUNNING,0,available); frames.append(frame); action=agent.choose_action(frames,frame)
        if action not in available: illegal+=1; continue
        actor=translate(actor,DIRS[mapping[ACTIONS.index(action)]])
        if actor&goal: return True,illegal
    return False,illegal

def click(seed):
    rng=random.Random(seed); grid=np.zeros((10,10),np.uint8); grid[1:4,1:4]=2; grid[6:9,1:4]=4; grid[1:4,6:9]=5; target=(rng.choice((6,7,8)),rng.choice((6,7,8))); grid[target]=9
    agent=MyAgent(game_id=f'click-{seed}'); frames=[]; seen=set()
    for step in range(12):
        frame=FrameData([grid.tolist()],GameState.RUNNING,0,(GameAction.ACTION6,)); frames.append(frame); action=agent.choose_action(frames,frame)
        if action is not GameAction.ACTION6 or not action.data: return False,step+1,len(seen)
        point=(int(action.data['y']),int(action.data['x']))
        if point in seen: return False,step+1,len(seen)
        seen.add(point)
        if point==target: return True,step+1,len(seen)
    return False,12,len(seen)

def run():
    out={}
    for kind in ('stationary','visible_modes','hidden_remap'):
        for policy in ('random','v43'): out[f'{kind}_{policy}']=summary([navigation(1000+i,kind,policy) for i in range(200)])
    for policy in ('random','v43'):
        out[f'footprint_{policy}']=summary([footprint(3000+i,policy) for i in range(120)])
        out[f'retention_{policy}']=summary([retention(5000+i,policy) for i in range(120)])
    out['role_inference_v43']=summary([roles(6500+i) for i in range(120)])
    legal=[restricted(7000+i) for i in range(100)]; out['restricted_v43']={'solved':sum(x[0] for x in legal),'episodes':100,'illegal':sum(x[1] for x in legal)}
    clicks=[click(9000+i) for i in range(100)]; out['click_v43']={'solved':sum(x[0] for x in clicks),'episodes':100,'mean_clicks':sum(x[1] for x in clicks)/100,'unique_clicks':sum(x[2] for x in clicks)}
    out['scope']='Controlled synthetic navigation/click tasks; not an official ARC-AGI-3 score.'
    return out

if __name__=='__main__':
    report=run(); print(json.dumps(report,indent=2,sort_keys=True))
    for key,threshold in {'stationary_v43':180,'visible_modes_v43':185,'hidden_remap_v43':170,'footprint_v43':100,'retention_v43':90,'role_inference_v43':100}.items(): assert report[key]['solved']>=threshold
    assert report['restricted_v43']['illegal']==0 and report['click_v43']['solved']>=90
    for value in report.values():
        if isinstance(value,dict) and 'illegal' in value: assert value['illegal']==0
