"""Deterministic paired benchmark for the v0.42 causal semantic learner."""
from __future__ import annotations

import importlib.util
import random
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class GameState(Enum):
    NOT_PLAYED = 0
    RUNNING = 1
    WIN = 2
    GAME_OVER = 3


class _Action:
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.reasoning = None

    def set_data(self, data):
        self.data = dict(data)

    def __repr__(self):
        return self.name


class GameAction:
    RESET = _Action("RESET")
    ACTION1 = _Action("ACTION1")
    ACTION2 = _Action("ACTION2")
    ACTION3 = _Action("ACTION3")
    ACTION4 = _Action("ACTION4")
    ACTION5 = _Action("ACTION5")
    ACTION6 = _Action("ACTION6")
    ACTION7 = _Action("ACTION7")


@dataclass
class FrameData:
    frame: list
    state: GameState
    levels_completed: int
    available_actions: tuple


class Agent:
    def __init__(self, *args, **kwargs):
        self.game_id = kwargs.get("game_id", "synthetic")

    @property
    def name(self):
        return "agent"


arcengine = types.ModuleType("arcengine")
arcengine.FrameData = FrameData
arcengine.GameAction = GameAction
arcengine.GameState = GameState
sys.modules["arcengine"] = arcengine
agents = types.ModuleType("agents")
agents_agent = types.ModuleType("agents.agent")
agents_agent.Agent = Agent
sys.modules["agents"] = agents
sys.modules["agents.agent"] = agents_agent

path = Path(__file__).resolve().parents[1] / "agent" / "my_agent.py"
spec = importlib.util.spec_from_file_location("my_agent_v42", path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)
MyAgent = module.MyAgent

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTIONS = (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4)


def render(h, w, player, goal, walls, mode_marker=0):
    grid = np.zeros((h, w), np.uint8)
    for y, x in walls:
        grid[y, x] = 2
    grid[goal] = 3
    grid[player] = 1
    if mode_marker:
        grid[0, w - 1] = mode_marker
    return [grid.tolist()]


def episode(seed: int, kind: str, policy: str) -> tuple[bool, int, int]:
    rng = random.Random(seed)
    h = w = 9
    walls = {(y, x) for y in range(h) for x in range(w) if rng.random() < 0.10}
    player, goal = (1, 1), (7, 7)
    walls.discard(player)
    walls.discard(goal)
    mapping0 = list(range(4))
    rng.shuffle(mapping0)
    mapping1 = list(range(4))
    rng.shuffle(mapping1)
    if mapping1 == mapping0:
        mapping1 = mapping1[1:] + mapping1[:1]
    agent = MyAgent(game_id=f"{kind}-{seed}") if policy == "v42" else None
    frames = []
    noops = 0
    for step in range(80):
        mode = (step // 8) % 2 if kind == "visible_modes" else 0
        if kind == "hidden_remap" and step >= 12:
            mode = 1
        mapping = mapping1 if mode else mapping0
        marker = 9 + mode if kind == "visible_modes" else 0
        frame = FrameData(render(h, w, player, goal, walls, marker), GameState.RUNNING, 0, ACTIONS)
        frames.append(frame)
        action = ACTIONS[rng.randrange(4)] if policy == "random" else agent.choose_action(frames, frame)
        logical = mapping[ACTIONS.index(action)]
        dy, dx = DIRS[logical]
        nxt = (player[0] + dy, player[1] + dx)
        if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in walls:
            player = nxt
        else:
            noops += 1
        if player == goal:
            return True, step + 1, noops
    return False, 80, noops


def run():
    results = {}
    for kind in ("stationary", "visible_modes", "hidden_remap"):
        for policy in ("random", "v42"):
            rows = [episode(1000 + i, kind, policy) for i in range(200)]
            results[f"{kind}_{policy}"] = {
                "solved": sum(row[0] for row in rows),
                "episodes": 200,
                "mean_steps": sum(row[1] for row in rows) / 200,
                "noops": sum(row[2] for row in rows),
            }
    return results


if __name__ == "__main__":
    import json

    report = run()
    print(json.dumps(report, indent=2, sort_keys=True))
    assert report["stationary_v42"]["solved"] >= report["stationary_random"]["solved"] + 80
    assert report["visible_modes_v42"]["solved"] >= report["visible_modes_random"]["solved"] + 60
    assert report["hidden_remap_v42"]["solved"] >= report["hidden_remap_random"]["solved"] + 50
