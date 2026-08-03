from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Minimal import stubs: the benchmark exercises SovereignNumpyCore directly.
arcengine = types.ModuleType("arcengine")
arcengine.FrameData = object
arcengine.GameAction = object
arcengine.GameState = object
sys.modules.setdefault("arcengine", arcengine)
agents = types.ModuleType("agents")
agent_module = types.ModuleType("agents.agent")
agent_module.Agent = object
sys.modules.setdefault("agents", agents)
sys.modules.setdefault("agents.agent", agent_module)

spec = importlib.util.spec_from_file_location("candidate_agent", ROOT / "agent" / "my_agent.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Core = module.SovereignNumpyCore


@dataclass
class Observation:
    grid: np.ndarray
    levels_completed: int = 0
    available_actions: tuple[int, ...] = (1, 2, 3, 4, 5)

    @property
    def frame(self) -> list[list[list[int]]]:
        return [self.grid.tolist()]


def render_nav(walls: np.ndarray, player: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    grid = np.zeros_like(walls, dtype=np.uint8)
    grid[walls] = 2
    grid[goal[1], goal[0]] = 3
    grid[player[1], player[0]] = 1
    return grid


def make_connected_case(seed: int, size: int = 11) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    rng = np.random.default_rng(seed)
    for _ in range(100):
        walls = rng.random((size, size)) < 0.18
        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True
        player = (1, 1)
        goal = (size - 2, size - 2)
        walls[player[1], player[0]] = False
        walls[goal[1], goal[0]] = False
        queue = [player]
        seen = {player}
        for x, y in queue:
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nxt = (x + dx, y + dy)
                if nxt in seen or walls[nxt[1], nxt[0]]:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        if goal in seen:
            return walls, player, goal
    raise RuntimeError("failed to generate connected maze")


def action_map(seed: int) -> dict[int, tuple[int, int]]:
    rng = np.random.default_rng(seed)
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    rng.shuffle(directions)
    return {code: directions[code - 1] for code in range(1, 5)} | {5: (0, 0)}


def run_nav(
    seed: int,
    adaptive: bool,
    remap_step: int | None = None,
    limit: int = 80,
) -> tuple[bool, int, int]:
    walls, player, goal = make_connected_case(seed)
    mapping = action_map(seed + 1000)
    remapped = action_map(seed + 2000)
    core = Core(f"synthetic-{seed}") if adaptive else None
    rng = np.random.default_rng(seed + 3000)
    illegal = 0
    levels = 0
    for step in range(limit):
        grid = render_nav(walls, player, goal)
        obs = Observation(grid, levels_completed=levels)
        if adaptive:
            decision = core.decide(obs)
            code = decision.action_code
        else:
            code = int(rng.choice(obs.available_actions))
        if code not in obs.available_actions:
            illegal += 1
            continue
        current_map = remapped if remap_step is not None and step >= remap_step else mapping
        dx, dy = current_map.get(code, (0, 0))
        nxt = (player[0] + dx, player[1] + dy)
        if not walls[nxt[1], nxt[0]]:
            player = nxt
        if player == goal:
            levels = 1
            final = Observation(render_nav(walls, player, goal), levels_completed=levels)
            if adaptive:
                core.decide(final)
            return True, step + 1, illegal
    return False, limit, illegal


def render_click(target: tuple[int, int], decoys: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((12, 12), dtype=np.uint8)
    grid[target[1], target[0]] = 3
    for x, y in decoys:
        grid[y : y + 2, x : x + 2] = 4
    return grid


def run_click(seed: int, adaptive: bool, limit: int = 12) -> tuple[bool, int, int]:
    rng = np.random.default_rng(seed)
    target = (int(rng.integers(1, 11)), int(rng.integers(1, 11)))
    decoys = [(1, 1), (8, 1), (1, 8), (8, 8)]
    decoys = [
        p
        for p in decoys
        if not (p[0] <= target[0] < p[0] + 2 and p[1] <= target[1] < p[1] + 2)
    ]
    grid = render_click(target, decoys)
    core = Core(f"click-{seed}") if adaptive else None
    illegal = 0
    for step in range(limit):
        obs = Observation(grid, available_actions=(6,))
        if adaptive:
            decision = core.decide(obs)
            x, y = int(decision.x), int(decision.y)
        else:
            x, y = int(rng.integers(0, 12)), int(rng.integers(0, 12))
        if not (0 <= x < 12 and 0 <= y < 12):
            illegal += 1
            continue
        if (x, y) == target:
            if adaptive:
                won = grid.copy()
                won[target[1], target[0]] = 5
                core.decide(Observation(won, levels_completed=1, available_actions=(6,)))
            return True, step + 1, illegal
    return False, limit, illegal


def aggregate(kind: str, episodes: int = 200) -> dict[str, Any]:
    adaptive_rows = []
    baseline_rows = []
    for seed in range(episodes):
        if kind == "navigation":
            adaptive_rows.append(run_nav(seed, True))
            baseline_rows.append(run_nav(seed, False))
        elif kind == "hidden_remap":
            adaptive_rows.append(run_nav(seed, True, remap_step=10, limit=100))
            baseline_rows.append(run_nav(seed, False, remap_step=10, limit=100))
        elif kind == "click":
            adaptive_rows.append(run_click(seed, True))
            baseline_rows.append(run_click(seed, False))
        else:
            raise ValueError(kind)

    def summarize(rows: list[tuple[bool, int, int]]) -> dict[str, Any]:
        wins = sum(int(row[0]) for row in rows)
        return {
            "episodes": len(rows),
            "wins": wins,
            "completion_rate": wins / len(rows),
            "mean_actions": float(np.mean([row[1] for row in rows])),
            "illegal_actions": int(sum(row[2] for row in rows)),
        }

    adaptive = summarize(adaptive_rows)
    baseline = summarize(baseline_rows)
    return {
        "condition": kind,
        "adaptive": adaptive,
        "random": baseline,
        "absolute_completion_gain": adaptive["completion_rate"] - baseline["completion_rate"],
    }


def main() -> None:
    report = {
        "policy": module.POLICY_VERSION,
        "seed_range": [0, 199],
        "conditions": [
            aggregate(name) for name in ("navigation", "hidden_remap", "click")
        ],
    }
    output = ROOT / "evidence" / "intelligence_benchmark_v42.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
