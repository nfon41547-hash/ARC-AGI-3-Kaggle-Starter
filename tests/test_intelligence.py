from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "benchmark_agent.py"
spec = importlib.util.spec_from_file_location("arc3_benchmark", BENCH)
assert spec and spec.loader
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def test_animation_sequence_uses_final_frame() -> None:
    first = [[0, 1], [0, 0]]
    final = [[0, 0], [1, 0]]
    obs = type("Obs", (), {"frame": [first, final]})()
    grid = bench.Core.final_grid(obs)
    assert grid.tolist() == final


def test_available_actions_are_strictly_respected() -> None:
    core = bench.Core("legality")
    grid = bench.render_click((5, 5), [(1, 1)])
    for _ in range(8):
        decision = core.decide(bench.Observation(grid, available_actions=(6,)))
        assert decision.action_code == 6
        assert 0 <= int(decision.x) < grid.shape[1]
        assert 0 <= int(decision.y) < grid.shape[0]


def test_navigation_is_clearly_better_than_random() -> None:
    adaptive = [bench.run_nav(seed, True) for seed in range(50)]
    random = [bench.run_nav(seed, False) for seed in range(50)]
    adaptive_rate = sum(row[0] for row in adaptive) / len(adaptive)
    random_rate = sum(row[0] for row in random) / len(random)
    assert adaptive_rate >= 0.94
    assert adaptive_rate - random_rate >= 0.85
    assert sum(row[2] for row in adaptive) == 0


def test_hidden_remap_self_repairs() -> None:
    adaptive = [
        bench.run_nav(seed, True, remap_step=10, limit=100)
        for seed in range(50)
    ]
    random = [
        bench.run_nav(seed, False, remap_step=10, limit=100)
        for seed in range(50)
    ]
    adaptive_rate = sum(row[0] for row in adaptive) / len(adaptive)
    random_rate = sum(row[0] for row in random) / len(random)
    assert adaptive_rate >= 0.88
    assert adaptive_rate - random_rate >= 0.75
    assert sum(row[2] for row in adaptive) == 0


def test_click_reasoning_rejects_large_decoys() -> None:
    adaptive = [bench.run_click(seed, True) for seed in range(50)]
    random = [bench.run_click(seed, False) for seed in range(50)]
    assert sum(row[0] for row in adaptive) == 50
    assert sum(row[0] for row in random) <= 12
    assert sum(row[2] for row in adaptive) == 0


def test_decisions_are_replay_deterministic() -> None:
    left = [bench.run_nav(seed, True) for seed in range(20)]
    right = [bench.run_nav(seed, True) for seed in range(20)]
    assert left == right


def test_source_has_no_random_or_network_policy() -> None:
    source = (ROOT / "agent" / "my_agent.py").read_text()
    forbidden = (
        "import random",
        "requests.",
        "urllib",
        "socket",
        "os.system",
        "subprocess",
        "eval(",
        "exec(",
    )
    assert not any(token in source for token in forbidden)
