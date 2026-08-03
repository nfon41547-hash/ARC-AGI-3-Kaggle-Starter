"""ARC3 Sovereign NumPy v0.43 competition adapter."""
from __future__ import annotations
from typing import Any
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
try:
    from .sovereign_v43_types import _available, _make_action, _to_grid
    from .sovereign_v43_memory import _Outcome, _SemanticMemory
    from .sovereign_v43_core import _CognitiveCore
except ImportError:
    from sovereign_v43_types import _available, _make_action, _to_grid
    from sovereign_v43_memory import _Outcome, _SemanticMemory
    from sovereign_v43_core import _CognitiveCore

class MyAgent(Agent):
    """Competition adapter with persistent verified semantics across resets."""

    MAX_ACTIONS = 400
    MAX_RECOVERIES = 4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.core = _CognitiveCore()
        self.recoveries = 0

    @property
    def name(self) -> str:
        return f"{super().name}.sovereign_numpy_v43"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _fallback_actions(self) -> Any:
        environment = getattr(self, "arc_env", None)
        return getattr(environment, "action_space", None)

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        fallback = self._fallback_actions()
        available = _available(latest_frame, fallback)

        if latest_frame.state is GameState.NOT_PLAYED:
            self.core.begin_level(retain_semantics=True)
            return GameAction.RESET

        if latest_frame.state is GameState.GAME_OVER:
            try:
                grid = _to_grid(latest_frame)
                levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
                self.core.observe(grid, levels, terminal_loss=True)
            except Exception:
                pass
            self.core.begin_level(retain_semantics=True)
            return GameAction.RESET

        try:
            grid = _to_grid(latest_frame)
            levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
            self.core.observe(grid, levels)
            action, coord, reason = self.core.choose(grid, available, levels)
            self.recoveries = 0
            return _make_action(action, coord, reason)
        except Exception as exc:
            self.recoveries += 1
            if self.recoveries > self.MAX_RECOVERIES:
                raise RuntimeError("bounded ARC3 recovery budget exhausted") from exc
            grid = _to_grid(latest_frame)
            action, coord = self.core.deterministic_recovery(grid, available)
            return _make_action(action, coord, f"bounded-recovery:{type(exc).__name__}")
