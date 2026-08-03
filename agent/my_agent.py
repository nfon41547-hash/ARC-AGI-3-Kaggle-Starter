"""ARC3 Sovereign NumPy v0.45 competition adapter.

The hot path contains no public-game lookup, fixed color semantics, or fixed
action mapping. Perception, object identity, action effects, goals, and rules are
learned from the authoritative frame stream in real time.
"""
from __future__ import annotations

import os
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

try:
    from .sovereign_v45_bundle import ensure_runtime
except ImportError:
    from sovereign_v45_bundle import ensure_runtime

ensure_runtime()

try:
    from .sovereign_v45_core import CognitiveCore
    from .sovereign_v45_gif import RealtimeGifRecorder
    from .sovereign_v45_perception import available_actions, is_complex_action, to_grid
except ImportError:
    from sovereign_v45_core import CognitiveCore
    from sovereign_v45_gif import RealtimeGifRecorder
    from sovereign_v45_perception import available_actions, is_complex_action, to_grid


class MyAgent(Agent):
    """Data-derived, deterministic, real-time competition agent."""

    MAX_ACTIONS = int(os.environ.get("ARC3_MAX_ACTIONS", "400"))
    MAX_RECOVERIES = int(os.environ.get("ARC3_MAX_RECOVERIES", "4"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        memory_budget = int(os.environ.get("ARC3_MEMORY_BUDGET", str(1 << 17)))
        self.core = CognitiveCore(memory_budget=memory_budget)
        self.recoveries = 0
        trace_path = os.environ.get("ARC3_TRACE_GIF", "").strip()
        self.gif = RealtimeGifRecorder(trace_path) if trace_path else None

    @property
    def name(self) -> str:
        return f"{super().name}.sovereign_numpy_v45"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _fallback_actions(self) -> Any:
        environment = getattr(self, "arc_env", None)
        return getattr(environment, "action_space", None)

    @staticmethod
    def _materialize_action(action: Any, coordinate: tuple[int, int] | None, reason: str, confidence: float) -> GameAction:
        if is_complex_action(action):
            x, y = coordinate or (0, 0)
            setter = getattr(action, "set_data", None)
            if not callable(setter):
                raise TypeError("complex action does not expose set_data")
            setter({"x": int(x), "y": int(y)})
        action.reasoning = {
            "text": reason[:240],
            "confidence": round(float(confidence), 6),
            "source": "online_observation_only",
        }
        return action

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        fallback = self._fallback_actions()

        if latest_frame.state is GameState.NOT_PLAYED:
            self.core.begin_level(retain_rules=True)
            return GameAction.RESET

        actions = available_actions(latest_frame, fallback)
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)

        if latest_frame.state is GameState.GAME_OVER:
            try:
                scene = self.core.observe_sequence(latest_frame, actions, levels, terminal_loss=True)
                if self.gif is not None:
                    self.gif.append_scene(scene, self.core.diagnostics())
            finally:
                self.core.begin_level(retain_rules=True)
            return GameAction.RESET

        try:
            scene = self.core.observe_sequence(latest_frame, actions, levels)
            if self.gif is not None:
                self.gif.append_scene(scene, self.core.diagnostics())
            decision = self.core.decide(scene, actions, levels)
            self.recoveries = 0
            return self._materialize_action(
                decision.action,
                decision.coordinate,
                decision.reason,
                decision.confidence,
            )
        except Exception as exc:
            self.recoveries += 1
            if self.recoveries > self.MAX_RECOVERIES:
                raise RuntimeError("bounded ARC3 recovery budget exhausted") from exc
            grid = to_grid(latest_frame)
            scene = self.core.last_scene or self.core.perception.observe(grid)
            decision = self.core.deterministic_recovery(scene, actions)
            return self._materialize_action(
                decision.action,
                decision.coordinate,
                f"{decision.reason}:{type(exc).__name__}",
                decision.confidence,
            )
