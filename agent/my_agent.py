"""ARC-AGI-3 competition agent backed by ARC3 Sovereign Pure NumPy v0.41.

The learned stack is proposal-only. Every non-reset action is legality-masked,
transactionally prepared, and assimilated from the official next observation.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_AGENT_DIR = Path(__file__).resolve().parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from sovereign_payload import ensure_runtime

ensure_runtime()

from arc3compiler.agent import AgentConfig, Decision, VerifiedAdaptiveAgent
from arc3compiler.kaggle_plugin import FrameNormalizer

logger = logging.getLogger(__name__)


class MyAgent(Agent):
    """Deterministic Pure NumPy + symbolic + bounded causal adaptation agent."""

    MAX_ACTIONS = 200
    MAX_RECOVERIES = 4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._normalizer = FrameNormalizer()
        self._brain = self._new_brain()
        self._initialized = False
        self._awaiting_reset = False
        self._pending: tuple[Any, Decision, int] | None = None
        self._last_levels_completed = 0
        self._recoveries = 0

    @staticmethod
    def _new_brain() -> VerifiedAdaptiveAgent:
        return VerifiedAdaptiveAgent(config=AgentConfig(
            enable_search=True,
            enable_click_memory=True,
            enable_symbolic_clicks=True,
            enable_program_synthesis=True,
            enable_numpy_brain=True,
            enable_novelty_router=True,
            certified_target=0.965,
            novelty_budget=0.035,
            enable_aegis_rt=True,
        ))

    @property
    def name(self) -> str:
        return f"{super().name}.sovereign_numpy_v41"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _available_actions(self, frame: FrameData) -> Any:
        actions = getattr(frame, "available_actions", None)
        return actions if actions else getattr(self.arc_env, "action_space", None)

    def _normalize(self, frame: FrameData):
        return self._normalizer.normalize(frame, self._available_actions(frame))

    def _recover(self, frame: FrameData, reason: BaseException) -> None:
        self._recoveries += 1
        logger.warning(
            "ARC3 cognitive recovery %d/%d: %s",
            self._recoveries,
            self.MAX_RECOVERIES,
            reason,
        )
        if self._recoveries > self.MAX_RECOVERIES:
            raise RuntimeError("ARC3 exceeded bounded recovery budget") from reason
        self._brain = self._new_brain()
        self._pending = None
        self._initialized = False
        self._awaiting_reset = False
        if frame.state is GameState.RUNNING:
            state = self._normalize(frame)
            self._brain.observe_initial(state)
            self._initialized = True
            self._last_levels_completed = int(getattr(frame, "levels_completed", 0) or 0)

    def _assimilate_pending(self, latest_frame: FrameData) -> bool:
        """Assimilate the prior dispatched action. Return True when one existed."""
        if self._pending is None:
            return False
        receipt, decision, before_levels = self._pending
        outcome = self._normalize(latest_frame)
        after_levels = int(getattr(latest_frame, "levels_completed", before_levels) or 0)
        level_transition = after_levels > before_levels
        self._brain.assimilate(
            receipt,
            outcome,
            decision,
            level_transition=level_transition,
            retain_mechanics=True,
        )
        self._pending = None
        self._last_levels_completed = after_levels
        return True

    @staticmethod
    def _to_game_action(decision: Decision) -> GameAction:
        action = GameAction.from_id(decision.action.code)
        if decision.action.code == 6:
            action.set_data({"x": decision.action.x, "y": decision.action.y})
        action.reasoning = {
            "text": decision.reason[:240],
            "confidence": round(float(decision.confidence), 6),
            "novelty": round(float(decision.novelty_score), 6),
            "regime": decision.regime.value,
        }
        return action

    def _dispatch_next(self, levels_completed: int) -> GameAction:
        decision = self._brain.decide()
        receipt = self._brain.prepare_decision(decision)
        self._brain.mark_dispatched(receipt)
        self._pending = (receipt, decision, levels_completed)
        return self._to_game_action(decision)

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # RESET belongs only to the official lifecycle. It never enters the
        # cognitive action mask or the transaction kernel.
        if latest_frame.state is GameState.NOT_PLAYED:
            self._awaiting_reset = True
            return GameAction.RESET

        try:
            assimilated = self._assimilate_pending(latest_frame)

            if latest_frame.state is GameState.GAME_OVER:
                self._awaiting_reset = True
                return GameAction.RESET
            if latest_frame.state is not GameState.RUNNING:
                self._awaiting_reset = True
                return GameAction.RESET

            levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
            if not assimilated:
                state = self._normalize(latest_frame)
                if not self._initialized:
                    self._brain.observe_initial(state)
                    self._initialized = True
                elif self._awaiting_reset:
                    self._brain.begin_level(state, retain_mechanics=True)
                else:
                    # A RUNNING frame with no pending transaction is an
                    # out-of-band observation. Re-anchor deterministically.
                    self._brain.begin_level(state, retain_mechanics=True)
            self._awaiting_reset = False
            self._last_levels_completed = levels
            return self._dispatch_next(levels)
        except Exception as exc:
            self._recover(latest_frame, exc)
            if latest_frame.state is not GameState.RUNNING:
                self._awaiting_reset = True
                return GameAction.RESET
            levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
            return self._dispatch_next(levels)
