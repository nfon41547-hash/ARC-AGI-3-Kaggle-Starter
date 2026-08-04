"""Bounded contextual causal memory for v0.43."""
from __future__ import annotations
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any
try:
    from .sovereign_v43_types import _ZERO_DELTA
except ImportError:
    from sovereign_v43_types import _ZERO_DELTA

@dataclass(frozen=True, slots=True)
class _Outcome:
    delta: tuple[int, int] | None
    utility: float
    blocked: bool
    changed: bool
    progressed: bool


@dataclass(slots=True)
class _ActionStats:
    outcomes: deque[_Outcome] = field(default_factory=lambda: deque(maxlen=48))

    def append(self, outcome: _Outcome) -> None:
        self.outcomes.append(outcome)

    def mapping(self) -> tuple[tuple[int, int] | None, float, float]:
        """Return (delta, confidence, effective sample count)."""
        weighted: defaultdict[tuple[int, int], float] = defaultdict(float)
        effective = 0.0
        for age, outcome in enumerate(reversed(self.outcomes)):
            if outcome.blocked or outcome.delta is None:
                continue
            weight = 0.90**age
            weighted[outcome.delta] += weight
            effective += weight
        if not weighted or effective <= 0.0:
            return None, 0.0, 0.0
        delta, best = max(weighted.items(), key=lambda item: (item[1], item[0]))
        confidence = best / effective
        confidence *= min(1.0, effective / 2.5)
        return delta, float(confidence), float(effective)

    def mean_utility(self) -> float:
        if not self.outcomes:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for age, outcome in enumerate(reversed(self.outcomes)):
            weight = 0.92**age
            weighted_sum += weight * outcome.utility
            weight_total += weight
        return weighted_sum / max(weight_total, 1e-12)

    def no_op_rate(self) -> float:
        usable = [outcome for outcome in self.outcomes if not outcome.blocked]
        if not usable:
            return 0.0
        no_ops = sum(outcome.delta == _ZERO_DELTA and not outcome.changed for outcome in usable)
        return no_ops / len(usable)

    def information_value(self) -> float:
        _, confidence, samples = self.mapping()
        return (1.0 - confidence) + 1.0 / (1.0 + samples)


class _SemanticMemory:
    """Bounded contextual posterior with recency and change-point repair."""

    def __init__(self, max_modes: int = 48) -> None:
        self.max_modes = max_modes
        self.contexts: OrderedDict[bytes, dict[str, _ActionStats]] = OrderedDict()
        self.epochs: Counter[bytes] = Counter()

    def _context(self, mode: bytes) -> dict[str, _ActionStats]:
        context = self.contexts.get(mode)
        if context is None:
            if len(self.contexts) >= self.max_modes:
                self.contexts.popitem(last=False)
            context = defaultdict(_ActionStats)
            self.contexts[mode] = context
        else:
            self.contexts.move_to_end(mode)
        return context

    def update(
        self,
        mode: bytes,
        action: str,
        outcome: _Outcome,
        predicted: tuple[int, int] | None,
        predicted_confidence: float,
    ) -> None:
        context = self._context(mode)
        stats = context[action]
        if (
            not outcome.blocked
            and predicted is not None
            and outcome.delta is not None
            and outcome.delta != _ZERO_DELTA
            and predicted != outcome.delta
            and predicted_confidence >= 0.72
        ):
            stats.outcomes.clear()
            self.epochs[mode] += 1
        stats.append(outcome)

    def prediction(self, mode: bytes, action: str) -> tuple[tuple[int, int] | None, float, float]:
        context = self.contexts.get(mode)
        if context is not None and action in context:
            self.contexts.move_to_end(mode)
            delta, confidence, samples = context[action].mapping()
            if delta is not None:
                return delta, confidence, samples

        votes: list[tuple[int, int]] = []
        strengths: list[float] = []
        for other in self.contexts.values():
            stats = other.get(action)
            if stats is None:
                continue
            delta, confidence, samples = stats.mapping()
            if delta is not None and confidence >= 0.78 and samples >= 2.5:
                votes.append(delta)
                strengths.append(confidence)
        if votes and all(delta == votes[0] for delta in votes):
            transfer_confidence = min(0.88, 0.54 + 0.08 * len(votes))
            return votes[0], transfer_confidence, float(len(votes))
        return None, 0.0, 0.0

    def utility(self, mode: bytes, action: str) -> float:
        context = self.contexts.get(mode)
        if context is None or action not in context:
            return 0.0
        return context[action].mean_utility()

    def no_op_rate(self, mode: bytes, action: str) -> float:
        context = self.contexts.get(mode)
        if context is None or action not in context:
            return 0.0
        return context[action].no_op_rate()

    def information_value(self, mode: bytes, action: str) -> float:
        context = self.contexts.get(mode)
        if context is None or action not in context:
            return 2.0
        return context[action].information_value()


class _BoundedCounter:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: OrderedDict[Any, int] = OrderedDict()

    def increment(self, key: Any) -> int:
        value = self.data.pop(key, 0) + 1
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return value

    def get(self, key: Any) -> int:
        return self.data.get(key, 0)

    def clear(self) -> None:
        self.data.clear()
