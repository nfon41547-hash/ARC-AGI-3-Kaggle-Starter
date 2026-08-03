"""Pure NumPy ARC-AGI-3 competition agent.

The implementation is intentionally self-contained because the starter
notebook copies this single file into the official ARC-AGI-3-Agents runtime.
It learns action semantics from observed transitions, builds a conservative
world model, plans shortest legal routes when evidence is sufficient, and
falls back to deterministic information-gain probes when it is not.

No network access, model server, random policy, score query, dynamic code
execution, or mutable external state is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from hashlib import blake2s
from math import sqrt
from typing import Any, Iterable, Sequence

import numpy as np

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


POLICY_VERSION = "sovereign-numpy-v42"
ACTION_MIN = 1
ACTION_MAX = 7
COMPLEX_ACTION = 6
MAX_GRID = 64
CARDINALS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
)


@dataclass(frozen=True)
class Component:
    color: int
    cells: tuple[tuple[int, int], ...]
    anchor: tuple[int, int]
    shape: tuple[tuple[int, int], ...]
    size: int
    centroid: tuple[float, float]


@dataclass
class ActionEvidence:
    trials: int = 0
    changes: int = 0
    noops: int = 0
    effects: Counter[tuple[int, int]] = field(default_factory=Counter)

    def dominant(self) -> tuple[tuple[int, int] | None, float, int]:
        if not self.effects:
            return None, 0.0, 0
        effect, count = sorted(
            self.effects.items(), key=lambda item: (-item[1], item[0])
        )[0]
        total = max(1, sum(self.effects.values()))
        return effect, float(count / total), int(count)


@dataclass(frozen=True)
class PendingTransition:
    grid: np.ndarray
    action_code: int
    state_digest: bytes
    levels_completed: int
    actor_anchor: tuple[int, int] | None
    click_xy: tuple[int, int] | None
    click_signature: bytes | None


@dataclass(frozen=True)
class CoreDecision:
    action_code: int
    x: int | None
    y: int | None
    reason: str
    confidence: float


class SovereignNumpyCore:
    """Bounded online world model and proposal policy."""

    def __init__(self, game_id: str, max_memory: int = 4096) -> None:
        self.game_id = str(game_id)
        self.max_memory = int(max(256, max_memory))
        self.action_evidence: dict[int, ActionEvidence] = {
            code: ActionEvidence() for code in range(ACTION_MIN, ACTION_MAX + 1)
        }
        self.actor_color_scores: Counter[int] = Counter()
        self.terrain_color_scores: Counter[int] = Counter()
        self.actor_color: int | None = None
        self.actor_anchor: tuple[int, int] | None = None
        self.pending: PendingTransition | None = None
        self.last_grid: np.ndarray | None = None
        self.last_levels = 0
        self.generation = 0
        self.epoch = 0
        self.surprise_streak = 0
        self.recent_actions: deque[int] = deque(maxlen=12)
        self.visited_actor_positions: Counter[tuple[int, int]] = Counter()
        self.blocked_state_actions: set[tuple[bytes, int]] = set()
        self.state_visits: Counter[bytes] = Counter()
        self.click_noop_signatures: Counter[bytes] = Counter()
        self.click_success_signatures: Counter[bytes] = Counter()
        self.click_noop_xy: Counter[tuple[int, int]] = Counter()
        self.click_success_xy: Counter[tuple[int, int]] = Counter()

    @staticmethod
    def final_grid(frame_data: Any) -> np.ndarray:
        raw = getattr(frame_data, "frame", frame_data)
        array = np.asarray(raw)
        if array.ndim == 3:
            if array.shape[0] == 0:
                raise ValueError("empty animation frame sequence")
            array = array[-1]
        elif array.ndim != 2:
            if isinstance(raw, Sequence) and raw:
                candidate = np.asarray(raw[-1])
                if candidate.ndim == 2:
                    array = candidate
            if array.ndim != 2:
                raise ValueError(f"expected a 2-D grid or frame sequence, got {array.shape}")
        if array.size == 0:
            raise ValueError("empty grid")
        height, width = map(int, array.shape)
        if height > MAX_GRID or width > MAX_GRID:
            raise ValueError(f"grid exceeds {MAX_GRID}x{MAX_GRID}: {array.shape}")
        if not np.issubdtype(array.dtype, np.integer):
            if not np.isfinite(array).all():
                raise ValueError("grid contains non-finite values")
            array = array.astype(np.int16)
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < 0 or maximum > 15:
            raise ValueError(f"grid cell outside [0,15]: min={minimum} max={maximum}")
        return np.ascontiguousarray(array, dtype=np.uint8)

    @staticmethod
    def _code(value: Any) -> int | None:
        if isinstance(value, (int, np.integer)):
            return int(value)
        raw = getattr(value, "value", None)
        if isinstance(raw, (int, np.integer)):
            return int(raw)
        if isinstance(value, str):
            upper = value.upper()
            if upper == "RESET":
                return 0
            if upper.startswith("ACTION"):
                try:
                    return int(upper.removeprefix("ACTION"))
                except ValueError:
                    return None
        if isinstance(value, dict):
            for key in ("id", "action", "action_id", "code", "name"):
                if key in value:
                    parsed = SovereignNumpyCore._code(value[key])
                    if parsed is not None:
                        return parsed
        return None

    @classmethod
    def available_codes(cls, frame_data: Any) -> tuple[int, ...]:
        raw = getattr(frame_data, "available_actions", None)
        if raw is None:
            action_space = getattr(frame_data, "action_space", None)
            raw = action_space if action_space is not None else range(1, 8)
        codes: set[int] = set()
        if isinstance(raw, (int, np.integer)):
            bitmask = int(raw)
            if 1 <= bitmask <= 7:
                codes.add(bitmask)
            else:
                for code in range(1, 8):
                    if bitmask & (1 << code):
                        codes.add(code)
        elif isinstance(raw, dict):
            for key, value in raw.items():
                parsed = cls._code(key)
                if parsed is None:
                    parsed = cls._code(value)
                if parsed is not None:
                    codes.add(parsed)
        else:
            try:
                for item in raw:
                    parsed = cls._code(item)
                    if parsed is not None:
                        codes.add(parsed)
            except TypeError:
                parsed = cls._code(raw)
                if parsed is not None:
                    codes.add(parsed)
        legal = tuple(sorted(code for code in codes if ACTION_MIN <= code <= ACTION_MAX))
        if not legal:
            raise RuntimeError("observation exposes no legal non-reset actions")
        return legal

    @staticmethod
    def levels_completed(frame_data: Any) -> int:
        try:
            return max(0, int(getattr(frame_data, "levels_completed", 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _background(grid: np.ndarray) -> int:
        counts = np.bincount(grid.reshape(-1), minlength=16)
        if int(counts[0]) >= max(1, int(grid.size * 0.05)):
            return 0
        return int(np.flatnonzero(counts == counts.max())[0])

    def _terrain_color(self, grid: np.ndarray) -> int:
        if self.terrain_color_scores:
            return sorted(
                self.terrain_color_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
        return self._background(grid)

    @staticmethod
    def _digest(grid: np.ndarray, available: Iterable[int]) -> bytes:
        digest = blake2s(digest_size=16, person=b"arc3-v42")
        digest.update(bytes(grid.shape))
        digest.update(grid.tobytes(order="C"))
        digest.update(bytes(sorted(int(code) for code in available)))
        return digest.digest()

    @staticmethod
    def components(grid: np.ndarray) -> tuple[Component, ...]:
        height, width = map(int, grid.shape)
        seen = np.zeros((height, width), dtype=np.uint8)
        output: list[Component] = []
        for y0 in range(height):
            for x0 in range(width):
                if seen[y0, x0]:
                    continue
                color = int(grid[y0, x0])
                queue = [(x0, y0)]
                seen[y0, x0] = 1
                cells: list[tuple[int, int]] = []
                cursor = 0
                while cursor < len(queue):
                    x, y = queue[cursor]
                    cursor += 1
                    cells.append((x, y))
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and not seen[ny, nx]
                            and int(grid[ny, nx]) == color
                        ):
                            seen[ny, nx] = 1
                            queue.append((nx, ny))
                min_x = min(x for x, _ in cells)
                min_y = min(y for _, y in cells)
                shape = tuple(sorted((x - min_x, y - min_y) for x, y in cells))
                output.append(
                    Component(
                        color=color,
                        cells=tuple(sorted(cells)),
                        anchor=(min_x, min_y),
                        shape=shape,
                        size=len(cells),
                        centroid=(
                            float(sum(x for x, _ in cells) / len(cells)),
                            float(sum(y for _, y in cells) / len(cells)),
                        ),
                    )
                )
        return tuple(output)

    @classmethod
    def _detect_translation(
        cls,
        previous: np.ndarray,
        current: np.ndarray,
        preferred_color: int | None,
    ) -> tuple[int, int, int, tuple[int, int], tuple[int, int]] | None:
        if previous.shape != current.shape or np.array_equal(previous, current):
            return None
        background = cls._background(previous)
        prev = [c for c in cls.components(previous) if c.color != background]
        curr = [c for c in cls.components(current) if c.color != background]
        curr_by_key: dict[tuple[int, tuple[tuple[int, int], ...]], list[Component]] = defaultdict(list)
        for component in curr:
            curr_by_key[(component.color, component.shape)].append(component)
        candidates: list[tuple[int, int, int, int, tuple[int, int], tuple[int, int]]] = []
        for before in prev:
            for after in curr_by_key.get((before.color, before.shape), ()):
                dx = int(after.anchor[0] - before.anchor[0])
                dy = int(after.anchor[1] - before.anchor[1])
                if dx == 0 and dy == 0:
                    continue
                if abs(dx) + abs(dy) > 8:
                    continue
                candidates.append(
                    (
                        0 if before.color == preferred_color else 1,
                        before.size,
                        abs(dx) + abs(dy),
                        before.color,
                        before.anchor,
                        after.anchor,
                    )
                )
        if not candidates:
            return None
        candidates.sort()
        _, _, _, color, before_anchor, after_anchor = candidates[0]
        return (
            int(after_anchor[0] - before_anchor[0]),
            int(after_anchor[1] - before_anchor[1]),
            int(color),
            before_anchor,
            after_anchor,
        )

    def _choose_actor(self, grid: np.ndarray, components: Sequence[Component]) -> Component | None:
        background = self._terrain_color(grid)
        non_background = [c for c in components if c.color != background]
        if not non_background:
            return None
        if self.actor_color is not None:
            same = [c for c in non_background if c.color == self.actor_color]
            if same:
                if self.actor_anchor is None:
                    return min(same, key=lambda c: (c.size, c.anchor))
                ax, ay = self.actor_anchor
                return min(
                    same,
                    key=lambda c: (
                        abs(c.anchor[0] - ax) + abs(c.anchor[1] - ay),
                        c.size,
                        c.anchor,
                    ),
                )
        color_counts = np.bincount(grid.reshape(-1), minlength=16)
        max_actor = max(1, int(grid.size * 0.12))
        candidates = [c for c in non_background if c.size <= max_actor] or non_background
        return min(
            candidates,
            key=lambda c: (int(color_counts[c.color]), c.size, c.anchor),
        )

    def _goal_candidates(
        self,
        grid: np.ndarray,
        components: Sequence[Component],
        actor: Component | None,
    ) -> list[Component]:
        background = self._terrain_color(grid)
        color_counts = np.bincount(grid.reshape(-1), minlength=16)
        output: list[Component] = []
        for component in components:
            if component.color == background:
                continue
            if actor is not None and component == actor:
                continue
            if component.size > max(4, int(grid.size * 0.20)):
                continue
            if int(color_counts[component.color]) > max(8, int(grid.size * 0.10)):
                continue
            output.append(component)
        origin = actor.centroid if actor is not None else (grid.shape[1] / 2.0, grid.shape[0] / 2.0)
        output.sort(
            key=lambda c: (
                int(color_counts[c.color]),
                c.size,
                abs(c.centroid[0] - origin[0]) + abs(c.centroid[1] - origin[1]),
                c.anchor,
            )
        )
        return output

    def _reset_level_local(self, keep_action_semantics: bool = True) -> None:
        self.actor_anchor = None
        self.pending = None
        self.last_grid = None
        self.recent_actions.clear()
        self.visited_actor_positions.clear()
        self.blocked_state_actions.clear()
        self.state_visits.clear()
        self.click_noop_xy.clear()
        self.click_success_xy.clear()
        if not keep_action_semantics:
            self.action_evidence = {
                code: ActionEvidence() for code in range(ACTION_MIN, ACTION_MAX + 1)
            }
            self.actor_color_scores.clear()
            self.terrain_color_scores.clear()
            self.actor_color = None
            self.surprise_streak = 0

    def _assimilate(self, grid: np.ndarray, levels: int, available: tuple[int, ...]) -> None:
        pending = self.pending
        if pending is None:
            return
        if levels < pending.levels_completed:
            self._reset_level_local(keep_action_semantics=False)
            return
        level_advanced = levels > pending.levels_completed
        changed = pending.grid.shape == grid.shape and not np.array_equal(pending.grid, grid)
        evidence = self.action_evidence[pending.action_code]
        evidence.trials += 1
        if changed or level_advanced:
            evidence.changes += 1
        else:
            evidence.noops += 1
            self.blocked_state_actions.add((pending.state_digest, pending.action_code))
        if pending.action_code == COMPLEX_ACTION:
            if pending.click_signature is not None:
                target = self.click_success_signatures if changed or level_advanced else self.click_noop_signatures
                target[pending.click_signature] += 1
            if pending.click_xy is not None:
                target_xy = self.click_success_xy if changed or level_advanced else self.click_noop_xy
                target_xy[pending.click_xy] += 1
        elif changed and pending.grid.shape == grid.shape:
            translation = self._detect_translation(pending.grid, grid, preferred_color=self.actor_color)
            if translation is not None:
                dx, dy, color, _before_anchor, after_anchor = translation
                prior_effect, prior_confidence, prior_count = evidence.dominant()
                observed = (dx, dy)
                if (
                    prior_effect is not None
                    and prior_count >= 2
                    and prior_confidence >= 0.67
                    and observed != prior_effect
                ):
                    self.surprise_streak += 1
                else:
                    self.surprise_streak = max(0, self.surprise_streak - 1)
                if self.surprise_streak >= 2:
                    self.epoch += 1
                    self.action_evidence = {
                        code: ActionEvidence() for code in range(ACTION_MIN, ACTION_MAX + 1)
                    }
                    evidence = self.action_evidence[pending.action_code]
                    evidence.trials = 1
                    evidence.changes = 1
                    self.surprise_streak = 0
                evidence.effects[observed] += 1
                self.actor_color_scores[color] += 2
                self.actor_color = sorted(
                    self.actor_color_scores.items(), key=lambda item: (-item[1], item[0])
                )[0][0]
                changed_mask = pending.grid != grid
                vacated = np.logical_and(changed_mask, pending.grid == color)
                entered = np.logical_and(changed_mask, grid == color)
                for terrain in grid[vacated].tolist():
                    if int(terrain) != color:
                        self.terrain_color_scores[int(terrain)] += 1
                for terrain in pending.grid[entered].tolist():
                    if int(terrain) != color:
                        self.terrain_color_scores[int(terrain)] += 1
                self.actor_anchor = after_anchor
                self.visited_actor_positions[after_anchor] += 1
        if level_advanced:
            self._reset_level_local(keep_action_semantics=True)

    def motion_map(self, available: Iterable[int]) -> dict[tuple[int, int], int]:
        result: dict[tuple[int, int], tuple[int, float, int]] = {}
        for code in sorted(int(c) for c in available if int(c) != COMPLEX_ACTION):
            effect, confidence, count = self.action_evidence[code].dominant()
            if effect not in CARDINALS or count < 1 or confidence < 0.60:
                continue
            existing = result.get(effect)
            candidate = (code, confidence, count)
            if existing is None or (confidence, count, -code) > (existing[1], existing[2], -existing[0]):
                result[effect] = candidate
        return {effect: item[0] for effect, item in result.items()}

    @staticmethod
    def _anchor_valid(
        anchor: tuple[int, int],
        offsets: Sequence[tuple[int, int]],
        passable: np.ndarray,
    ) -> bool:
        x0, y0 = anchor
        height, width = passable.shape
        for ox, oy in offsets:
            x = x0 + ox
            y = y0 + oy
            if not (0 <= x < width and 0 <= y < height and bool(passable[y, x])):
                return False
        return True

    def _plan(
        self,
        grid: np.ndarray,
        actor: Component | None,
        goals: Sequence[Component],
        available: tuple[int, ...],
    ) -> tuple[int, str, float] | None:
        if actor is None or not goals:
            return None
        mapping = self.motion_map(available)
        if len(mapping) < 2:
            return None
        background = self._terrain_color(grid)
        offsets = actor.shape
        start = actor.anchor
        height, width = map(int, grid.shape)
        for goal in goals[:8]:
            passable = np.logical_or(grid == background, grid == actor.color)
            for x, y in goal.cells:
                passable[y, x] = True
            goal_cells = set(goal.cells)
            queue: deque[tuple[int, int]] = deque([start])
            parent: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]] | None] = {start: None}
            found: tuple[int, int] | None = None
            while queue and len(parent) <= height * width:
                anchor = queue.popleft()
                occupied = {(anchor[0] + ox, anchor[1] + oy) for ox, oy in offsets}
                if occupied & goal_cells:
                    found = anchor
                    break
                for direction in CARDINALS:
                    code = mapping.get(direction)
                    if code is None:
                        continue
                    nxt = (anchor[0] + direction[0], anchor[1] + direction[1])
                    if nxt in parent:
                        continue
                    if not self._anchor_valid(nxt, offsets, passable):
                        continue
                    parent[nxt] = (anchor, direction)
                    queue.append(nxt)
            if found is None or found == start:
                continue
            cursor = found
            reversed_directions: list[tuple[int, int]] = []
            while parent[cursor] is not None:
                previous, direction = parent[cursor]  # type: ignore[misc]
                reversed_directions.append(direction)
                cursor = previous
            first = reversed_directions[-1]
            code = mapping[first]
            state_key = self._digest(grid, available)
            if (state_key, code) in self.blocked_state_actions:
                continue
            path_length = len(reversed_directions)
            confidence = min(0.98, 0.72 + 0.04 * len(mapping) + 0.02 * min(path_length, 5))
            return code, f"shortest-route length={path_length} goal_color={goal.color}", confidence
        return None

    @staticmethod
    def _component_signature(component: Component) -> bytes:
        digest = blake2s(digest_size=12, person=b"a3click")
        digest.update(bytes((component.color, min(255, component.size))))
        for x, y in component.shape:
            digest.update(bytes((x & 0xFF, y & 0xFF)))
        return digest.digest()

    def _click_decision(
        self,
        grid: np.ndarray,
        components: Sequence[Component],
    ) -> CoreDecision:
        background = self._terrain_color(grid)
        color_counts = np.bincount(grid.reshape(-1), minlength=16)
        candidates: list[tuple[float, int, int, bytes]] = []
        for component in components:
            if component.color == background:
                continue
            signature = self._component_signature(component)
            x = int(round(component.centroid[0]))
            y = int(round(component.centroid[1]))
            success = self.click_success_signatures[signature] + self.click_success_xy[(x, y)]
            noops = self.click_noop_signatures[signature] + self.click_noop_xy[(x, y)]
            rarity = 1.0 / max(1, int(color_counts[component.color]))
            compactness = 1.0 / max(1, component.size)
            unexplored = 1.5 if success + noops == 0 else 0.0
            score = 4.0 * success - 3.0 * noops + 3.0 * rarity + compactness + unexplored
            candidates.append((score, x, y, signature))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[2], item[1], item[3]))
            score, x, y, _ = candidates[0]
            confidence = 0.90 if score >= 4.0 else 0.55
            return CoreDecision(COMPLEX_ACTION, x, y, "ranked compact click target", confidence)
        height, width = map(int, grid.shape)
        anchors = [
            (width // 2, height // 2),
            (width // 4, height // 4),
            (3 * width // 4, height // 4),
            (width // 4, 3 * height // 4),
            (3 * width // 4, 3 * height // 4),
        ]
        x, y = min(anchors, key=lambda xy: (self.click_noop_xy[xy], xy[1], xy[0]))
        return CoreDecision(COMPLEX_ACTION, x, y, "coverage click", 0.20)

    def _probe(
        self,
        grid: np.ndarray,
        available: tuple[int, ...],
        actor: Component | None,
    ) -> CoreDecision:
        digest = self._digest(grid, available)
        mapping = self.motion_map(available)
        reverse_map = {code: effect for effect, code in mapping.items()}
        background = self._terrain_color(grid)
        passable = grid == background
        if actor is not None:
            passable = np.logical_or(passable, grid == actor.color)
        scored: list[tuple[float, int, str]] = []
        for code in available:
            if code == COMPLEX_ACTION:
                continue
            evidence = self.action_evidence[code]
            score = 4.0 / sqrt(1.0 + evidence.trials)
            reason: list[str] = []
            if evidence.trials == 0:
                score += 3.0
                reason.append("untried")
            if evidence.trials:
                score += 2.5 * (evidence.changes / evidence.trials)
                score -= 2.0 * (evidence.noops / evidence.trials)
            if (digest, code) in self.blocked_state_actions:
                score -= 8.0
                reason.append("known-noop")
            score -= 0.65 * sum(1 for recent in self.recent_actions if recent == code)
            effect = reverse_map.get(code)
            if effect is not None and actor is not None:
                nxt = (actor.anchor[0] + effect[0], actor.anchor[1] + effect[1])
                if self._anchor_valid(nxt, actor.shape, passable):
                    score += 3.0 / (1.0 + self.visited_actor_positions[nxt])
                    reason.append("frontier")
                else:
                    score -= 4.0
                    reason.append("collision")
            stable = blake2s(
                f"{self.game_id}|{self.epoch}|{self.generation}|{code}".encode(),
                digest_size=2,
                person=b"arc3-tie",
            ).digest()
            score += int.from_bytes(stable, "big") * 1e-9
            scored.append((score, code, "+".join(reason) or "information-gain"))
        if not scored:
            if COMPLEX_ACTION in available:
                return self._click_decision(grid, self.components(grid))
            raise RuntimeError("no legal action candidate")
        scored.sort(key=lambda item: (-item[0], item[1]))
        _, code, reason = scored[0]
        confidence = 0.35 if self.action_evidence[code].trials else 0.05
        return CoreDecision(code, None, None, f"probe:{reason}", confidence)

    def decide(self, frame_data: Any) -> CoreDecision:
        grid = self.final_grid(frame_data)
        available = self.available_codes(frame_data)
        levels = self.levels_completed(frame_data)
        self._assimilate(grid, levels, available)
        self.generation += 1
        state_digest = self._digest(grid, available)
        self.state_visits[state_digest] += 1
        components = self.components(grid)
        actor = self._choose_actor(grid, components)
        if actor is not None:
            self.actor_anchor = actor.anchor
            self.visited_actor_positions[actor.anchor] += 1
        goals = self._goal_candidates(grid, components, actor)
        if len(available) == 1:
            code = available[0]
            decision = (
                self._click_decision(grid, components)
                if code == COMPLEX_ACTION
                else CoreDecision(code, None, None, "only legal action", 1.0)
            )
        else:
            plan = self._plan(grid, actor, goals, available)
            if plan is not None:
                code, reason, confidence = plan
                decision = CoreDecision(code, None, None, reason, confidence)
            else:
                simple = [code for code in available if code != COMPLEX_ACTION]
                all_simple_sampled = bool(simple) and all(
                    self.action_evidence[code].trials >= 1 for code in simple
                )
                productive_click = any(self.click_success_signatures.values())
                if COMPLEX_ACTION in available and (not simple or all_simple_sampled or productive_click):
                    click = self._click_decision(grid, components)
                    simple_change = sum(self.action_evidence[c].changes for c in simple)
                    simple_trials = sum(self.action_evidence[c].trials for c in simple)
                    decision = (
                        click
                        if productive_click or (simple_trials >= len(simple) and simple_change == 0)
                        else self._probe(grid, available, actor)
                    )
                else:
                    decision = self._probe(grid, available, actor)
        if decision.action_code not in available:
            raise RuntimeError(f"internal illegal action {decision.action_code}; legal={available}")
        if decision.action_code == COMPLEX_ACTION:
            if decision.x is None or decision.y is None:
                raise RuntimeError("ACTION6 requires x/y")
            height, width = map(int, grid.shape)
            if not (0 <= decision.x < width and 0 <= decision.y < height):
                raise RuntimeError("ACTION6 coordinate outside current grid")
        click_signature: bytes | None = None
        if decision.action_code == COMPLEX_ACTION and decision.x is not None and decision.y is not None:
            for component in components:
                if (decision.x, decision.y) in component.cells:
                    click_signature = self._component_signature(component)
                    break
        self.pending = PendingTransition(
            grid=np.array(grid, copy=True),
            action_code=decision.action_code,
            state_digest=state_digest,
            levels_completed=levels,
            actor_anchor=actor.anchor if actor is not None else None,
            click_xy=(decision.x, decision.y) if decision.action_code == COMPLEX_ACTION else None,
            click_signature=click_signature,
        )
        self.last_grid = np.array(grid, copy=True)
        self.last_levels = levels
        self.recent_actions.append(decision.action_code)
        return decision


class MyAgent(Agent):
    """Official ARC-AGI-3-Agents adapter for the Pure NumPy core."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.core = SovereignNumpyCore(game_id=self.game_id)

    @property
    def name(self) -> str:
        return f"{super().name}.{POLICY_VERSION}.{self.MAX_ACTIONS}"

    @staticmethod
    def _state_name(state: Any) -> str:
        return str(getattr(state, "name", state)).upper()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return self._state_name(latest_frame.state).endswith("WIN")

    @staticmethod
    def _action_from_code(code: int) -> GameAction:
        factory = getattr(GameAction, "from_id", None)
        if callable(factory):
            return factory(int(code))
        try:
            return GameAction(int(code))
        except Exception:
            for action in GameAction:
                if int(getattr(action, "value", -1)) == int(code):
                    return action
        raise RuntimeError(f"cannot construct GameAction from code={code}")

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        state_name = self._state_name(latest_frame.state)
        if state_name.endswith("NOT_PLAYED") or state_name.endswith("GAME_OVER"):
            action = GameAction.RESET
            action.reasoning = {
                "policy": POLICY_VERSION,
                "reason": "official lifecycle reset",
            }
            return action
        if state_name.endswith("WIN"):
            raise RuntimeError("choose_action called after WIN")
        decision = self.core.decide(latest_frame)
        action = self._action_from_code(decision.action_code)
        if decision.action_code == COMPLEX_ACTION:
            action.set_data({"x": int(decision.x), "y": int(decision.y)})
        action.reasoning = {
            "policy": POLICY_VERSION,
            "reason": decision.reason,
            "confidence": round(float(decision.confidence), 6),
            "epoch": self.core.epoch,
            "generation": self.core.generation,
        }
        return action
