"""Verified online cognition, planning, and repair for v0.43."""
from __future__ import annotations
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
try:
    from .sovereign_v43_types import (_ACTION_NAMES, _MOVE_NAMES, _CARDINALS, _ZERO_DELTA, _Component, _anchor_and_offsets, _background, _components, _grid_digest, _mode_digest)
    from .sovereign_v43_memory import _BoundedCounter, _Outcome, _SemanticMemory
except ImportError:
    from sovereign_v43_types import (_ACTION_NAMES, _MOVE_NAMES, _CARDINALS, _ZERO_DELTA, _Component, _anchor_and_offsets, _background, _components, _grid_digest, _mode_digest)
    from sovereign_v43_memory import _BoundedCounter, _Outcome, _SemanticMemory

@dataclass(slots=True)
class _Pending:
    action: str
    before: np.ndarray
    actor_anchor: tuple[int, int] | None
    actor_cells: tuple[tuple[int, int], ...]
    levels: int
    mode: bytes
    predicted_delta: tuple[int, int] | None
    predicted_confidence: float
    predicted_clear: bool | None
    click_coord: tuple[int, int] | None


class _CognitiveCore:
    """Verified online learner, planner, and bounded self-repair loop."""

    def __init__(self) -> None:
        self.memory = _SemanticMemory()
        self.pending: _Pending | None = None
        self.actor_color: int | None = None
        self.goal_color: int | None = None
        self.actor_component: _Component | None = None
        self.goal_component: _Component | None = None
        self.last_levels = 0
        self.stagnation = 0
        self.state_visits = _BoundedCounter(2048)
        self.state_actions = _BoundedCounter(4096)
        self.click_attempts: OrderedDict[tuple[int, int], float] = OrderedDict()
        self.last_changed_cells: tuple[tuple[int, int], ...] = ()

    def begin_level(self, *, retain_semantics: bool = True) -> None:
        if not retain_semantics:
            self.memory = _SemanticMemory()
            self.actor_color = None
            self.goal_color = None
        self.pending = None
        self.actor_component = None
        self.goal_component = None
        self.stagnation = 0
        self.state_visits.clear()
        self.state_actions.clear()
        self.click_attempts.clear()
        self.last_changed_cells = ()

    @staticmethod
    def _nearest_component(
        components: Sequence[_Component],
        color: int,
        previous: _Component | None,
    ) -> _Component | None:
        candidates = [component for component in components if component.color == color]
        if not candidates:
            return None
        if previous is None:
            return min(candidates, key=lambda component: (component.area, component.centroid))
        py, px = previous.centroid
        return min(
            candidates,
            key=lambda component: (
                abs(component.centroid[0] - py) + abs(component.centroid[1] - px),
                abs(component.area - previous.area),
                component.centroid,
            ),
        )

    def _select_roles(self, grid: np.ndarray) -> tuple[_Component | None, _Component | None]:
        components = _components(grid)
        if not components:
            return None, None
        by_color: defaultdict[int, list[_Component]] = defaultdict(list)
        for component in components:
            by_color[component.color].append(component)
        total_area = {
            color: sum(component.area for component in group)
            for color, group in by_color.items()
        }

        actor: _Component | None = None
        if self.actor_color is not None:
            actor = self._nearest_component(components, self.actor_color, self.actor_component)
        if actor is None:
            color_one = [component for component in components if component.color == 1]
            if color_one:
                actor = min(color_one, key=lambda component: (component.area, component.centroid))
            else:
                actor = min(
                    components,
                    key=lambda component: (
                        len(by_color[component.color]) > 1,
                        total_area[component.color] > 16,
                        component.touches_border,
                        component.area > 16,
                        component.area,
                        component.color,
                        component.centroid,
                    ),
                )
            self.actor_color = actor.color

        goal: _Component | None = None
        if self.goal_color == self.actor_color:
            self.goal_color = None
        if self.goal_color is not None:
            goal = self._nearest_component(components, self.goal_color, self.goal_component)
        if goal is None:
            color_three = [component for component in components if component.color == 3 and component != actor]
            if color_three:
                goal = min(color_three, key=lambda component: (component.area, component.centroid))
            else:
                candidates = [component for component in components if component != actor]
                if candidates:
                    actor_area = actor.area if actor is not None else 1
                    goal = min(
                        candidates,
                        key=lambda component: (
                            len(by_color[component.color]) > 1,
                            total_area[component.color] > max(16, 4 * actor_area),
                            component.touches_border,
                            abs(component.area - actor_area),
                            total_area[component.color],
                            component.area,
                            component.color,
                        ),
                    )
            if goal is not None:
                self.goal_color = goal.color
        return actor, goal

    def _actor_after(self, grid: np.ndarray) -> _Component | None:
        components = _components(grid)
        if self.actor_color is None:
            return None
        return self._nearest_component(components, self.actor_color, self.actor_component)

    @staticmethod
    def _translated_component(
        before: np.ndarray, after: np.ndarray
    ) -> tuple[_Component, _Component, tuple[int, int]] | None:
        if before.shape != after.shape:
            return None
        before_components = _components(before)
        after_components = _components(after)
        candidates: list[tuple[int, int, int, _Component, _Component, tuple[int, int]]] = []
        for source in before_components:
            sy, sx, _, _ = source.bbox
            source_cells = set(source.cells)
            for target in after_components:
                if target.color != source.color or target.area != source.area:
                    continue
                ty, tx, _, _ = target.bbox
                delta = (ty - sy, tx - sx)
                if delta == _ZERO_DELTA or abs(delta[0]) + abs(delta[1]) > 8:
                    continue
                translated = {(y + delta[0], x + delta[1]) for y, x in source_cells}
                if translated != set(target.cells):
                    continue
                changed_support = sum(
                    before[y, x] != after[y, x]
                    for y, x in source.cells
                    if 0 <= y < after.shape[0] and 0 <= x < after.shape[1]
                )
                candidates.append((changed_support, -source.area, -source.color, source, target, delta))
        if not candidates:
            return None
        _, _, _, source, target, delta = max(candidates, key=lambda item: item[:3])
        return source, target, delta

    @staticmethod
    def _changed_cells(before: np.ndarray, after: np.ndarray) -> tuple[tuple[int, int], ...]:
        if before.shape != after.shape:
            return ()
        points = np.argwhere(before != after)
        return tuple((int(y), int(x)) for y, x in points[:256])

    def observe(self, grid: np.ndarray, levels: int, *, terminal_loss: bool = False) -> None:
        progressed = levels > self.last_levels
        if self.pending is not None:
            actor_after = self._actor_after(grid)
            moved = self._translated_component(self.pending.before, grid)
            delta: tuple[int, int] | None = None
            if moved is not None:
                _, moved_after, moved_delta = moved
                delta = moved_delta
                if self.actor_color != moved_after.color:
                    self.actor_color = moved_after.color
                    self.actor_component = moved_after
                    if self.goal_color == self.actor_color:
                        self.goal_color = None
                actor_after = moved_after
            elif self.pending.actor_anchor is not None and actor_after is not None:
                anchor_after, _ = _anchor_and_offsets(actor_after)
                delta = (
                    anchor_after[0] - self.pending.actor_anchor[0],
                    anchor_after[1] - self.pending.actor_anchor[1],
                )
                if abs(delta[0]) + abs(delta[1]) > 8:
                    delta = None
            changed_cells = self._changed_cells(self.pending.before, grid)
            changed = bool(changed_cells) or self.pending.before.shape != grid.shape
            blocked = bool(
                delta == _ZERO_DELTA
                and self.pending.predicted_delta not in (None, _ZERO_DELTA)
                and self.pending.predicted_clear is False
            )
            utility = 6.0 if progressed else (1.2 if changed else -1.0)
            if terminal_loss:
                utility -= 5.0
            if delta == _ZERO_DELTA and not blocked:
                utility -= 0.8
            outcome = _Outcome(
                delta=delta,
                utility=utility,
                blocked=blocked,
                changed=changed,
                progressed=progressed,
            )
            self.memory.update(
                self.pending.mode,
                self.pending.action,
                outcome,
                self.pending.predicted_delta,
                self.pending.predicted_confidence,
            )
            if self.pending.click_coord is not None:
                self.click_attempts[self.pending.click_coord] = utility
                self.click_attempts.move_to_end(self.pending.click_coord)
                while len(self.click_attempts) > 512:
                    self.click_attempts.popitem(last=False)
            self.last_changed_cells = changed_cells
            self.stagnation = 0 if changed or progressed else self.stagnation + 1
            self.pending = None

        if progressed:
            self.actor_component = None
            self.goal_component = None
            self.state_visits.clear()
            self.state_actions.clear()
            self.click_attempts.clear()
            self.stagnation = 0

        actor, goal = self._select_roles(grid)
        self.actor_component = actor
        self.goal_component = goal
        self.last_levels = levels
        self.state_visits.increment(_grid_digest(grid))

    @staticmethod
    def _anchor_valid(
        anchor: tuple[int, int],
        offsets: Sequence[tuple[int, int]],
        allowed: np.ndarray,
    ) -> bool:
        height, width = allowed.shape
        ay, ax = anchor
        for oy, ox in offsets:
            y, x = ay + oy, ax + ox
            if not (0 <= y < height and 0 <= x < width and bool(allowed[y, x])):
                return False
        return True

    @staticmethod
    def _footprint_cells(
        anchor: tuple[int, int], offsets: Sequence[tuple[int, int]]
    ) -> tuple[tuple[int, int], ...]:
        return tuple((anchor[0] + oy, anchor[1] + ox) for oy, ox in offsets)

    @classmethod
    def _plan_first_step(
        cls,
        grid: np.ndarray,
        actor: _Component,
        goal: _Component,
    ) -> tuple[tuple[int, int] | None, int | None, bool | None]:
        start, offsets = _anchor_and_offsets(actor)
        background = _background(grid)
        allowed = grid == background
        for cell in actor.cells:
            allowed[cell] = True
        for cell in goal.cells:
            allowed[cell] = True
        goal_cells = set(goal.cells)

        def reaches_goal(anchor: tuple[int, int]) -> bool:
            return any(cell in goal_cells for cell in cls._footprint_cells(anchor, offsets))

        queue = deque([start])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        distance = {start: 0}
        terminal: tuple[int, int] | None = start if reaches_goal(start) else None
        while queue and terminal is None:
            current = queue.popleft()
            for dy, dx in _CARDINALS:
                nxt = (current[0] + dy, current[1] + dx)
                if nxt in previous or not cls._anchor_valid(nxt, offsets, allowed):
                    continue
                previous[nxt] = current
                distance[nxt] = distance[current] + 1
                if reaches_goal(nxt):
                    terminal = nxt
                    break
                queue.append(nxt)

        if terminal is None or terminal == start:
            return None, (0 if terminal == start else None), None
        cursor = terminal
        while previous[cursor] != start:
            parent = previous[cursor]
            if parent is None:
                return None, None, None
            cursor = parent
        first_step = (cursor[0] - start[0], cursor[1] - start[1])
        first_clear = cls._anchor_valid(cursor, offsets, allowed)
        return first_step, distance[terminal], first_clear

    @staticmethod
    def _predicted_clear(
        grid: np.ndarray,
        actor: _Component | None,
        goal: _Component | None,
        delta: tuple[int, int] | None,
    ) -> bool | None:
        if actor is None or delta is None:
            return None
        anchor, offsets = _anchor_and_offsets(actor)
        background = _background(grid)
        allowed = grid == background
        for cell in actor.cells:
            allowed[cell] = True
        if goal is not None:
            for cell in goal.cells:
                allowed[cell] = True
        target = (anchor[0] + delta[0], anchor[1] + delta[1])
        return _CognitiveCore._anchor_valid(target, offsets, allowed)

    def _click_candidate(
        self, grid: np.ndarray, *, exclude_roles: bool = True
    ) -> tuple[int, int] | None:
        components = _components(grid)
        if not components:
            return None
        color_counts = Counter(component.color for component in components)
        actor_cells = (
            set(self.actor_component.cells) if exclude_roles and self.actor_component else set()
        )
        goal_cells = (
            set(self.goal_component.cells) if exclude_roles and self.goal_component else set()
        )
        changed = set(self.last_changed_cells)
        candidates: list[tuple[float, int, int]] = []
        for component in components:
            if actor_cells.intersection(component.cells) or goal_cells.intersection(component.cells):
                continue
            y, x = component.centroid
            coordinate = (x, y)
            previous = self.click_attempts.get(coordinate)
            if previous is not None and previous <= 0.0:
                repeat_penalty = 8.0
            elif previous is not None:
                repeat_penalty = 2.0
            else:
                repeat_penalty = 0.0
            change_distance = 8.0
            if changed:
                change_distance = min(abs(y - cy) + abs(x - cx) for cy, cx in changed)
            rarity = 1.0 / color_counts[component.color]
            salience = (
                5.0 * rarity
                + 4.0 / math.sqrt(component.area)
                + 2.0 / (1.0 + change_distance)
                - 1.5 * component.touches_border
                - repeat_penalty
            )
            candidates.append((salience, x, y))
        if not candidates:
            return None
        _, x, y = max(candidates, key=lambda item: (item[0], -item[2], -item[1]))
        return x, y

    def _action_score(
        self,
        state_digest: bytes,
        mode: bytes,
        action: str,
        desired: tuple[int, int] | None,
        predicted: tuple[int, int] | None,
        confidence: float,
        samples: float,
    ) -> float:
        score = 0.9 * self.memory.utility(mode, action)
        score -= 1.4 * self.memory.no_op_rate(mode, action)
        score -= 0.45 * self.state_actions.get((state_digest, action))
        if desired is not None:
            if predicted == desired:
                score += 11.0 * confidence
            elif predicted is not None:
                score -= 4.5 * confidence
            elif action in _MOVE_NAMES:
                score += 2.4 * self.memory.information_value(mode, action)
        elif action in _MOVE_NAMES:
            score += 0.5 * self.memory.information_value(mode, action)
        if action == "ACTION5":
            score -= 1.0 if self.stagnation < 2 else -0.8
        elif action == "ACTION7":
            score -= 3.0 if self.stagnation < 4 else 0.5
        if samples <= 0.0:
            score += 0.2
        return score

    def choose(
        self,
        grid: np.ndarray,
        available: tuple[str, ...],
        levels: int,
    ) -> tuple[str, tuple[int, int] | None, str]:
        if not available:
            raise RuntimeError("no legal action available")
        actor = self.actor_component
        goal = self.goal_component
        actor_cells = actor.cells if actor is not None else ()
        mode = _mode_digest(grid, actor_cells)
        state_digest = _grid_digest(grid)

        desired: tuple[int, int] | None = None
        path_length: int | None = None
        desired_clear: bool | None = None
        if actor is not None and goal is not None:
            desired, path_length, desired_clear = self._plan_first_step(grid, actor, goal)

        move_available = tuple(action for action in available if action in _MOVE_NAMES)
        click_coord = (
            self._click_candidate(grid, exclude_roles=bool(move_available))
            if "ACTION6" in available
            else None
        )

        if click_coord is not None and (not move_available or actor is None or goal is None or desired is None):
            action = "ACTION6"
            self.state_actions.increment((state_digest, action))
            self.pending = _Pending(
                action=action,
                before=grid.copy(),
                actor_anchor=(_anchor_and_offsets(actor)[0] if actor else None),
                actor_cells=tuple(actor_cells),
                levels=levels,
                mode=mode,
                predicted_delta=None,
                predicted_confidence=0.0,
                predicted_clear=None,
                click_coord=click_coord,
            )
            return action, click_coord, "salience-ranked causal click probe"

        scored: list[tuple[float, str, tuple[int, int] | None, float, float]] = []
        for order, action in enumerate(available):
            if action == "ACTION6":
                continue
            predicted, confidence, samples = self.memory.prediction(mode, action)
            score = self._action_score(
                state_digest,
                mode,
                action,
                desired,
                predicted,
                confidence,
                samples,
            )
            score -= order * 1e-7
            scored.append((score, action, predicted, confidence, samples))

        if not scored:
            if click_coord is None:
                click_coord = (0, 0)
            action = "ACTION6"
            predicted = None
            confidence = 0.0
        else:
            _, action, predicted, confidence, _ = max(scored, key=lambda item: (item[0], item[1]))

        predicted_clear = self._predicted_clear(grid, actor, goal, predicted)
        if predicted is None and desired is not None and action in _MOVE_NAMES:
            predicted_clear = desired_clear

        self.state_actions.increment((state_digest, action))
        self.pending = _Pending(
            action=action,
            before=grid.copy(),
            actor_anchor=(_anchor_and_offsets(actor)[0] if actor else None),
            actor_cells=tuple(actor_cells),
            levels=levels,
            mode=mode,
            predicted_delta=predicted,
            predicted_confidence=confidence,
            predicted_clear=predicted_clear,
            click_coord=(click_coord if action == "ACTION6" else None),
        )
        reason = (
            f"verified-causal-plan desired={desired} path={path_length} "
            f"predicted={predicted} conf={confidence:.3f} stagnation={self.stagnation}"
        )
        return action, (click_coord if action == "ACTION6" else None), reason

    def deterministic_recovery(
        self,
        grid: np.ndarray,
        available: tuple[str, ...],
    ) -> tuple[str, tuple[int, int] | None]:
        state = _grid_digest(grid)
        simple = [action for action in available if action != "ACTION6"]
        if simple:
            action = min(simple, key=lambda candidate: (self.state_actions.get((state, candidate)), candidate))
            self.state_actions.increment((state, action))
            return action, None
        click = self._click_candidate(grid, exclude_roles=False) or (0, 0)
        return "ACTION6", click
