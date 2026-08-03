"""ARC3 Sovereign NumPy v0.42 competition agent.

A deterministic, bounded, evidence-driven controller for ARC-AGI-3. It learns
per-game action semantics from committed frame transitions, detects causal mode
changes, plans on the visible grid, and falls back to information-gain probes.

The implementation is intentionally self-contained because the official starter
splices this file into the Kaggle submission notebook.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_ACTION_NAMES = tuple(f"ACTION{i}" for i in range(1, 8))
_MOVE_NAMES = _ACTION_NAMES[:4]
_CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _name(action: Any) -> str:
    raw = getattr(action, "name", action)
    text = str(raw)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def _to_grid(frame: Any) -> np.ndarray:
    """Extract the newest 2-D uint8 grid from FrameData/dicts/lists/arrays."""
    candidates: list[Any] = [frame]
    if isinstance(frame, dict):
        candidates.extend(frame.get(k) for k in ("frame", "grid", "observation", "data") if k in frame)
    else:
        for attr in ("frame", "grid", "observation", "data"):
            try:
                candidates.append(getattr(frame, attr))
            except Exception:
                pass
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, dict):
            for key in ("frame", "grid", "cells", "board"):
                if key in value:
                    value = value[key]
                    break
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        while arr.ndim > 2:
            arr = arr[-1]
        if arr.ndim != 2 or arr.size == 0:
            continue
        if arr.shape[0] > 64 or arr.shape[1] > 64:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        arr = np.ascontiguousarray(arr, dtype=np.int16)
        if int(arr.min()) < 0 or int(arr.max()) > 15:
            continue
        return arr.astype(np.uint8, copy=False)
    raise ValueError("no valid ARC grid in frame")


def _available(frame: Any) -> tuple[str, ...]:
    raw = getattr(frame, "available_actions", None)
    if raw is None and isinstance(frame, dict):
        raw = frame.get("available_actions")
    if raw is None:
        return _ACTION_NAMES
    result: list[str] = []
    try:
        values = list(raw)
    except TypeError:
        values = [raw]
    for item in values:
        n = _name(item)
        if n in _ACTION_NAMES and n not in result:
            result.append(n)
    return tuple(result) or _ACTION_NAMES


def _make_action(name: str, coord: tuple[int, int] | None = None, reason: str = "") -> GameAction:
    action = getattr(GameAction, name)
    if name == "ACTION6":
        x, y = coord or (0, 0)
        action.set_data({"x": int(np.clip(x, 0, 63)), "y": int(np.clip(y, 0, 63))})
    action.reasoning = {"text": reason[:240]}
    return action


def _components(grid: np.ndarray) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    h, w = grid.shape
    bg = int(Counter(grid.ravel().tolist()).most_common(1)[0][0])
    seen = np.zeros((h, w), dtype=np.bool_)
    out: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if color == bg or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in _CARDINALS:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((color, tuple(sorted(cells))))
    return out


def _centroid(cells: Iterable[tuple[int, int]]) -> tuple[int, int]:
    pts = tuple(cells)
    return (round(sum(p[0] for p in pts) / len(pts)), round(sum(p[1] for p in pts) / len(pts)))


def _mode_digest(grid: np.ndarray, player: tuple[int, int] | None) -> bytes:
    """Structural context key that removes the moving player's cell."""
    copy = grid.copy()
    if player is not None:
        y, x = player
        if 0 <= y < copy.shape[0] and 0 <= x < copy.shape[1]:
            copy[y, x] = Counter(copy.ravel().tolist()).most_common(1)[0][0]
    h = hashlib.blake2b(digest_size=8, person=b"arc3v42")
    h.update(bytes(copy.shape))
    h.update(copy.tobytes())
    return h.digest()


@dataclass(slots=True)
class _Pending:
    action: str
    before: np.ndarray
    player: tuple[int, int] | None
    levels: int
    mode: bytes


class _SemanticMemory:
    """Bounded contextual action-effect posterior with change detection."""

    def __init__(self, max_modes: int = 32) -> None:
        self.max_modes = max_modes
        self.effects: dict[bytes, dict[str, Counter[tuple[int, int]]]] = {}
        self.quality: dict[bytes, dict[str, list[float]]] = {}
        self.epochs: dict[bytes, int] = defaultdict(int)
        self.order: deque[bytes] = deque()

    def _touch(self, mode: bytes) -> None:
        if mode not in self.effects:
            if len(self.effects) >= self.max_modes:
                old = self.order.popleft()
                self.effects.pop(old, None)
                self.quality.pop(old, None)
            self.effects[mode] = defaultdict(Counter)
            self.quality[mode] = defaultdict(list)
            self.order.append(mode)

    def update(self, mode: bytes, action: str, delta: tuple[int, int] | None, utility: float) -> None:
        self._touch(mode)
        if delta is not None:
            counts = self.effects[mode][action]
            if counts and sum(counts.values()) >= 3 and delta not in counts and counts.most_common(1)[0][1] >= 3:
                self.epochs[mode] += 1
                self.effects[mode] = defaultdict(Counter)
                self.quality[mode] = defaultdict(list)
            self.effects[mode][action][delta] += 1
        values = self.quality[mode][action]
        values.append(float(utility))
        if len(values) > 24:
            del values[:-24]

    def predicted_delta(self, mode: bytes, action: str) -> tuple[tuple[int, int] | None, float]:
        counts = self.effects.get(mode, {}).get(action)
        if not counts:
            votes: Counter[tuple[int, int]] = Counter()
            sources = 0
            for mapping in self.effects.values():
                c = mapping.get(action)
                if c and sum(c.values()) >= 3:
                    d, n = c.most_common(1)[0]
                    if n / sum(c.values()) >= 0.8:
                        votes[d] += 1
                        sources += 1
            if sources and len(votes) == 1:
                return next(iter(votes)), min(0.85, 0.55 + 0.1 * sources)
            return None, 0.0
        delta, count = counts.most_common(1)[0]
        total = sum(counts.values())
        return delta, count / total

    def utility(self, mode: bytes, action: str) -> float:
        values = self.quality.get(mode, {}).get(action, ())
        return float(sum(values) / len(values)) if values else 0.0


class _CognitiveCore:
    def __init__(self) -> None:
        self.memory = _SemanticMemory()
        self.pending: _Pending | None = None
        self.player_color: int | None = None
        self.goal_color: int | None = None
        self.player: tuple[int, int] | None = None
        self.goal: tuple[int, int] | None = None
        self.visits: Counter[bytes] = Counter()
        self.tried_clicks: set[tuple[int, int]] = set()
        self.stagnation = 0

    @staticmethod
    def _candidate_roles(grid: np.ndarray) -> tuple[tuple[int, int] | None, tuple[int, int] | None, int | None, int | None]:
        comps = _components(grid)
        if not comps:
            return None, None, None, None
        by_color: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
        for color, cells in comps:
            by_color[color].append(cells)
        player_color = 1 if 1 in by_color else None
        goal_color = 3 if 3 in by_color else None
        if player_color is None:
            player_color = min(by_color, key=lambda c: (min(len(v) for v in by_color[c]), len(by_color[c]), c))
        if goal_color is None:
            candidates = [c for c in by_color if c != player_color]
            goal_color = min(candidates, key=lambda c: (len(by_color[c]), min(len(v) for v in by_color[c]), c)) if candidates else None
        p_cells = min(by_color[player_color], key=len)
        g_cells = min(by_color[goal_color], key=len) if goal_color is not None else None
        return _centroid(p_cells), (_centroid(g_cells) if g_cells else None), player_color, goal_color

    def observe(self, grid: np.ndarray, levels: int) -> None:
        if self.pending is not None:
            p_after, _, _, _ = self._candidate_roles(grid)
            if self.player_color is not None:
                locs = np.argwhere(grid == self.player_color)
                if len(locs):
                    p_after = tuple(map(int, locs[0]))
            delta = None
            if self.pending.player is not None and p_after is not None:
                delta = (p_after[0] - self.pending.player[0], p_after[1] - self.pending.player[1])
                if abs(delta[0]) + abs(delta[1]) > 4:
                    delta = None
            changed = not np.array_equal(self.pending.before, grid)
            progressed = levels > self.pending.levels
            utility = 4.0 if progressed else (1.0 if changed else -1.0)
            if delta == (0, 0):
                utility -= 0.7
            self.memory.update(self.pending.mode, self.pending.action, delta, utility)
            self.stagnation = 0 if changed or progressed else self.stagnation + 1
            self.pending = None

        p, g, pc, gc = self._candidate_roles(grid)
        if self.player_color is None:
            self.player_color = pc
        if self.goal_color is None:
            self.goal_color = gc
        if self.player_color is not None:
            locs = np.argwhere(grid == self.player_color)
            if len(locs):
                p = tuple(map(int, locs[0]))
        if self.goal_color is not None:
            locs = np.argwhere(grid == self.goal_color)
            if len(locs):
                g = tuple(map(int, locs[0]))
        self.player, self.goal = p, g
        self.visits[hashlib.blake2b(grid.tobytes(), digest_size=8).digest()] += 1

    @staticmethod
    def _passable(grid: np.ndarray, player: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
        bg = int(Counter(grid.ravel().tolist()).most_common(1)[0][0])
        mask = grid == bg
        mask[player] = True
        mask[goal] = True
        return mask

    @staticmethod
    def _bfs_first_step(grid: np.ndarray, player: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int] | None:
        passable = _CognitiveCore._passable(grid, player, goal)
        q = deque([player])
        prev: dict[tuple[int, int], tuple[int, int] | None] = {player: None}
        h, w = grid.shape
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for dy, dx in _CARDINALS:
                nxt = (cur[0] + dy, cur[1] + dx)
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and passable[nxt] and nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)
        if goal not in prev:
            return None
        cur = goal
        while prev[cur] != player:
            parent = prev[cur]
            if parent is None:
                return None
            cur = parent
        return (cur[0] - player[0], cur[1] - player[1])

    def choose(self, grid: np.ndarray, available: tuple[str, ...], levels: int) -> tuple[str, tuple[int, int] | None, str]:
        p, g = self.player, self.goal
        mode = _mode_digest(grid, p)

        if "ACTION6" in available and not any(a in available for a in _MOVE_NAMES):
            candidates = []
            for color, cells in _components(grid):
                y, x = _centroid(cells)
                if (x, y) not in self.tried_clicks:
                    candidates.append((len(cells), color, x, y))
            if candidates:
                _, _, x, y = min(candidates)
                self.tried_clicks.add((x, y))
                action = "ACTION6"
                self.pending = _Pending(action, grid.copy(), p, levels, mode)
                return action, (x, y), "rare-component information-gain click"

        desired = self._bfs_first_step(grid, p, g) if p is not None and g is not None else None
        scores: list[tuple[float, str]] = []
        for idx, action in enumerate(available):
            if action == "ACTION6":
                continue
            predicted, confidence = self.memory.predicted_delta(mode, action)
            score = self.memory.utility(mode, action)
            if desired is not None and predicted is not None:
                score += 8.0 * confidence if predicted == desired else -3.0 * confidence
            elif desired is not None and action in _MOVE_NAMES:
                score += 2.0
            if predicted == (0, 0):
                score -= 4.0
            if action == "ACTION7":
                score -= 2.5
            score -= idx * 1e-6
            scores.append((score, action))

        action = max(scores)[1] if scores else available[0]
        self.pending = _Pending(action, grid.copy(), p, levels, mode)
        return action, None, f"causal-plan desired={desired} stagnation={self.stagnation}"


class MyAgent(Agent):
    """Competition adapter with bounded recovery and no random action path."""

    MAX_ACTIONS = 400

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.core = _CognitiveCore()
        self.recoveries = 0

    @property
    def name(self) -> str:
        return f"{super().name}.sovereign_numpy_v42"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.core = _CognitiveCore()
            return GameAction.RESET
        try:
            grid = _to_grid(latest_frame)
            levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
            self.core.observe(grid, levels)
            available = _available(latest_frame)
            action, coord, reason = self.core.choose(grid, available, levels)
            return _make_action(action, coord, reason)
        except Exception as exc:
            self.recoveries += 1
            if self.recoveries > 4:
                raise RuntimeError("bounded ARC3 recovery budget exhausted") from exc
            available = _available(latest_frame)
            safe = next((a for a in available if a != "ACTION6"), available[0])
            return _make_action(safe, (0, 0) if safe == "ACTION6" else None, f"recovery:{type(exc).__name__}")
