"""Strict ARC frame/action normalization and grid primitives for v0.43."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Iterator, Sequence
import numpy as np
from arcengine import GameAction

_ACTION_NAMES = tuple(f"ACTION{i}" for i in range(1, 8))
_MOVE_NAMES = _ACTION_NAMES[:4]
_CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ZERO_DELTA = (0, 0)


def _name(action: Any) -> str:
    raw = getattr(action, "name", action)
    text = str(raw)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def _to_grid(frame: Any) -> np.ndarray:
    """Extract the newest 2-D ARC grid as contiguous uint8."""
    candidates: list[Any] = [frame]
    if isinstance(frame, dict):
        candidates.extend(frame.get(key) for key in ("frame", "grid", "observation", "data") if key in frame)
    else:
        for attr in ("frame", "grid", "observation", "data"):
            try:
                candidates.append(getattr(frame, attr))
            except Exception:
                continue

    for value in candidates:
        if value is None:
            continue
        if isinstance(value, dict):
            for key in ("frame", "grid", "cells", "board"):
                if key in value:
                    value = value[key]
                    break
        try:
            array = np.asarray(value)
        except Exception:
            continue
        while array.ndim > 2:
            array = array[-1]
        if array.ndim != 2 or array.size == 0:
            continue
        if array.shape[0] > 64 or array.shape[1] > 64:
            continue
        if not np.issubdtype(array.dtype, np.number):
            continue
        array = np.ascontiguousarray(array, dtype=np.int16)
        if not np.isfinite(array).all() or int(array.min()) < 0 or int(array.max()) > 15:
            continue
        return array.astype(np.uint8, copy=False)
    raise ValueError("no valid ARC grid in frame")


def _iter_action_values(raw: Any) -> Iterator[Any]:
    if raw is None:
        return
    if isinstance(raw, dict):
        yield from raw.keys()
        return
    for attr in ("actions", "available_actions"):
        value = getattr(raw, attr, None)
        if value is not None:
            yield from _iter_action_values(value)
            return
    try:
        yield from list(raw)
    except TypeError:
        yield raw


def _available(frame: Any, fallback: Any = None) -> tuple[str, ...]:
    raw = getattr(frame, "available_actions", None)
    if raw is None and isinstance(frame, dict):
        raw = frame.get("available_actions")
    values = list(_iter_action_values(raw))
    if not values:
        values = list(_iter_action_values(fallback))

    result: list[str] = []
    for item in values:
        action_name = _name(item)
        if action_name in _ACTION_NAMES and action_name not in result:
            result.append(action_name)
    return tuple(result) or _ACTION_NAMES


def _make_action(name: str, coord: tuple[int, int] | None = None, reason: str = "") -> GameAction:
    if name not in _ACTION_NAMES:
        raise ValueError(f"unsupported action: {name}")
    action = getattr(GameAction, name)
    if name == "ACTION6":
        x, y = coord or (0, 0)
        action.set_data({"x": int(np.clip(x, 0, 63)), "y": int(np.clip(y, 0, 63))})
    action.reasoning = {"text": reason[:240]}
    return action


def _background(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel(), minlength=16).argmax())


@dataclass(frozen=True, slots=True)
class _Component:
    color: int
    cells: tuple[tuple[int, int], ...]
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    touches_border: bool


def _centroid(cells: Iterable[tuple[int, int]]) -> tuple[int, int]:
    points = tuple(cells)
    if not points:
        raise ValueError("cannot compute centroid of an empty component")
    return (
        round(sum(point[0] for point in points) / len(points)),
        round(sum(point[1] for point in points) / len(points)),
    )


def _components(grid: np.ndarray) -> list[_Component]:
    height, width = grid.shape
    background = _background(grid)
    seen = np.zeros((height, width), dtype=np.bool_)
    result: list[_Component] = []
    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if color == background or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in _CARDINALS:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ordered = tuple(sorted(cells))
            ys = [point[0] for point in ordered]
            xs = [point[1] for point in ordered]
            bbox = (min(ys), min(xs), max(ys), max(xs))
            result.append(
                _Component(
                    color=color,
                    cells=ordered,
                    area=len(ordered),
                    bbox=bbox,
                    centroid=_centroid(ordered),
                    touches_border=(
                        bbox[0] == 0
                        or bbox[1] == 0
                        or bbox[2] == height - 1
                        or bbox[3] == width - 1
                    ),
                )
            )
    return result


def _anchor_and_offsets(component: _Component) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    y0, x0, _, _ = component.bbox
    offsets = tuple((y - y0, x - x0) for y, x in component.cells)
    return (y0, x0), offsets


def _grid_digest(grid: np.ndarray) -> bytes:
    digest = hashlib.blake2b(digest_size=12, person=b"arc3v43state")
    digest.update(bytes(grid.shape))
    digest.update(grid.tobytes())
    return digest.digest()


def _mode_digest(grid: np.ndarray, actor_cells: Sequence[tuple[int, int]]) -> bytes:
    """Context key that removes only the tracked actor footprint."""
    canonical = grid.copy()
    background = _background(canonical)
    for y, x in actor_cells:
        if 0 <= y < canonical.shape[0] and 0 <= x < canonical.shape[1]:
            canonical[y, x] = background
    digest = hashlib.blake2b(digest_size=10, person=b"arc3v43mode")
    digest.update(bytes(canonical.shape))
    digest.update(canonical.tobytes())
    return digest.digest()
