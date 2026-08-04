"""Bounded proposal-only adapter for a local Gemma 4 OpenAI-compatible endpoint."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Iterable
from urllib import request

try:
    from .sovereign_v45_perception import action_key
except ImportError:
    from sovereign_v45_perception import action_key


@dataclass(frozen=True, slots=True)
class GemmaProposal:
    action_key: str
    coordinate: tuple[int, int] | None
    confidence: float
    hypothesis: str


class Gemma4Oracle:
    """Stateless, bounded oracle. The deterministic core remains authoritative."""

    def __init__(self) -> None:
        self.base_url = os.environ.get("ARC3_GEMMA_URL", "").rstrip("/")
        self.model = os.environ.get("ARC3_GEMMA_SERVED_NAME", "arc3-gemma4-31b")
        self.timeout = float(os.environ.get("ARC3_GEMMA_TIMEOUT_SECONDS", "8"))
        self.max_tokens = int(os.environ.get("ARC3_GEMMA_MAX_TOKENS", "192"))
        self.enabled = bool(self.base_url)

    @staticmethod
    def _grid_text(grid: Any) -> str:
        rows = []
        for row in grid.tolist():
            rows.append(" ".join(str(int(value)) for value in row))
        return "\n".join(rows)

    def propose(self, scene: Any, actions: Iterable[Any], diagnostics: dict[str, Any]) -> GemmaProposal | None:
        if not self.enabled:
            return None
        legal = {action_key(action): action for action in actions}
        prompt = {
            "task": "Select one legal ARC action from current evidence. Return JSON only.",
            "legal_actions": sorted(legal),
            "grid_shape": list(scene.grid.shape),
            "grid": self._grid_text(scene.grid),
            "objects": [
                {
                    "track_id": int(track.track_id),
                    "value": int(track.component.value),
                    "area": int(track.component.area),
                    "bbox": list(track.component.bbox),
                    "centroid": [float(v) for v in track.component.centroid],
                    "velocity": [float(v) for v in track.velocity],
                    "morphology": track.component.morphology,
                }
                for track in scene.tracks
            ],
            "diagnostics": diagnostics,
            "output_schema": {
                "action": "one legal action string",
                "x": "integer or null",
                "y": "integer or null",
                "confidence": "number from 0 to 1",
                "hypothesis": "short causal reason",
            },
        }
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a bounded ARC action hypothesis generator. Output one JSON object and nothing else."},
                    {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
                ],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"].strip()
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                return None
            item = json.loads(match.group(0))
            key = str(item["action"]).upper()
            if key not in legal:
                return None
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
            coordinate = None
            if item.get("x") is not None and item.get("y") is not None:
                x, y = int(item["x"]), int(item["y"])
                height, width = scene.grid.shape
                if not (0 <= x < width and 0 <= y < height):
                    return None
                coordinate = (x, y)
            return GemmaProposal(
                action_key=key,
                coordinate=coordinate,
                confidence=confidence,
                hypothesis=str(item.get("hypothesis", ""))[:240],
            )
        except Exception:
            return None
