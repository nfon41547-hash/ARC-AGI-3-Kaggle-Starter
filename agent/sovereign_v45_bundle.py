"""Activate the deterministic ARC3 Sovereign v0.45 source bundle.

The bundle is a regular ZIP of auditable Python source, not bytecode. It is
verified before being inserted into ``sys.path``. The Kaggle notebook embeds the
same bytes and therefore runs the exact source verified in this repository.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import zipfile

_RUNTIME_SHA256 = "7efbde6a558c942a8cb1b57d54310e2b530cb55fc73d55e2bf06ea9b6e701793"
_REQUIRED = {
    "sovereign_v45_perception.py",
    "sovereign_v45_rules.py",
    "sovereign_v45_core.py",
    "sovereign_v45_gif.py",
}


def ensure_runtime() -> Path:
    path = Path(__file__).with_name("sovereign_v45_runtime.zip")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _RUNTIME_SHA256:
        raise RuntimeError("ARC3 v0.45 runtime bundle checksum mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != _REQUIRED:
            raise RuntimeError("ARC3 v0.45 runtime bundle inventory mismatch")
        for info in archive.infolist():
            if info.is_dir() or info.flag_bits & 0x1:
                raise RuntimeError("invalid ARC3 v0.45 runtime member")
            if Path(info.filename).name != info.filename:
                raise RuntimeError("unsafe ARC3 v0.45 runtime member path")
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
    return path


__all__ = ["ensure_runtime"]
