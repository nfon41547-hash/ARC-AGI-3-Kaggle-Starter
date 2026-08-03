"""Verified embedded ARC3 Sovereign Pure NumPy v0.41 runtime loader."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

_RUNTIME_VERSION = "0.41.0"
_RUNTIME_SHA256 = "1fda15b9725e388f9d98f9643cd9e1897803de6267f8102ac1f6af7936b71026"
_PARTS = (
    "sovereign_payload.part00.b85",
    "sovereign_payload.part01.b85",
    "sovereign_payload.part02.b85",
)
_MAX_FILES = 256
_MAX_UNCOMPRESSED = 8 * 1024 * 1024


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        raise RuntimeError(f"unsafe runtime archive member: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe runtime archive member: {name!r}")
    if "\\" in name or ":" in path.parts[0]:
        raise RuntimeError(f"unsafe runtime archive member: {name!r}")
    return path


def _read_payload() -> bytes:
    base = Path(__file__).resolve().parent
    encoded = "".join(
        (base / name).read_text(encoding="ascii").strip()
        for name in _PARTS
    )
    raw = base64.b85decode(encoded.encode("ascii"))
    if hashlib.sha256(raw).hexdigest() != _RUNTIME_SHA256:
        raise RuntimeError("embedded ARC3 runtime checksum mismatch")
    return raw


def ensure_runtime() -> Path:
    """Verify, extract, and activate the embedded runtime exactly once."""
    raw = _read_payload()
    root = Path(tempfile.gettempdir()) / (
        f"arc3-sovereign-numpy-{_RUNTIME_VERSION}-{_RUNTIME_SHA256[:16]}"
    )
    marker = root / ".verified"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != _RUNTIME_SHA256:
        staging = root.with_name(root.name + f".staging-{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=False)
        archive_path = staging / "runtime.zip"
        archive_path.write_bytes(raw)
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_FILES:
                raise RuntimeError("embedded runtime contains too many files")
            total = 0
            names: set[str] = set()
            for info in infos:
                member = _safe_member(info.filename)
                if info.filename in names:
                    raise RuntimeError(
                        f"duplicate runtime archive member: {info.filename}"
                    )
                names.add(info.filename)
                if info.flag_bits & 0x1:
                    raise RuntimeError("encrypted runtime archive member is forbidden")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise RuntimeError("runtime archive symlink is forbidden")
                total += int(info.file_size)
                if total > _MAX_UNCOMPRESSED:
                    raise RuntimeError("embedded runtime exceeds extraction limit")
                target = staging / Path(*member.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not info.is_dir():
                    with archive.open(info) as source, target.open("wb") as sink:
                        shutil.copyfileobj(source, sink, length=1 << 20)
        archive_path.unlink(missing_ok=True)

        manifest_path = staging / "ARC3_RUNTIME_MANIFEST.json"
        runtime_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if runtime_manifest.get("version") != _RUNTIME_VERSION:
            raise RuntimeError("embedded runtime version mismatch")
        for record in runtime_manifest.get("files", []):
            path = staging / record["path"]
            data = path.read_bytes()
            if len(data) != int(record["size"]):
                raise RuntimeError(f"runtime size mismatch: {record['path']}")
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise RuntimeError(f"runtime digest mismatch: {record['path']}")

        (staging / ".verified").write_text(
            _RUNTIME_SHA256 + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(root, ignore_errors=True)
        staging.replace(root)

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


__all__ = ["ensure_runtime"]
