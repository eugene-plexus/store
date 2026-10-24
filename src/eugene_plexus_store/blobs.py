"""The blob folder: each file's bytes once, named by their SHA-256.

**No name here comes from a request.** A blob's path is computed from a
digest this process calculated, checked to be 64 hex digits, and a part's
from an id this process made, checked to be its own pattern; nothing a
caller sends can make a path. Two owners who keep the same bytes share
one blob, and nothing about either request shows it: both are written in
full, hashed, and only then found to exist.

Writes go to a temporary file first and are renamed into place, so a blob
is either whole or absent. On Windows a blob another request is reading
cannot be unlinked; the unlink is tried again at the next sweep, which
removes every blob no file names (`orphans`).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

log = logging.getLogger(__name__)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_PART = re.compile(r"^part_[0-9a-f]{24}$")
CHUNK = 1024 * 1024
#: A temporary file older than this is left over from a crash.
STALE_TEMP_SECONDS = 3600


@dataclass(frozen=True)
class Written:
    path: Path
    size: int
    sha256: str
    md5: str


@dataclass(frozen=True)
class Digest:
    size: int
    sha256: str
    md5: str


class Blobs:
    def __init__(self, folder: Path) -> None:
        self.root = folder / "blobs"
        self.parts_root = folder / "parts"
        self.temp_root = folder / "tmp"

    def prepare(self) -> None:
        for folder in (self.root, self.parts_root, self.temp_root):
            folder.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        if not _SHA.match(sha256):
            raise ValueError("a blob is named by a SHA-256 in hex")
        return self.root / sha256[:2] / sha256

    def part_path(self, part_id: str) -> Path:
        if not _PART.match(part_id):
            raise ValueError("not a part id")
        return self.parts_root / part_id

    def temp_path(self) -> Path:
        return self.temp_root / f"{uuid.uuid4().hex}.tmp"

    def write_temp(self, chunks: Iterable[bytes]) -> Written:
        """Write `chunks` to a new temporary file, hashing as it goes."""
        path = self.temp_path()
        sha = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        size = 0
        try:
            with path.open("xb") as handle:
                for chunk in chunks:
                    handle.write(chunk)
                    sha.update(chunk)
                    md5.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            self.discard(path)
            raise
        return Written(path, size, sha.hexdigest(), md5.hexdigest())

    @staticmethod
    def digest(sources: Iterable[Iterable[bytes]]) -> Digest:
        """Size and digests of the bytes, read and not written anywhere."""
        sha = hashlib.sha256()
        md5 = hashlib.md5(usedforsecurity=False)
        size = 0
        for chunks in sources:
            for chunk in chunks:
                sha.update(chunk)
                md5.update(chunk)
                size += len(chunk)
        return Digest(size, sha.hexdigest(), md5.hexdigest())

    def keep(self, written: Written) -> bool:
        """Move a written temp file into place as its blob. False when the
        blob was there already (the temp file is then removed)."""
        target = self.path_for(written.sha256)
        if target.exists():
            self.discard(written.path)
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(written.path, target)
        return True

    def keep_part(self, written: Written, part_id: str) -> None:
        os.replace(written.path, self.part_path(part_id))

    def read(self, sha256: str) -> Iterator[bytes]:
        with self.path_for(sha256).open("rb") as handle:
            while chunk := handle.read(CHUNK):
                yield chunk

    def read_part(self, part_id: str) -> Iterator[bytes]:
        with self.part_path(part_id).open("rb") as handle:
            while chunk := handle.read(CHUNK):
                yield chunk

    def unlink(self, sha256: str) -> bool:
        """Remove a blob; False when it could not be (it is retried)."""
        return self.discard(self.path_for(sha256))

    def unlink_part(self, part_id: str) -> bool:
        return self.discard(self.part_path(part_id))

    @staticmethod
    def discard(path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not delete %s yet (%s); the next sweep tries again", path, exc)
            return False
        return True

    def blob_names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return [
            path.name
            for folder in self.root.iterdir()
            if folder.is_dir()
            for path in folder.iterdir()
            if _SHA.match(path.name)
        ]

    def part_names(self) -> list[str]:
        if not self.parts_root.is_dir():
            return []
        return [path.name for path in self.parts_root.iterdir() if _PART.match(path.name)]

    def stale_temps(self, *, now: float | None = None) -> list[Path]:
        if not self.temp_root.is_dir():
            return []
        cutoff = (time.time() if now is None else now) - STALE_TEMP_SECONDS
        found = []
        for path in self.temp_root.iterdir():
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    found.append(path)
        return found

    def size_of(self, sha256: str) -> int:
        try:
            return self.path_for(sha256).stat().st_size
        except OSError:
            return 0


def chunked(stream: BinaryIO) -> Iterator[bytes]:
    while chunk := stream.read(CHUNK):
        yield chunk
