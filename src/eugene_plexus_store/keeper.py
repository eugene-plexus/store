"""Files and uploads, each kept for one owner until it is deleted or ends.

Every method is synchronous and runs in a worker thread (the routes call
it through `run_in_threadpool`). Every change to a row or a blob happens
under the database's one lock, so a delete and an upload of the same
bytes cannot interleave: a blob is unlinked only when, under that lock,
no file names it.

**An owner sees only its own objects.** Every lookup is `id = ? AND
owner = ?`, one indexed query whether the id is missing, ended or
another owner's, and each of those answers the same 404 (`not_found`).
A part is found only through its own upload and owner, so no upload can
join another's parts.

**The quota bounds the disk, not only the rows** (review of row 53):

* Capacity is reserved, under the lock, before any byte is written: a
  whole file's size, or a part's against its upload's declared size.
  What is in flight counts against the owner until it is kept or dropped.
* An upload is joined once at a time; a second `complete` while one runs
  is refused. Its parts are deleted as they are copied, so joining never
  holds the file twice.
* A file that has ended counts until its bytes are deleted. An owner at
  its quota has its own ended files deleted first, then is asked again.

**A removed person is hidden, then deleted** (call R53-REMOVE, Choice 2:
row 7's rules). An owner in `removed` is answered as if it kept nothing:
every lookup and listing leaves its objects out, the same 404 as an id
that does not exist, and nothing new is kept for it. Its objects stay on
disk, counted against it, until its date (`delete_at`, fixed when it was
hidden) or the owner's Delete now; a person the root has again before
that is restored with the same ids and bytes.

**Nothing already under way comes back** (second review). Every write
counts as busy for its owner from the moment its room is taken until it has
ended or been refused. Every commit (a whole file, an assembled upload, a
part) re-checks the owner's removal state in its own transaction and refuses
for a hidden owner, so a write paused across the removal publishes nothing.
Deleting a removed person waits for their busy writes, then deletes; the
marker is the retry record and is removed last, only once every row, blob and
`.bak` copy is gone. A marker whose date has passed is final: a restoration
after it changes nothing. Deletion is compare-and-delete on the marker's own
generation, so an answer that is stale by the time it is used cannot delete a
person restored meanwhile.

**A write is busy from admission, not from its body** (third review): the
routes call `begin_write` before they read a byte of the body, which counts
the write for its owner and refuses a hidden or deleted owner on the spot.
**A deleted person leaves a tombstone**: once everything is gone the marker
becomes state `deleted` (the id and the date, no bytes), never removed. A
removed person's id is never reused, so it never blocks a real person; every
commit refuses a hidden or deleted owner.

**Deletion is fenced by generation.** Every file and upload carries the
store's generation counter at its making. The sweep reads the counter
before it asks the root who was removed, and deletes only what is older:
a person restored under the same id who keeps something new in between
keeps it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .blobs import Blobs, Written
from .config import GIB, ConfigStore
from .database import BackupPurgeFailed, Database
from .problems import StoreProblem, not_found

log = logging.getLogger(__name__)

OWNER = re.compile(r"^(install|(person|key|component):[A-Za-z0-9._@:-]{1,200})$")
FILE_ID = re.compile(r"^file-[0-9a-f]{24}$")
UPLOAD_ID = re.compile(r"^upload_[0-9a-f]{24}$")
PART_ID = re.compile(r"^part_[0-9a-f]{24}$")
PURPOSES = ("assistants", "batch", "fine-tune", "vision", "user_data", "evals")
#: OpenAI's: an upload not completed within an hour is deleted with its parts.
UPLOAD_SECONDS = 3600
#: OpenAI's longest `expires_after`: thirty days.
MAX_SECONDS = 30 * 24 * 3600
#: The gateway's body limit, and so the most one part or one file may be.
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_PARTS = 10000
DEFAULT_MIME = "application/octet-stream"
#: An object that has ended: it has an end, and the end has passed.
ENDED = "expires_at IS NOT NULL AND expires_at <= ?"
#: An object that has not ended.
LIVE = "(expires_at IS NULL OR expires_at > ?)"
#: An object whose owner was not removed (a hidden owner keeps nothing visible).
SHOWN = "owner NOT IN (SELECT owner FROM removed)"


def _new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(12)


def _removed() -> StoreProblem:
    return StoreProblem(
        403,
        "Removed",
        "Eugene removed this person: what they kept here is hidden or deleted, and nothing "
        "new is kept for them.",
    )


def _human(size: int) -> str:
    if size >= GIB:
        return f"{size / GIB:.2f} GiB"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MiB"
    return f"{size} bytes"


@dataclass
class SweepCounts:
    files: int = 0
    uploads: int = 0
    parts: int = 0
    blobs: int = 0
    freed_bytes: int = 0
    responses: int = 0
    conversations: int = 0
    unlink_later: list[str] = field(default_factory=list)

    def add(self, other: SweepCounts) -> None:
        self.files += other.files
        self.uploads += other.uploads
        self.parts += other.parts
        self.blobs += other.blobs
        self.freed_bytes += other.freed_bytes
        self.responses += other.responses
        self.conversations += other.conversations


class Keeper:
    def __init__(
        self, folder: Path, config: ConfigStore, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.folder = folder
        self.config = config
        self.clock = clock
        self.database = Database(folder)
        self.blobs = Blobs(folder)
        # Bytes and files on their way to disk, by owner, and part bytes by upload.
        self._reserved_bytes: dict[str, int] = {}
        self._reserved_files: dict[str, int] = {}
        self._part_bytes: dict[str, int] = {}
        self._part_owner: dict[str, str] = {}
        self._assembling: set[str] = set()
        # Writes under way, by owner; a removal waits for its owner's to end.
        self._busy: dict[str, int] = {}
        self._guard = threading.Condition(threading.Lock())
        #: How long a deletion waits for the owner's writes to end, in seconds.
        self.idle_wait_seconds = 120.0

    def open(self) -> None:
        self.blobs.prepare()
        self.database.open()
        for stale in self.blobs.temp_root.iterdir():
            Blobs.discard(stale)

    def close(self) -> None:
        self.database.close()

    def now(self) -> int:
        return int(self.clock())

    # -- shapes ------------------------------------------------------------

    @staticmethod
    def _file(row: sqlite3.Row) -> dict[str, Any]:
        kept: dict[str, Any] = {
            "id": row["id"],
            "filename": row["filename"],
            "purpose": row["purpose"],
            "bytes": row["bytes"],
            "mimeType": row["mime_type"],
            "sha256": row["sha256"],
            "createdAt": row["created_at"],
            "status": "processed",
        }
        if row["expires_at"] is not None:
            kept["expiresAt"] = row["expires_at"]
        return kept

    def _upload(self, row: sqlite3.Row) -> dict[str, Any]:
        status = row["status"]
        if status == "pending" and row["expires_at"] <= self.now():
            status = "expired"
        upload: dict[str, Any] = {
            "id": row["id"],
            "filename": row["filename"],
            "purpose": row["purpose"],
            "bytes": row["bytes"],
            "mimeType": row["mime_type"],
            "createdAt": row["created_at"],
            "expiresAt": row["expires_at"],
            "status": status,
        }
        if row["file_id"]:
            found = self.database.connection.execute(
                "SELECT * FROM files WHERE id = ? AND owner = ?", (row["file_id"], row["owner"])
            ).fetchone()
            if found is not None:
                upload["file"] = self._file(found)
        return upload

    # -- checks ------------------------------------------------------------

    @staticmethod
    def check_owner(owner: str) -> str:
        if not OWNER.match(owner):
            raise StoreProblem(
                400,
                "Bad owner",
                "X-Eugene-Plexus-Owner must be install, person:<id>, key:<id> or component:<sub>.",
            )
        return owner

    def _files_on(self) -> None:
        if not self.config.files_enabled():
            raise StoreProblem(
                403,
                "Files are off",
                "This install keeps no files: the owner turned it off in Settings -> Store.",
            )

    def _seconds(self, expires_after: int | None) -> int | None:
        """The life a new file gets: what its app asked, else the install's
        default, which by default is none (kept until deleted, as OpenAI keeps)."""
        if expires_after is None:
            return self.config.default_seconds()
        if expires_after < 1:
            raise StoreProblem(400, "Bad end", "expires_after: must be at least 1 second.")
        if expires_after > MAX_SECONDS:
            raise StoreProblem(
                400,
                "Bad end",
                f"expires_after: at most {MAX_SECONDS} seconds (30 days), as OpenAI allows.",
            )
        return expires_after

    @staticmethod
    def _purpose(purpose: str) -> str:
        if purpose not in PURPOSES:
            raise StoreProblem(
                400, "Bad purpose", f"purpose: must be one of {', '.join(PURPOSES)}."
            )
        return purpose

    @staticmethod
    def _filename(name: str) -> str:
        cleaned = name.strip()
        if not cleaned or len(cleaned) > 255:
            raise StoreProblem(400, "Bad filename", "filename: 1 to 255 characters.")
        return cleaned

    # -- the quota -----------------------------------------------------------

    def _used(self, owner: str) -> tuple[int, int]:
        """(files, bytes) the owner holds on disk or is about to: every file row
        until its bytes are deleted (ended or not), uploads in progress until
        swept, and what is in flight. Called under the lock."""
        connection = self.database.connection
        files = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM files WHERE owner = ?", (owner,)
        ).fetchone()
        pending = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM uploads "
            "WHERE owner = ? AND status = 'pending'",
            (owner,),
        ).fetchone()
        # Row 54: kept responses and conversations count their bytes, not as files.
        kept = connection.execute(
            "SELECT (SELECT COALESCE(SUM(bytes), 0) FROM responses WHERE owner = ?) "
            "+ (SELECT COALESCE(SUM(bytes), 0) FROM conversations WHERE owner = ?)",
            (owner, owner),
        ).fetchone()
        with self._guard:
            flying_files = self._reserved_files.get(owner, 0)
            flying_bytes = self._reserved_bytes.get(owner, 0)
            staging = [
                (upload_id, size)
                for upload_id, size in self._part_bytes.items()
                if self._part_owner.get(upload_id) == owner
            ]
        # A part still being written counts until its bytes are gone, even when its
        # upload was cancelled or ended meanwhile: only a pending upload's declared
        # size already covers it.
        for upload_id, size in staging:
            row = connection.execute(
                "SELECT status FROM uploads WHERE id = ?", (upload_id,)
            ).fetchone()
            if row is None or row["status"] != "pending":
                flying_bytes += size
        return (
            int(files[0]) + int(pending[0]) + flying_files,
            int(files[1]) + int(pending[1]) + flying_bytes + int(kept[0]),
        )

    def _over(self, owner: str, adding: int, files: int = 1) -> StoreProblem | None:
        count, used = self._used(owner)
        most_files = self.config.quota_files()
        most_bytes = self.config.quota_bytes()
        if files and count + files > most_files:
            return StoreProblem(
                413,
                "Over quota",
                f"This owner keeps {count} file{'' if count == 1 else 's'}, and the most it "
                f"may keep is {most_files} (Settings -> Store). Delete files it no longer needs.",
            )
        if used + adding > most_bytes:
            return StoreProblem(
                413,
                "Over quota",
                f"This owner keeps {_human(used)} ({used} bytes); this adds {_human(adding)} "
                f"({adding} bytes), and the most it may keep is {_human(most_bytes)} "
                f"({most_bytes} bytes) (Settings -> Store). Delete "
                + ("files" if files else "responses or conversations")
                + " it no longer needs.",
            )
        return None

    def _check_quota(self, owner: str, adding: int, files: int = 1) -> None:
        """Room for one more object of `adding` bytes, or the 413 with the numbers.
        An owner at its quota has its own ended objects deleted first. A removed
        person keeps nothing new, so no new object reclaims their hidden ones.
        `files` is how many files it adds: none for a kept response or a
        conversation (row 54), which count only their bytes. Called under the
        lock, outside any transaction."""
        if self._is_removed(owner):
            raise _removed()
        problem = self._over(owner, adding, files)
        if problem is None:
            return
        reclaimed = self._delete_where(f"owner = ? AND {ENDED}", (owner, self.now()))
        if reclaimed.files or reclaimed.uploads or reclaimed.responses or reclaimed.conversations:
            problem = self._over(owner, adding, files)
        if problem is not None:
            raise problem

    def _enter(self, owner: str) -> None:
        """A write of `owner` begins. Called with `_guard` held."""
        self._busy[owner] = self._busy.get(owner, 0) + 1

    def _leave(self, owner: str) -> None:
        """A write of `owner` ended or was refused; wakes a deletion waiting for it."""
        with self._guard:
            self._busy[owner] -= 1
            if not self._busy[owner]:
                del self._busy[owner]
            self._guard.notify_all()

    def begin_write(self, owner: str) -> None:
        """A write for `owner` is admitted: refused for a hidden or deleted owner, else
        counted busy until `end_write`. Called before the body is read, so a deletion
        waits for a body still arriving, and none starts after the removal."""
        self.check_owner(owner)
        with self.database.lock:
            if self._is_removed(owner):
                raise _removed()
            with self._guard:
                self._enter(owner)

    def end_write(self, owner: str) -> None:
        self._leave(owner)

    def _wait_idle(self, owner: str) -> bool:
        deadline = time.perf_counter() + self.idle_wait_seconds
        with self._guard:
            while self._busy.get(owner):
                left = deadline - time.perf_counter()
                if left <= 0:
                    return False
                self._guard.wait(left)
        return True

    @contextlib.contextmanager
    def _reservation(self, owner: str, size: int) -> Iterator[None]:
        """Hold room for one file of `size` bytes while it is written. The write is
        busy for its owner until it ends."""
        with self.database.lock:
            self._check_quota(owner, size)
            with self._guard:
                self._reserved_bytes[owner] = self._reserved_bytes.get(owner, 0) + size
                self._reserved_files[owner] = self._reserved_files.get(owner, 0) + 1
                self._enter(owner)
        try:
            yield
        finally:
            with self._guard:
                self._reserved_bytes[owner] -= size
                self._reserved_files[owner] -= 1
                if not self._reserved_files[owner]:
                    del self._reserved_files[owner]
                    del self._reserved_bytes[owner]
            self._leave(owner)

    def generation(self) -> int:
        """The newest generation given out. The removal fence."""
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT value FROM counters WHERE name = 'generation'"
            ).fetchone()
        return int(row[0])

    @staticmethod
    def _next_generation(connection: sqlite3.Connection) -> int:
        connection.execute("UPDATE counters SET value = value + 1 WHERE name = 'generation'")
        return int(
            connection.execute("SELECT value FROM counters WHERE name = 'generation'").fetchone()[0]
        )

    # -- files -------------------------------------------------------------

    def create_file(
        self,
        owner: str,
        *,
        filename: str,
        purpose: str,
        mime_type: str | None,
        expires_after: int | None,
        chunks: Sequence[bytes],
    ) -> dict[str, Any]:
        self.check_owner(owner)
        self._files_on()
        name = self._filename(filename)
        purpose = self._purpose(purpose)
        seconds = self._seconds(expires_after)
        size = sum(len(chunk) for chunk in chunks)
        if size > MAX_BODY_BYTES:
            raise StoreProblem(413, "Too large", "A file sent whole is at most 16 MiB.")
        with self._reservation(owner, size):
            written = self.blobs.write_temp(chunks)
            try:
                with self.database.transaction() as connection:
                    row = self._insert_file(
                        connection, owner, name, purpose, mime_type, seconds, written
                    )
                    return self._file(row)
            finally:
                Blobs.discard(written.path)

    def _insert_file(
        self,
        connection: sqlite3.Connection,
        owner: str,
        name: str,
        purpose: str,
        mime_type: str | None,
        seconds: int | None,
        written: Written,
    ) -> sqlite3.Row:
        """The row, then the blob: a failure before the blob leaves no blob
        without a row. Under the lock, so no delete of the same bytes runs between."""
        if self._is_removed(owner):
            raise _removed()
        now = self.now()
        file_id = _new_id("file-")
        connection.execute(
            "INSERT INTO files (id, owner, filename, purpose, bytes, mime_type, sha256, "
            "created_at, expires_at, generation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_id,
                owner,
                name,
                purpose,
                written.size,
                (mime_type or DEFAULT_MIME)[:255],
                written.sha256,
                now,
                None if seconds is None else now + seconds,
                self._next_generation(connection),
            ),
        )
        if written.path.exists():
            self.blobs.keep(written)
        elif not self.blobs.path_for(written.sha256).exists():
            raise RuntimeError("the file's bytes are gone before they were kept")
        row = connection.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return row  # type: ignore[no-any-return]

    def _owned_file(self, owner: str, file_id: str) -> sqlite3.Row:
        self.check_owner(owner)
        if FILE_ID.match(file_id):
            row = self.database.connection.execute(
                f"SELECT * FROM files WHERE id = ? AND owner = ? AND {LIVE} AND {SHOWN}",
                (file_id, owner, self.now()),
            ).fetchone()
            if row is not None:
                return row  # type: ignore[no-any-return]
        raise not_found("file", file_id)

    def get_file(self, owner: str, file_id: str) -> dict[str, Any]:
        with self.database.lock:
            return self._file(self._owned_file(owner, file_id))

    def content(self, owner: str, file_id: str) -> tuple[dict[str, Any], Iterator[bytes]]:
        with self.database.lock:
            row = self._owned_file(owner, file_id)
            # Opened under the lock, so a delete cannot unlink it in between.
            handle = self.blobs.path_for(row["sha256"]).open("rb")

        def chunks() -> Iterator[bytes]:
            with handle:
                while chunk := handle.read(1024 * 1024):
                    yield chunk

        return self._file(row), chunks()

    def delete_file(self, owner: str, file_id: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            row = self._owned_file(owner, file_id)
            connection.execute("DELETE FROM files WHERE id = ?", (row["id"],))
            self._release_blobs(connection, [row["sha256"]])
        return {"id": row["id"], "deleted": True}

    def list_files(
        self,
        owner: str,
        *,
        purpose: str | None = None,
        limit: int = 10000,
        order: str = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> dict[str, Any]:
        self.check_owner(owner)
        if purpose is not None:
            self._purpose(purpose)
        if order not in ("asc", "desc"):
            raise StoreProblem(400, "Bad order", "order: asc or desc.")
        limit = max(1, min(int(limit), 10000))
        descending = order == "desc"
        clauses = ["owner = ?", LIVE, SHOWN]
        values: list[Any] = [owner, self.now()]
        if purpose is not None:
            clauses.append("purpose = ?")
            values.append(purpose)
        with self.database.lock:
            connection = self.database.connection
            for cursor, forward in ((after, True), (before, False)):
                if cursor is None:
                    continue
                anchor = self._owned_file(owner, cursor) if cursor else None
                if anchor is None:
                    continue
                later = forward != descending
                clauses.append(f"(created_at, id) {'>' if later else '<'} (?, ?)")
                values.extend([anchor["created_at"], anchor["id"]])
            direction = "DESC" if descending else "ASC"
            reverse = before is not None and after is None
            if reverse:
                direction = "ASC" if descending else "DESC"
            rows = connection.execute(
                f"SELECT * FROM files WHERE {' AND '.join(clauses)} "
                f"ORDER BY created_at {direction}, id {direction} LIMIT ?",
                (*values, limit + 1),
            ).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        if reverse:
            rows.reverse()
        return {"data": [self._file(row) for row in rows], "hasMore": more}

    # -- uploads -----------------------------------------------------------

    def _owned_upload(self, owner: str, upload_id: str, *, pending: bool) -> sqlite3.Row:
        self.check_owner(owner)
        if UPLOAD_ID.match(upload_id):
            row = self.database.connection.execute(
                f"SELECT * FROM uploads WHERE id = ? AND owner = ? AND {SHOWN}",
                (upload_id, owner),
            ).fetchone()
            if row is not None:
                if pending and row["status"] != "pending":
                    raise StoreProblem(
                        400,
                        "Upload not pending",
                        f"Upload {upload_id} is {row['status']}; start a new one.",
                    )
                if pending and row["expires_at"] <= self.now():
                    raise StoreProblem(
                        400,
                        "Upload ended",
                        f"Upload {upload_id} ended an hour after it started; start a new one.",
                    )
                return row  # type: ignore[no-any-return]
        raise not_found("upload", upload_id)

    def create_upload(
        self,
        owner: str,
        *,
        filename: str,
        purpose: str,
        size: int,
        mime_type: str | None,
        expires_after: int | None,
    ) -> dict[str, Any]:
        self.check_owner(owner)
        self._files_on()
        name = self._filename(filename)
        purpose = self._purpose(purpose)
        seconds = self._seconds(expires_after)
        if size < 1:
            raise StoreProblem(400, "Bad size", "bytes: the file's size, at least 1.")
        with self.database.lock:
            # The declared size counts from now, so no part can be written
            # past what the quota allows.
            self._check_quota(owner, size)
            with self.database.transaction() as connection:
                now = self.now()
                upload_id = _new_id("upload_")
                connection.execute(
                    "INSERT INTO uploads (id, owner, filename, purpose, bytes, mime_type, "
                    "created_at, expires_at, file_seconds, status, generation) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                    (
                        upload_id,
                        owner,
                        name,
                        purpose,
                        size,
                        (mime_type or DEFAULT_MIME)[:255],
                        now,
                        now + UPLOAD_SECONDS,
                        seconds,
                        self._next_generation(connection),
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM uploads WHERE id = ?", (upload_id,)
                ).fetchone()
                return self._upload(row)

    def add_part(self, owner: str, upload_id: str, chunks: Sequence[bytes]) -> dict[str, Any]:
        self.check_owner(owner)
        size = sum(len(chunk) for chunk in chunks)
        if size < 1:
            raise StoreProblem(400, "Empty part", "data: a part holds at least one byte.")
        if size > MAX_BODY_BYTES:
            raise StoreProblem(413, "Too large", "A part is at most 16 MiB.")
        with self.database.lock:
            # Room is taken before a byte is written: the parts kept, those on
            # their way, and this one, against the size the upload declared.
            upload = self._owned_upload(owner, upload_id, pending=True)
            if upload_id in self._assembling:
                raise StoreProblem(
                    409, "Upload completing", f"Upload {upload_id} is being completed."
                )
            total, count = self.database.connection.execute(
                "SELECT COALESCE(SUM(bytes), 0), COUNT(*) FROM parts WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            if count >= MAX_PARTS:
                raise StoreProblem(
                    400, "Too many parts", f"An upload has at most {MAX_PARTS} parts."
                )
            with self._guard:
                flying = self._part_bytes.get(upload_id, 0)
                if int(total) + flying + size > int(upload["bytes"]):
                    raise StoreProblem(
                        400,
                        "Parts too large",
                        f"The parts would be {int(total) + flying + size} bytes; the upload "
                        f"declared {upload['bytes']}.",
                    )
                self._part_bytes[upload_id] = flying + size
                self._part_owner[upload_id] = owner
                self._enter(owner)
        try:
            written = self.blobs.write_temp(chunks)
            try:
                with self.database.transaction() as connection:
                    self._owned_upload(owner, upload_id, pending=True)
                    part_id = _new_id("part_")
                    now = self.now()
                    connection.execute(
                        "INSERT INTO parts (id, upload_id, owner, bytes, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (part_id, upload_id, owner, written.size, now),
                    )
                    self.blobs.keep_part(written, part_id)
                return {"id": part_id, "uploadId": upload_id, "createdAt": now, "bytes": size}
            finally:
                Blobs.discard(written.path)
        finally:
            with self._guard:
                self._part_bytes[upload_id] -= size
                if not self._part_bytes[upload_id]:
                    del self._part_bytes[upload_id]
                    self._part_owner.pop(upload_id, None)
            self._leave(owner)

    def complete_upload(
        self,
        owner: str,
        upload_id: str,
        *,
        part_ids: list[str],
        md5: str | None = None,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        self.check_owner(owner)
        if not part_ids:
            raise StoreProblem(400, "No parts", "part_ids: name the parts, in order.")
        if len(set(part_ids)) != len(part_ids):
            raise StoreProblem(400, "Part named twice", "part_ids: name each part once.")
        with self.database.lock:
            upload = self._owned_upload(owner, upload_id, pending=True)
            with self._guard:
                if upload_id in self._assembling or self._part_bytes.get(upload_id):
                    raise StoreProblem(
                        409,
                        "Upload busy",
                        f"Upload {upload_id} is being completed or is taking a part; try "
                        "again when that has finished.",
                    )
                self._assembling.add(upload_id)
                self._enter(owner)
            rows = {
                row["id"]: row
                for row in self.database.connection.execute(
                    "SELECT * FROM parts WHERE upload_id = ? AND owner = ?", (upload_id, owner)
                )
            }
        try:
            return self._complete(owner, upload, rows, part_ids, md5, sha256)
        finally:
            with self._guard:
                self._assembling.discard(upload_id)
            self._leave(owner)

    def _complete(
        self,
        owner: str,
        upload: sqlite3.Row,
        rows: dict[str, sqlite3.Row],
        part_ids: list[str],
        md5: str | None,
        sha256: str | None,
    ) -> dict[str, Any]:
        upload_id = upload["id"]
        for part_id in part_ids:
            if part_id not in rows:
                raise not_found("part", part_id)
        total = sum(int(rows[p]["bytes"]) for p in part_ids)
        if total != int(upload["bytes"]):
            raise StoreProblem(
                400,
                "Size does not match",
                f"The parts named are {total} bytes; the upload declared {upload['bytes']}. "
                "Nothing was kept; the upload is still open.",
            )
        # Checked first, by reading the parts: a wrong digest writes nothing.
        digest = self.blobs.digest(self.blobs.read_part(p) for p in part_ids)
        if digest.size != total:
            raise StoreProblem(
                400, "Size does not match", f"The parts hold {digest.size} bytes; expected {total}."
            )
        for name, expected, actual in (
            ("md5", md5, digest.md5),
            ("sha256", sha256, digest.sha256),
        ):
            if expected is not None and expected.lower() != actual:
                raise StoreProblem(
                    400,
                    "Checksum does not match",
                    f"{name}: the joined file's is {actual}, not {expected.lower()}. "
                    "Nothing was kept; the upload is still open.",
                )
        written: Written | None = None
        try:
            while True:
                with self.database.lock:
                    if written is not None or self.blobs.path_for(digest.sha256).exists():
                        with self.database.transaction() as connection:
                            row = self._insert_file(
                                connection,
                                owner,
                                upload["filename"],
                                upload["purpose"],
                                upload["mime_type"],
                                upload["file_seconds"],
                                written
                                or Written(
                                    self.blobs.temp_path(), digest.size, digest.sha256, digest.md5
                                ),
                            )
                            connection.execute(
                                "UPDATE uploads SET status = 'completed', file_id = ? WHERE id = ?",
                                (row["id"], upload_id),
                            )
                            self._drop_parts(upload_id)
                            done = connection.execute(
                                "SELECT * FROM uploads WHERE id = ?", (upload_id,)
                            ).fetchone()
                            result = self._upload(done)
                            result["file"] = self._file(row)
                            return result
                # The bytes are not on disk yet: join the parts, deleting each
                # as it is copied, so the upload is never held twice.
                written = self._join(upload_id, part_ids)
                if written.sha256 != digest.sha256 or written.size != digest.size:
                    raise StoreProblem(
                        500, "Upload not joined", "The parts changed while they were joined."
                    )
        finally:
            if written is not None:
                Blobs.discard(written.path)

    def _join(self, upload_id: str, part_ids: list[str]) -> Written:
        def consumed() -> Iterator[bytes]:
            for part_id in part_ids:
                yield from self.blobs.read_part(part_id)
                self.blobs.unlink_part(part_id)

        try:
            return self.blobs.write_temp(consumed())
        except Exception as exc:
            # Some parts are gone: the upload cannot be completed again.
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE uploads SET status = 'cancelled' WHERE id = ?", (upload_id,)
                )
                self._drop_parts(upload_id)
            raise StoreProblem(
                500,
                "Upload not joined",
                f"Joining the parts failed ({exc}); start the upload again.",
            ) from exc

    def cancel_upload(self, owner: str, upload_id: str) -> dict[str, Any]:
        with self.database.lock:
            self._owned_upload(owner, upload_id, pending=True)
            with self._guard:
                if upload_id in self._assembling:
                    raise StoreProblem(
                        409, "Upload completing", f"Upload {upload_id} is being completed."
                    )
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE uploads SET status = 'cancelled' WHERE id = ?", (upload_id,)
                )
                self._drop_parts(upload_id)
                row = connection.execute(
                    "SELECT * FROM uploads WHERE id = ?", (upload_id,)
                ).fetchone()
                return self._upload(row)

    def _drop_parts(self, upload_id: str) -> int:
        connection = self.database.connection
        parts = [
            row["id"]
            for row in connection.execute("SELECT id FROM parts WHERE upload_id = ?", (upload_id,))
        ]
        connection.execute("DELETE FROM parts WHERE upload_id = ?", (upload_id,))
        for part_id in parts:
            self.blobs.unlink_part(part_id)
        return len(parts)

    # -- deletion ------------------------------------------------------------

    def _release_blobs(self, connection: sqlite3.Connection, digests: Iterable[str]) -> SweepCounts:
        """Unlink each blob no file names any more. Called under the lock."""
        counts = SweepCounts()
        for digest in set(digests):
            named = connection.execute(
                "SELECT 1 FROM files WHERE sha256 = ? LIMIT 1", (digest,)
            ).fetchone()
            if named is None:
                size = self.blobs.size_of(digest)
                if self.blobs.unlink(digest):
                    counts.blobs += 1
                    counts.freed_bytes += size
        return counts

    def _delete_where(self, where: str, values: tuple[Any, ...]) -> SweepCounts:
        counts = SweepCounts()
        with self.database.transaction() as connection:
            files = connection.execute(
                f"SELECT id, sha256 FROM files WHERE {where}", values
            ).fetchall()
            connection.execute(f"DELETE FROM files WHERE {where}", values)
            counts.files = len(files)
            uploads = connection.execute(f"SELECT id FROM uploads WHERE {where}", values).fetchall()
            with self._guard:
                busy = set(self._assembling)
            uploads = [u for u in uploads if u["id"] not in busy]
            for upload in uploads:
                counts.parts += self._drop_parts(upload["id"])
                connection.execute("DELETE FROM uploads WHERE id = ?", (upload["id"],))
            counts.uploads = len(uploads)
            counts.add(self._release_blobs(connection, [row["sha256"] for row in files]))
            # Row 54: stored responses, and conversations with their items, by the
            # same clause (both carry owner, generation and expires_at).
            counts.responses = connection.execute(
                f"DELETE FROM responses WHERE {where}", values
            ).rowcount
            ended = [
                row["id"]
                for row in connection.execute(f"SELECT id FROM conversations WHERE {where}", values)
            ]
            for conversation in ended:
                connection.execute(
                    "DELETE FROM conversation_items WHERE conversation_id = ?", (conversation,)
                )
            connection.execute(f"DELETE FROM conversations WHERE {where}", values)
            counts.conversations = len(ended)
        return counts

    def sweep(self) -> SweepCounts:
        """Delete what has ended, and anything on disk no row names."""
        now = self.now()
        # Row 54: a conversation's end follows the setting in effect now, from its
        # last use (none by default, as OpenAI keeps one); written first, so the
        # clause below deletes what has ended.
        seconds = self.config.conversation_seconds()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE conversations SET expires_at = CASE WHEN ? IS NULL THEN NULL "
                "ELSE used_at + ? END",
                (seconds, seconds),
            )
        counts = self._delete_where(ENDED, (now,))
        with self.database.lock:
            connection = self.database.connection
            named = {row[0] for row in connection.execute("SELECT DISTINCT sha256 FROM files")}
            for digest in self.blobs.blob_names():
                if digest not in named:
                    size = self.blobs.size_of(digest)
                    if self.blobs.unlink(digest):
                        counts.blobs += 1
                        counts.freed_bytes += size
            parts = {row[0] for row in connection.execute("SELECT id FROM parts")}
            for part_id in self.blobs.part_names():
                if part_id not in parts and self.blobs.unlink_part(part_id):
                    counts.parts += 1
        for stale in self.blobs.stale_temps():
            Blobs.discard(stale)
        return counts

    def delete_removed(
        self,
        owner: str,
        *,
        marker: int | None = None,
        force: bool = False,
        fence: int | None = None,
    ) -> SweepCounts | None:
        """Delete a removed person's objects: rows, blobs and every `.bak` copy.

        Compare-and-delete: only the marker the caller saw (`marker`), and only when
        its date has passed (or `force`, the owner's Delete now); None when the
        marker is gone, replaced or not yet due. Waits for the owner's writes under
        way (they refuse at commit, the owner being hidden). The marker is the retry
        record and goes last: a copy that cannot be purged raises, naming it, and
        the marker stays for the next sweep. What was kept after `fence` stays."""
        self.check_owner(owner)
        now = self.now()
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM removed WHERE owner = ?", (owner,)).fetchone()
            if row is None:
                if marker is None:
                    raise StoreProblem(
                        404, "Not removed", f"{owner} is not a person Eugene removed."
                    )
                return None
            if row["state"] == "deleted":
                if marker is None:
                    raise StoreProblem(
                        404, "Not removed", f"{owner}'s files here are already deleted."
                    )
                return None
            if marker is not None and row["marker"] != marker:
                return None
            if force:
                connection.execute(
                    "UPDATE removed SET delete_at = MIN(delete_at, ?) WHERE owner = ?", (now, owner)
                )
            elif row["delete_at"] > now:
                return None
            marker = int(row["marker"])
        if not self._wait_idle(owner):
            raise StoreProblem(
                503,
                "Writes under way",
                f"{owner} still has an upload being written here. Nothing was deleted; "
                "the next sweep tries again.",
            )
        if fence is None:
            fence = self.generation()
        counts = self._delete_where("owner = ? AND generation <= ?", (owner, fence))
        try:
            self.database.purge_backups([owner])
        except BackupPurgeFailed as exc:
            raise StoreProblem(
                503,
                "Copy not purged",
                f"{owner}'s files here are deleted, but a copy kept before a migration could "
                f"not be purged: {exc}. They stay removed, and the next sweep tries again.",
            ) from exc
        with self.database.transaction() as connection:
            left = connection.execute(
                "SELECT 1 FROM files WHERE owner = ? UNION ALL "
                "SELECT 1 FROM uploads WHERE owner = ? UNION ALL "
                "SELECT 1 FROM responses WHERE owner = ? UNION ALL "
                "SELECT 1 FROM conversations WHERE owner = ? LIMIT 1",
                (owner, owner, owner, owner),
            ).fetchone()
            if left is not None:
                # Something newer than the fence survived: the person is back (a stale
                # answer deleted them), so there is nothing to keep tombstoned.
                connection.execute(
                    "DELETE FROM removed WHERE owner = ? AND marker = ?", (owner, marker)
                )
            else:
                connection.execute(
                    "UPDATE removed SET state = 'deleted', delete_at = ? "
                    "WHERE owner = ? AND marker = ?",
                    (self.now(), owner, marker),
                )
        return counts

    def delete_now(self, owner: str) -> SweepCounts:
        """The owner's Delete now for a removed person: at once and for good, every
        copy (MRC27). Only a removed person: a person Eugene still has keeps their
        files (404)."""
        counts = self.delete_removed(owner, force=True)
        assert counts is not None
        return counts

    # -- people and the console ----------------------------------------------

    def person_owners(self) -> list[str]:
        """Every person who owns something here, the hidden ones included."""
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT owner FROM files WHERE owner LIKE 'person:%' "
                "UNION SELECT owner FROM uploads WHERE owner LIKE 'person:%' "
                "UNION SELECT owner FROM responses WHERE owner LIKE 'person:%' "
                "UNION SELECT owner FROM conversations WHERE owner LIKE 'person:%' "
                "UNION SELECT owner FROM removed WHERE state = 'hidden' ORDER BY 1"
            ).fetchall()
        return [row[0] for row in rows]

    def _is_removed(self, owner: str) -> bool:
        found = self.database.connection.execute(
            "SELECT 1 FROM removed WHERE owner = ?", (owner,)
        ).fetchone()
        return found is not None

    def removed_people(self) -> dict[str, dict[str, int]]:
        """Each hidden owner: when it was found removed, its delete date and its marker."""
        with self.database.lock:
            return {
                row["owner"]: {
                    "removedAt": row["found_at"],
                    "deleteAt": row["delete_at"],
                    "marker": row["marker"],
                }
                for row in self.database.connection.execute(
                    "SELECT * FROM removed WHERE state = 'hidden'"
                )
            }

    def hide(self, owners: list[str], seconds: int) -> list[str]:
        """Hide these removed people now. Each one's date is fixed here, from the
        setting now: a later change never shortens it. The marker carries the
        generation it was made at, for compare-and-delete."""
        now = self.now()
        hidden = []
        with self.database.transaction() as connection:
            for owner in owners:
                self.check_owner(owner)
                if self._is_removed(owner):
                    continue
                connection.execute(
                    "INSERT INTO removed (owner, found_at, delete_at, marker) VALUES (?, ?, ?, ?)",
                    (owner, now, now + seconds, self._next_generation(connection)),
                )
                hidden.append(owner)
        return hidden

    def restore(self, owners: list[str]) -> list[str]:
        """The root has these people again: what they kept is theirs again, unless
        their date has passed; a date that has passed is final."""
        now = self.now()
        restored = []
        with self.database.transaction() as connection:
            for owner in owners:
                cursor = connection.execute(
                    "DELETE FROM removed WHERE owner = ? AND delete_at > ? AND state = 'hidden'",
                    (owner, now),
                )
                if cursor.rowcount:
                    restored.append(owner)
        return restored

    def owners(self) -> dict[str, Any]:
        found: dict[str, dict[str, Any]] = {}

        def entry(owner: str) -> dict[str, Any]:
            return found.setdefault(
                owner,
                {
                    "owner": owner,
                    "files": 0,
                    "bytes": 0,
                    "uploads": 0,
                    "responses": 0,
                    "conversations": 0,
                },
            )

        with self.database.lock:
            connection = self.database.connection
            for row in connection.execute(
                "SELECT owner, COUNT(*), COALESCE(SUM(bytes), 0) FROM files GROUP BY owner"
            ):
                entry(row[0]).update(files=row[1], bytes=row[2])
            # Each kind counted the same way: never a name, a purpose or content.
            for table, count in (
                ("uploads WHERE status = 'pending'", "uploads"),
                ("responses", "responses"),
                ("conversations", "conversations"),
            ):
                for row in connection.execute(
                    f"SELECT owner, COUNT(*), COALESCE(SUM(bytes), 0) FROM {table} GROUP BY owner"
                ):
                    entry(row[0])[count] = row[1]
                    entry(row[0])["bytes"] += row[2]
            blob_bytes = connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) FROM "
                "(SELECT sha256, MAX(bytes) AS bytes FROM files GROUP BY sha256)"
            ).fetchone()[0]
            for row in connection.execute("SELECT * FROM removed WHERE state = 'hidden'"):
                if row["owner"] in found:
                    found[row["owner"]]["removedAt"] = row["found_at"]
                    found[row["owner"]]["deleteAt"] = row["delete_at"]
        owners = sorted(found.values(), key=lambda entry: (-entry["bytes"], entry["owner"]))
        return {
            "owners": owners,
            "files": sum(entry["files"] for entry in owners),
            "bytes": sum(entry["bytes"] for entry in owners),
            "blobBytes": int(blob_bytes),
        }

    def save_report(self, report: dict[str, Any]) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO sweeps (id, report) VALUES (1, ?) "
                "ON CONFLICT (id) DO UPDATE SET report = excluded.report",
                (json.dumps(report),),
            )

    def last_report(self) -> dict[str, Any] | None:
        with self.database.lock:
            row = self.database.connection.execute(
                "SELECT report FROM sweeps WHERE id = 1"
            ).fetchone()
        return json.loads(row[0]) if row is not None else None


def report_of(counts: SweepCounts, *, at: float, removal: dict[str, Any] | None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "at": datetime.fromtimestamp(at, UTC).isoformat().replace("+00:00", "Z"),
        "files": counts.files,
        "uploads": counts.uploads,
        "parts": counts.parts,
        "blobs": counts.blobs,
        "freedBytes": counts.freed_bytes,
        "responses": counts.responses,
        "conversations": counts.conversations,
    }
    if removal is not None:
        report["removal"] = removal
    return report
