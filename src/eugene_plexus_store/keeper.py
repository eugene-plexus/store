"""Files and uploads, each kept for one owner until its end.

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
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .blobs import Blobs, Written
from .config import GIB, ConfigStore
from .database import OWNER_TABLES, Database
from .problems import StoreProblem, not_found

log = logging.getLogger(__name__)

OWNER = re.compile(r"^(install|(person|key|component):[A-Za-z0-9._@:-]{1,200})$")
FILE_ID = re.compile(r"^file-[0-9a-f]{24}$")
UPLOAD_ID = re.compile(r"^upload_[0-9a-f]{24}$")
PART_ID = re.compile(r"^part_[0-9a-f]{24}$")
PURPOSES = ("assistants", "batch", "fine-tune", "vision", "user_data", "evals")
#: OpenAI's: an upload not completed within an hour is deleted with its parts.
UPLOAD_SECONDS = 3600
#: The gateway's body limit, and so the most one part or one file may be.
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_PARTS = 10000
DEFAULT_MIME = "application/octet-stream"


def _new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(12)


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
    unlink_later: list[str] = field(default_factory=list)

    def add(self, other: SweepCounts) -> None:
        self.files += other.files
        self.uploads += other.uploads
        self.parts += other.parts
        self.blobs += other.blobs
        self.freed_bytes += other.freed_bytes


class Keeper:
    def __init__(
        self, folder: Path, config: ConfigStore, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.folder = folder
        self.config = config
        self.clock = clock
        self.database = Database(folder)
        self.blobs = Blobs(folder)

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
        return {
            "id": row["id"],
            "filename": row["filename"],
            "purpose": row["purpose"],
            "bytes": row["bytes"],
            "mimeType": row["mime_type"],
            "sha256": row["sha256"],
            "createdAt": row["created_at"],
            "expiresAt": row["expires_at"],
            "status": "processed",
        }

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

    def _seconds(self, expires_after: int | None) -> int:
        if expires_after is None:
            return self.config.default_seconds()
        if expires_after < 1:
            raise StoreProblem(400, "Bad end", "expires_after: must be at least 1 second.")
        most = self.config.max_seconds()
        if expires_after > most:
            raise StoreProblem(
                400,
                "Bad end",
                f"expires_after: at most {most} seconds ({most // 86400} days) here; the "
                "owner sets the longest in Settings -> Store.",
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

    def _used(self, connection: sqlite3.Connection, owner: str, now: int) -> tuple[int, int]:
        """(files, bytes) the owner holds: kept files, and uploads in progress."""
        files = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM files "
            "WHERE owner = ? AND expires_at > ?",
            (owner, now),
        ).fetchone()
        pending = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM uploads "
            "WHERE owner = ? AND status = 'pending' AND expires_at > ?",
            (owner, now),
        ).fetchone()
        return int(files[0]) + int(pending[0]), int(files[1]) + int(pending[1])

    def _quota(
        self,
        connection: sqlite3.Connection,
        owner: str,
        adding: int,
        now: int,
        *,
        except_upload: int = 0,
    ) -> None:
        count, used = self._used(connection, owner, now)
        count -= 1 if except_upload else 0
        used -= except_upload
        most_files = self.config.quota_files()
        most_bytes = self.config.quota_bytes()
        if count + 1 > most_files:
            raise StoreProblem(
                413,
                "Over quota",
                f"This owner keeps {count} file{'' if count == 1 else 's'}, and the most it "
                f"may keep is {most_files} (Settings -> Store). Delete files it no longer needs.",
            )
        if used + adding > most_bytes:
            raise StoreProblem(
                413,
                "Over quota",
                f"This owner keeps {_human(used)} ({used} bytes); this adds {_human(adding)} "
                f"({adding} bytes), and the most it may keep is {_human(most_bytes)} "
                f"({most_bytes} bytes) (Settings -> Store). Delete files it no longer needs.",
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
        chunks: Iterable[bytes],
    ) -> dict[str, Any]:
        self.check_owner(owner)
        self._files_on()
        name = self._filename(filename)
        purpose = self._purpose(purpose)
        seconds = self._seconds(expires_after)
        written = self.blobs.write_temp(chunks)
        try:
            return self._keep(owner, name, purpose, mime_type, seconds, written)
        finally:
            Blobs.discard(written.path)

    def _keep(
        self,
        owner: str,
        name: str,
        purpose: str,
        mime_type: str | None,
        seconds: int,
        written: Written,
        *,
        upload: str | None = None,
    ) -> dict[str, Any]:
        with self.database.transaction() as connection:
            now = self.now()
            # An upload's declared size already counts; it is not counted twice.
            self._quota(
                connection, owner, written.size, now, except_upload=written.size if upload else 0
            )
            file_id = _new_id("file-")
            connection.execute(
                "INSERT INTO files (id, owner, filename, purpose, bytes, mime_type, sha256, "
                "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    owner,
                    name,
                    purpose,
                    written.size,
                    (mime_type or DEFAULT_MIME)[:255],
                    written.sha256,
                    now,
                    now + seconds,
                ),
            )
            if upload is not None:
                connection.execute(
                    "UPDATE uploads SET status = 'completed', file_id = ? WHERE id = ?",
                    (file_id, upload),
                )
                self._drop_parts(upload)
            # Last, so a failure above leaves no blob without a row. Under the
            # lock, so no delete of the same bytes runs between.
            self.blobs.keep(written)
            row = connection.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
            return self._file(row)

    def _owned_file(self, owner: str, file_id: str) -> sqlite3.Row:
        self.check_owner(owner)
        if FILE_ID.match(file_id):
            row = self.database.connection.execute(
                "SELECT * FROM files WHERE id = ? AND owner = ? AND expires_at > ?",
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
        clauses = ["owner = ?", "expires_at > ?"]
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
                "SELECT * FROM uploads WHERE id = ? AND owner = ?", (upload_id, owner)
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
        with self.database.transaction() as connection:
            now = self.now()
            self._quota(connection, owner, size, now)
            upload_id = _new_id("upload_")
            connection.execute(
                "INSERT INTO uploads (id, owner, filename, purpose, bytes, mime_type, "
                "created_at, expires_at, file_seconds, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
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
                ),
            )
            row = connection.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
            return self._upload(row)

    def add_part(self, owner: str, upload_id: str, chunks: Iterable[bytes]) -> dict[str, Any]:
        self.check_owner(owner)
        with self.database.lock:
            self._owned_upload(owner, upload_id, pending=True)
        written = self.blobs.write_temp(chunks)
        try:
            if written.size < 1:
                raise StoreProblem(400, "Empty part", "data: a part holds at least one byte.")
            with self.database.transaction() as connection:
                upload = self._owned_upload(owner, upload_id, pending=True)
                total, count = connection.execute(
                    "SELECT COALESCE(SUM(bytes), 0), COUNT(*) FROM parts WHERE upload_id = ?",
                    (upload_id,),
                ).fetchone()
                if count >= MAX_PARTS:
                    raise StoreProblem(
                        400, "Too many parts", f"An upload has at most {MAX_PARTS} parts."
                    )
                if int(total) + written.size > int(upload["bytes"]):
                    raise StoreProblem(
                        400,
                        "Parts too large",
                        f"The parts would be {int(total) + written.size} bytes; the upload "
                        f"declared {upload['bytes']}.",
                    )
                part_id = _new_id("part_")
                now = self.now()
                connection.execute(
                    "INSERT INTO parts (id, upload_id, owner, bytes, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (part_id, upload_id, owner, written.size, now),
                )
                self.blobs.keep_part(written, part_id)
            return {"id": part_id, "uploadId": upload_id, "createdAt": now, "bytes": written.size}
        finally:
            Blobs.discard(written.path)

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
            rows = {
                row["id"]: row
                for row in self.database.connection.execute(
                    "SELECT * FROM parts WHERE upload_id = ? AND owner = ?", (upload_id, owner)
                )
            }
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

        def joined() -> Iterator[bytes]:
            for part_id in part_ids:
                yield from self.blobs.read_part(part_id)

        written = self.blobs.write_temp(joined())
        try:
            if written.size != total:
                raise StoreProblem(
                    400,
                    "Size does not match",
                    f"The parts joined to {written.size} bytes; expected {total}.",
                )
            for name, expected, actual in (
                ("md5", md5, written.md5),
                ("sha256", sha256, written.sha256),
            ):
                if expected is not None and expected.lower() != actual:
                    raise StoreProblem(
                        400,
                        "Checksum does not match",
                        f"{name}: the joined file's is {actual}, not {expected.lower()}. "
                        "Nothing was kept; the upload is still open.",
                    )
            with self.database.lock:
                upload = self._owned_upload(owner, upload_id, pending=True)
                kept = self._keep(
                    owner,
                    upload["filename"],
                    upload["purpose"],
                    upload["mime_type"],
                    int(upload["file_seconds"]),
                    written,
                    upload=upload_id,
                )
                row = self.database.connection.execute(
                    "SELECT * FROM uploads WHERE id = ?", (upload_id,)
                ).fetchone()
                result = self._upload(row)
                result["file"] = kept
                return result
        finally:
            Blobs.discard(written.path)

    def cancel_upload(self, owner: str, upload_id: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            self._owned_upload(owner, upload_id, pending=True)
            connection.execute("UPDATE uploads SET status = 'cancelled' WHERE id = ?", (upload_id,))
            self._drop_parts(upload_id)
            row = connection.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
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
            for upload in uploads:
                counts.parts += self._drop_parts(upload["id"])
            connection.execute(f"DELETE FROM uploads WHERE {where}", values)
            counts.uploads = len(uploads)
            counts.add(self._release_blobs(connection, [row["sha256"] for row in files]))
        return counts

    def sweep(self) -> SweepCounts:
        """Delete what has ended, and anything on disk no row names."""
        now = self.now()
        counts = self._delete_where("expires_at <= ?", (now,))
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

    def remove_owners(self, owners: list[str]) -> SweepCounts:
        """Delete everything these owners keep, here and in every `.bak` copy."""
        counts = SweepCounts()
        if not owners:
            return counts
        for owner in owners:
            self.check_owner(owner)
            counts.add(self._delete_where("owner = ?", (owner,)))
        with self.database.transaction() as connection:
            for table in OWNER_TABLES:
                connection.executemany(
                    f"DELETE FROM {table} WHERE owner = ?", [(owner,) for owner in owners]
                )
        self.database.purge_backups(owners)
        return counts

    # -- people and the console ----------------------------------------------

    def person_owners(self) -> list[str]:
        with self.database.lock:
            rows = self.database.connection.execute(
                "SELECT owner FROM files WHERE owner LIKE 'person:%' "
                "UNION SELECT owner FROM uploads WHERE owner LIKE 'person:%' ORDER BY 1"
            ).fetchall()
        return [row[0] for row in rows]

    def held(self) -> list[str]:
        with self.database.lock:
            return [
                row[0]
                for row in self.database.connection.execute("SELECT owner FROM held ORDER BY owner")
            ]

    def hold(self, owners: list[str]) -> None:
        with self.database.transaction() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO held (owner, found_at) VALUES (?, ?)",
                [(owner, self.now()) for owner in owners],
            )

    def unhold(self, owners: list[str]) -> None:
        with self.database.transaction() as connection:
            connection.executemany(
                "DELETE FROM held WHERE owner = ?", [(owner,) for owner in owners]
            )

    def owners(self) -> dict[str, Any]:
        now = self.now()
        found: dict[str, dict[str, Any]] = {}
        with self.database.lock:
            connection = self.database.connection
            for row in connection.execute(
                "SELECT owner, COUNT(*), COALESCE(SUM(bytes), 0) FROM files "
                "WHERE expires_at > ? GROUP BY owner",
                (now,),
            ):
                found[row[0]] = {"owner": row[0], "files": row[1], "bytes": row[2], "uploads": 0}
            for row in connection.execute(
                "SELECT owner, COUNT(*), COALESCE(SUM(bytes), 0) FROM uploads "
                "WHERE status = 'pending' AND expires_at > ? GROUP BY owner",
                (now,),
            ):
                entry = found.setdefault(
                    row[0], {"owner": row[0], "files": 0, "bytes": 0, "uploads": 0}
                )
                entry["uploads"] = row[1]
                entry["bytes"] += row[2]
            blob_bytes = connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) FROM "
                "(SELECT sha256, MAX(bytes) AS bytes FROM files GROUP BY sha256)"
            ).fetchone()[0]
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
    }
    if removal is not None:
        report["removal"] = removal
    return report
