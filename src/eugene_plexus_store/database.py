"""The store's SQLite database: one file, migrated in place, a copy kept first.

`secure_delete` is on for every connection, so a deleted row's bytes are
overwritten with zeros in the file rather than left in a free page. The
journal is SQLite's default rollback journal, not WAL: a WAL file keeps
old pages until a checkpoint, and a deleted person's rows must not
outlive the delete in a file beside the database.

**A copy before each migration** (as Workbench does): before a database
at schema N is moved to N+1 it is copied to `store.sqlite3.schema-N.bak`,
the way back if the new code is wrong. Removing a person removes them
from every copy too (`purge_backups`), so a copy is a way back for
everyone else and for nobody who was removed.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

log = logging.getLogger(__name__)

FILENAME = "store.sqlite3"
#: How long a purge waits on a copy another process holds locked, in seconds.
BACKUP_TIMEOUT = 5.0


class BackupPurgeFailed(Exception):
    """A copy kept before a migration could not be purged of a person's rows.
    `failures` names each file and why."""

    def __init__(self, failures: list[tuple[Path, str]]) -> None:
        self.failures = failures
        super().__init__("; ".join(f"{path.name}: {why}" for path, why in failures))


def _run(connection: sqlite3.Connection, script: str) -> None:
    """Each statement on its own: `executescript` would commit the
    migration's transaction before running, leaving a half-made schema
    possible after a crash."""
    for statement in script.split(";"):
        if statement.strip():
            connection.execute(statement)


def _schema_1(connection: sqlite3.Connection) -> None:
    _run(
        connection,
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            filename TEXT NOT NULL,
            purpose TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER,
            generation INTEGER NOT NULL
        );
        CREATE INDEX files_by_owner ON files(owner, created_at, id);
        CREATE INDEX files_by_end ON files(expires_at);
        CREATE INDEX files_by_blob ON files(sha256);
        CREATE TABLE uploads (
            id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            filename TEXT NOT NULL,
            purpose TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            file_seconds INTEGER,
            status TEXT NOT NULL,
            file_id TEXT,
            generation INTEGER NOT NULL
        );
        CREATE TABLE counters (
            name TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        );
        INSERT INTO counters (name, value) VALUES ('generation', 0);
        CREATE INDEX uploads_by_owner ON uploads(owner);
        CREATE INDEX uploads_by_end ON uploads(expires_at);
        CREATE TABLE parts (
            id TEXT PRIMARY KEY,
            upload_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX parts_by_upload ON parts(upload_id);
        CREATE TABLE removed (
            owner TEXT PRIMARY KEY,
            found_at INTEGER NOT NULL,
            delete_at INTEGER NOT NULL,
            marker INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'hidden' CHECK (state IN ('hidden', 'deleted'))
        );
        CREATE TABLE sweeps (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            report TEXT NOT NULL
        );
        """,
    )


def _schema_2(connection: sqlite3.Connection) -> None:
    """Row 54: stored responses, conversations and their items.

    A response keeps the items its own request sent (`input`), what it made
    (`output`) and the response object without its output (`body`), as JSON;
    a chain is its `previous_id` links, whose first is `root_id`. A
    conversation's items are rows in order (`position`). Every table carries
    `owner` and the store's `generation`; responses and conversations an
    `expires_at`, so row 53's sweep, removal and quota clauses read them as
    they read files. A conversation's `used_at` is its last use: its end,
    when the owner sets one (`conversationDays`), is counted from it. A
    response's `item_bytes` is its input and output alone, so a chain's size is
    summed before any of it is read."""
    _run(
        connection,
        """
        CREATE TABLE responses (
            id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL,
            previous_id TEXT,
            root_id TEXT NOT NULL,
            depth INTEGER NOT NULL,
            conversation_id TEXT,
            conversation_position INTEGER,
            input TEXT NOT NULL,
            output TEXT NOT NULL,
            body TEXT NOT NULL,
            instructions INTEGER NOT NULL,
            item_bytes INTEGER NOT NULL,
            bytes INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER,
            generation INTEGER NOT NULL
        );
        CREATE INDEX responses_by_owner ON responses(owner);
        CREATE INDEX responses_by_end ON responses(expires_at);
        CREATE INDEX responses_by_previous ON responses(previous_id);
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            metadata TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            next_position INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            used_at INTEGER NOT NULL,
            expires_at INTEGER,
            generation INTEGER NOT NULL
        );
        CREATE INDEX conversations_by_owner ON conversations(owner);
        CREATE INDEX conversations_by_end ON conversations(expires_at);
        CREATE TABLE conversation_items (
            conversation_id TEXT NOT NULL,
            id TEXT NOT NULL,
            owner TEXT NOT NULL,
            position INTEGER NOT NULL,
            item TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (conversation_id, id)
        );
        CREATE INDEX conversation_items_in_order ON conversation_items(conversation_id, position);
        CREATE INDEX conversation_items_by_owner ON conversation_items(owner)
        """,
    )


#: Schema N is made by MIGRATIONS[N - 1]. Append; never edit one that shipped.
MIGRATIONS: Sequence[Callable[[sqlite3.Connection], None]] = (_schema_1, _schema_2)

#: The tables that hold an owner's rows, for removal: from the database
#: and from every `.bak` copy.
OWNER_TABLES = (
    "files",
    "uploads",
    "parts",
    "responses",
    "conversations",
    "conversation_items",
    "removed",
)


def _connect(path: Path, timeout: float = 5.0) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path, check_same_thread=False, isolation_level=None, timeout=timeout
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA secure_delete = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = DELETE")
    return connection


class Database:
    """One connection, used under one lock: the store is one process and its
    writes are small, so serialising them costs nothing and rules out every
    interleaving of a delete with an upload of the same content."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.path = folder / FILENAME
        self.lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def open(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        connection = _connect(self.path)
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > len(MIGRATIONS):
            connection.close()
            raise RuntimeError(
                f"{self.path} is at schema {version}, newer than this store knows "
                f"({len(MIGRATIONS)}): it was written by a newer Eugene. Update this machine."
            )
        if existed and 0 < version < len(MIGRATIONS):
            backup = self.path.with_name(f"{FILENAME}.schema-{version}.bak")
            connection.close()
            shutil.copy2(self.path, backup)
            log.info("copied the store's database to %s before migrating it", backup)
            connection = _connect(self.path)
        for number in range(version + 1, len(MIGRATIONS) + 1):
            with _transaction(connection):
                MIGRATIONS[number - 1](connection)
                connection.execute(f"PRAGMA user_version = {number}")
        self._connection = connection

    def close(self) -> None:
        with self.lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("the store's database is not open")
        return self._connection

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock, _transaction(self.connection):
            yield self.connection

    def backups(self) -> list[Path]:
        return sorted(self.folder.glob(f"{FILENAME}.schema-*.bak"))

    def purge_backups(self, owners: Sequence[str]) -> None:
        """Remove `owners` from every copy kept before a migration. Every copy is
        tried; one that cannot be purged is named in `BackupPurgeFailed`, raised
        after the rest, so the caller keeps its retry record."""
        failures: list[tuple[Path, str]] = []
        for backup in self.backups():
            try:
                self._purge_one(backup, owners)
            except (sqlite3.Error, OSError) as exc:
                failures.append((backup, str(exc) or type(exc).__name__))
        if failures:
            raise BackupPurgeFailed(failures)

    @staticmethod
    def _purge_one(backup: Path, owners: Sequence[str]) -> None:
        connection = _connect(backup, BACKUP_TIMEOUT)
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            with _transaction(connection):
                for table in OWNER_TABLES:
                    if table in tables:
                        connection.executemany(
                            f"DELETE FROM {table} WHERE owner = ?",
                            [(owner,) for owner in owners],
                        )
            connection.execute("VACUUM")
        finally:
            connection.close()


@contextlib.contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")
