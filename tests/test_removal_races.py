"""The second review's removal and quota races (row 53).

A removed person's objects must never come back from a write that was already
under way, a copy that could not be purged must be retried, an overdue deletion
is final, and a stale answer from the root must not delete a restored person's
files. Writes are paused inside the blob layer to make each interleaving
certain, not likely.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_store import blobs as blobs_module
from eugene_plexus_store import database as database_module
from eugene_plexus_store.config import ConfigStore
from eugene_plexus_store.keeper import Keeper
from eugene_plexus_store.problems import StoreProblem
from eugene_plexus_store.settings import Settings

from .conftest import AgentStub, Caller, Clock, blob_files

DAY = 24 * 3600
ALICE = "person:alice"


@pytest.fixture
def keeper(client: TestClient) -> Keeper:
    found: Keeper = client.app.state.keeper  # type: ignore[attr-defined]
    return found


def _staged(settings: Settings) -> list[str]:
    """Anything left half-written: temporary files and unjoined parts."""
    folder = settings.data_folder()
    names: list[str] = []
    for sub in ("tmp", "parts"):
        root = folder / sub
        if root.is_dir():
            names += [str(p) for p in root.rglob("*") if p.is_file()]
    return names


def _run(target: Callable[[], Any], results: dict[str, Any], name: str) -> threading.Thread:
    def go() -> None:
        try:
            results[name] = target()
        except Exception as exc:  # recorded for the assertions
            results[name] = exc

    thread = threading.Thread(target=go, name=name)
    thread.start()
    return thread


def _create(keeper: Keeper, content: bytes = b"in flight") -> dict[str, Any]:
    return keeper.create_file(
        ALICE,
        filename="a.txt",
        purpose="user_data",
        mime_type=None,
        expires_after=None,
        chunks=[content],
    )


# --- 2. cancelling an upload while a part is being written -------------------------------


def test_cancelling_an_upload_does_not_free_the_bytes_a_part_is_still_staging(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    keeper: Keeper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ConfigStore, "quota_bytes", lambda self: 100)
    alice = as_owner(ALICE)
    upload = alice.post("/v1/uploads", json={"filename": "a", "purpose": "batch", "bytes": 80})
    assert upload.status_code == 201, upload.text
    upload_id = upload.json()["id"]

    inside, release = threading.Event(), threading.Event()
    original = blobs_module.Blobs.write_temp

    def held(self: Any, chunks: Any) -> Any:
        inside.set()
        assert release.wait(10)
        return original(self, chunks)

    monkeypatch.setattr(blobs_module.Blobs, "write_temp", held)
    results: dict[str, Any] = {}
    writer = _run(lambda: keeper.add_part(ALICE, upload_id, [b"p" * 80]), results, "part")
    assert inside.wait(10)

    cancelled = alice.post(f"/v1/uploads/{upload_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    # The part's bytes are still being staged: they count until they are gone.
    assert keeper._used(ALICE)[1] >= 80
    monkeypatch.setattr(blobs_module.Blobs, "write_temp", original)
    refused = alice.upload(b"z" * 80)
    assert refused.status_code == 413, refused.text

    release.set()
    writer.join(10)
    assert isinstance(results["part"], StoreProblem), results["part"]
    # The writer let go of its room when it ended: nothing is counted, nothing is staged.
    assert keeper._used(ALICE) == (0, 0)
    assert alice.upload(b"z" * 80).status_code == 201


# --- 3 and 5. the date, a stale answer, a late restoration -------------------------------


def test_a_stale_unknown_answer_does_not_delete_a_person_restored_before_the_date(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    keeper: Keeper,
    clock: Clock,
    settings: Settings,
) -> None:
    alice = as_owner(ALICE)
    kept = alice.upload(b"alice's own words").json()["id"]
    agent.removed = {"alice"}
    client.post("/v1/sweep", headers=operator)
    assert alice.get(f"/v1/files/{kept}").status_code == 404

    clock.advance(29 * DAY)
    answer = agent.handler

    def restored_then_late(request: httpx.Request) -> httpx.Response:
        # The root answers "unknown" for a question asked before the date. By the time
        # the answer is used the person was restored (before the date) and the date passed.
        response = answer(request)
        keeper.restore([ALICE])
        clock.advance(2 * DAY)
        return response

    client.app.state.sweeper.people._client._transport = httpx.MockTransport(restored_then_late)  # type: ignore[attr-defined]
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["removal"]["deleted"] == 0, report
    assert alice.get(f"/v1/files/{kept}").status_code == 200
    assert len(blob_files(settings)) == 1


def test_a_restoration_after_the_date_does_not_cancel_the_deletion(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    clock: Clock,
    settings: Settings,
) -> None:
    alice = as_owner(ALICE)
    kept = alice.upload(b"overdue").json()["id"]
    agent.removed = {"alice"}
    client.post("/v1/sweep", headers=operator)

    clock.advance(31 * DAY)
    agent.removed = set()  # the root has her again, but the date has passed
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["removal"]["deleted"] == 1, report
    assert report["removal"]["restored"] == 0, report
    assert report["removal"]["lateRestored"] == 1, report
    assert alice.get(f"/v1/files/{kept}").status_code == 404
    assert blob_files(settings) == []


# --- 4. Delete now never lets a write already under way come back ------------------------


def test_delete_now_waits_for_a_paused_write_and_the_write_leaves_nothing(
    client: TestClient,
    operator: dict[str, str],
    as_owner: Callable[[str], Caller],
    keeper: Keeper,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inside, release = threading.Event(), threading.Event()
    original = blobs_module.Blobs.write_temp

    def held(self: Any, chunks: Any) -> Any:
        inside.set()
        assert release.wait(10)
        return original(self, chunks)

    monkeypatch.setattr(blobs_module.Blobs, "write_temp", held)
    results: dict[str, Any] = {}
    writer = _run(lambda: _create(keeper), results, "write")
    assert inside.wait(10)
    keeper.hide([ALICE], 30 * DAY)
    deleter = _run(lambda: keeper.delete_now(ALICE), results, "delete")
    time.sleep(0.3)
    assert deleter.is_alive(), "Delete now returned while a write of that person was under way"
    release.set()
    writer.join(10)
    deleter.join(10)

    assert isinstance(results["write"], StoreProblem), results["write"]
    assert results["write"].status == 403
    assert not isinstance(results["delete"], Exception), results["delete"]
    assert as_owner(ALICE).get("/v1/files").json()["data"] == []
    assert blob_files(settings) == []
    assert _staged(settings) == []
    assert keeper._used(ALICE) == (0, 0)
    owners = {o["owner"] for o in client.get("/v1/owners", headers=operator).json()["owners"]}
    assert ALICE not in owners


def test_delete_now_waits_for_a_paused_assembly_and_the_upload_leaves_nothing(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    keeper: Keeper,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alice = as_owner(ALICE)
    upload = alice.post("/v1/uploads", json={"filename": "a", "purpose": "batch", "bytes": 6})
    upload_id = upload.json()["id"]
    part = alice.post(f"/v1/uploads/{upload_id}/parts", content=b"abcdef").json()["id"]

    inside, release = threading.Event(), threading.Event()
    original = blobs_module.Blobs.read_part
    calls = {"n": 0}

    def held(self: Any, part_id: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            inside.set()
            assert release.wait(10)
        yield from original(self, part_id)

    monkeypatch.setattr(blobs_module.Blobs, "read_part", held)
    results: dict[str, Any] = {}
    assembler = _run(
        lambda: keeper.complete_upload(ALICE, upload_id, part_ids=[part]), results, "complete"
    )
    assert inside.wait(10)
    keeper.hide([ALICE], 30 * DAY)
    deleter = _run(lambda: keeper.delete_now(ALICE), results, "delete")
    time.sleep(0.3)
    assert deleter.is_alive(), "Delete now returned while an assembly of that person was under way"
    release.set()
    assembler.join(10)
    deleter.join(10)

    assert isinstance(results["complete"], StoreProblem), results["complete"]
    assert not isinstance(results["delete"], Exception), results["delete"]
    assert alice.get("/v1/files").json()["data"] == []
    assert blob_files(settings) == []
    assert _staged(settings) == []
    assert keeper._used(ALICE) == (0, 0)


def test_a_commit_for_a_hidden_owner_is_refused_in_its_own_transaction(
    keeper: Keeper, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hide the owner between the quota check and the commit: the commit itself refuses."""
    original = blobs_module.Blobs.write_temp

    def hide_meanwhile(self: Any, chunks: Any) -> Any:
        written = original(self, chunks)
        keeper.hide([ALICE], 30 * DAY)
        return written

    monkeypatch.setattr(blobs_module.Blobs, "write_temp", hide_meanwhile)
    with pytest.raises(StoreProblem) as raised:
        _create(keeper)
    assert raised.value.status == 403
    assert blob_files(settings) == []
    assert _staged(settings) == []


# --- 6. a copy that cannot be purged keeps the retry record ------------------------------


@pytest.fixture
def locked_copy(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[], None]]:
    """A `.bak` copy of the database that another process holds locked."""
    monkeypatch.setattr(database_module, "BACKUP_TIMEOUT", 0.05, raising=False)
    folder = settings.data_folder()
    copy = folder / "store.sqlite3.schema-1.bak"
    holder: dict[str, sqlite3.Connection] = {}

    def lock() -> None:
        holder["c"] = sqlite3.connect(copy, isolation_level=None)
        holder["c"].execute("BEGIN EXCLUSIVE")

    def unlock() -> None:
        holder["c"].execute("ROLLBACK")
        holder["c"].close()

    lock.unlock = unlock  # type: ignore[attr-defined]
    yield lock
    if "c" in holder:
        with contextlib.suppress(sqlite3.Error):
            unlock()


def test_a_copy_that_cannot_be_purged_keeps_the_marker_until_the_next_sweep_finishes(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    keeper: Keeper,
    settings: Settings,
    locked_copy: Any,
) -> None:
    alice = as_owner(ALICE)
    alice.upload(b"alice only")
    folder = settings.data_folder()
    import shutil

    shutil.copyfile(folder / "store.sqlite3", folder / "store.sqlite3.schema-1.bak")
    agent.removed = {"alice"}
    client.post("/v1/sweep", headers=operator)

    locked_copy()
    refused = client.post("/v1/removals/delete", headers=operator, json={"owner": ALICE})
    assert refused.status_code >= 500, refused.text
    assert "store.sqlite3.schema-1.bak" in refused.text, refused.text
    # The retry record survives: she is still a removed person, and only that.
    assert ALICE in keeper.removed_people()
    assert alice.get("/v1/files").json()["data"] == []

    # Still locked: a sweep names the file and keeps the marker.
    report = client.post("/v1/sweep", headers=operator).json()
    assert "store.sqlite3.schema-1.bak" in report["removal"]["error"], report
    assert ALICE in keeper.removed_people()

    locked_copy.unlock()
    report = client.post("/v1/sweep", headers=operator).json()
    assert "error" not in report["removal"], report
    assert keeper.removed_people() == {}
    connection = sqlite3.connect(folder / "store.sqlite3.schema-1.bak")
    try:
        for table in ("files", "uploads", "parts", "removed"):
            count = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE owner = ?", (ALICE,)
            ).fetchone()[0]
            assert count == 0, table
    finally:
        connection.close()


def test_a_marker_replaced_since_it_was_read_is_left_alone(
    keeper: Keeper, settings: Settings
) -> None:
    """Compare-and-delete: a deletion names the marker it saw, and a person hidden
    again since (a new marker, a new date) is not deleted by the old one."""
    kept = _create(keeper, b"kept")
    keeper.hide([ALICE], 30 * DAY)
    old = keeper.removed_people()[ALICE]["marker"]
    keeper.restore([ALICE])
    keeper.hide([ALICE], 0)  # hidden again, already due, under a new marker
    new = keeper.removed_people()[ALICE]["marker"]
    assert new != old
    assert keeper.delete_removed(ALICE, marker=old) is None
    assert len(blob_files(settings)) == 1, kept
    assert keeper.delete_removed(ALICE, marker=new) is not None
    assert blob_files(settings) == []


def test_a_marker_that_is_gone_deletes_nothing(keeper: Keeper, settings: Settings) -> None:
    _create(keeper, b"kept")
    keeper.hide([ALICE], 0)
    seen = keeper.removed_people()[ALICE]["marker"]
    # The date is already now, so it is final: a restoration cannot unhide.
    assert keeper.restore([ALICE]) == []
    assert keeper.delete_removed(ALICE, marker=seen) is not None
    # The marker is gone with the deletion: naming it again deletes nothing.
    assert keeper.delete_removed(ALICE, marker=seen) is None
    assert blob_files(settings) == []
