"""Every object ends, and the sweep deletes what ended: rows and blobs."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from fastapi.testclient import TestClient

from eugene_plexus_store.settings import Settings

from .conftest import Caller, Clock, blob_files


def _rows(settings: Settings, table: str) -> int:
    with sqlite3.connect(settings.data_folder() / "store.sqlite3") as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_expired_files_are_deleted_by_the_sweep(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    clock: Clock,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    short = alice.upload(b"gone soon", expiresAfterSeconds=1).json()
    kept = alice.upload(b"stays").json()
    assert short["expiresAt"] == short["createdAt"] + 1
    assert kept["expiresAt"] == kept["createdAt"] + 30 * 86400
    assert len(blob_files(settings)) == 2

    clock.advance(2)
    # Ended is gone at once, before any sweep.
    assert alice.get(f"/v1/files/{short['id']}").status_code == 404
    assert [f["id"] for f in alice.get("/v1/files").json()["data"]] == [kept["id"]]
    assert _rows(settings, "files") == 2

    report = client.post("/v1/sweep", headers=operator)
    assert report.status_code == 200, report.text
    assert report.json()["files"] == 1
    assert report.json()["blobs"] == 1
    assert _rows(settings, "files") == 1
    assert len(blob_files(settings)) == 1
    assert alice.get(f"/v1/files/{kept['id']}/content").content == b"stays"
    status = client.get("/v1/store/status", headers=operator).json()
    assert status["lastSweep"]["files"] == 1


def test_a_shared_blob_goes_with_its_last_file(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    clock: Clock,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    alice.upload(b"same bytes", expiresAfterSeconds=1)
    his = bob.upload(b"same bytes", expiresAfterSeconds=100).json()
    assert len(blob_files(settings)) == 1

    clock.advance(2)
    client.post("/v1/sweep", headers=operator)
    assert len(blob_files(settings)) == 1
    assert bob.get(f"/v1/files/{his['id']}/content").content == b"same bytes"

    assert bob.delete(f"/v1/files/{his['id']}").json() == {"id": his["id"], "deleted": True}
    assert blob_files(settings) == []


def test_an_upload_ends_after_an_hour_with_its_parts(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    clock: Clock,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    upload = alice.post(
        "/v1/uploads", json={"filename": "a", "purpose": "batch", "bytes": 10}
    ).json()["id"]
    alice.post(f"/v1/uploads/{upload}/parts", content=b"12345")
    clock.advance(3601)
    late = alice.post(f"/v1/uploads/{upload}/parts", content=b"67890")
    assert late.status_code == 400
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["uploads"] == 1
    assert report["parts"] == 1
    assert list((settings.data_folder() / "parts").iterdir()) == []
    assert _rows(settings, "uploads") == 0


def test_an_end_past_the_installs_longest_is_refused(as_owner: Callable[[str], Caller]) -> None:
    refused = as_owner("key:k1").upload(b"x", expiresAfterSeconds=31 * 86400)
    assert refused.status_code == 400
    assert "2592000 seconds" in refused.json()["detail"]["detail"]


def test_a_blob_no_file_names_is_removed_by_the_sweep(
    client: TestClient, operator: dict[str, str], settings: Settings
) -> None:
    stray = settings.data_folder() / "blobs" / "ab" / ("ab" + "0" * 62)
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"left behind")
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["blobs"] == 1
    assert not stray.exists()
