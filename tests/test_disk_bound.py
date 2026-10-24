"""The quota bounds the disk (review of row 53): nothing is written past it.

Room is taken before a byte is written, an upload is joined once at a
time, and an ended file counts until its bytes are deleted.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_store import blobs as blobs_module
from eugene_plexus_store.config import ConfigStore
from eugene_plexus_store.settings import Settings

from .conftest import Caller, Clock, blob_files


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every temporary file the store starts, by the bytes it was given."""
    seen: list[int] = []
    original = blobs_module.Blobs.write_temp

    def counted(self: Any, chunks: Any) -> Any:
        seen.append(-1)
        return original(self, chunks)

    monkeypatch.setattr(blobs_module.Blobs, "write_temp", counted)
    return seen


@pytest.fixture
def small_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ConfigStore, "quota_bytes", lambda self: 100)


def test_a_file_over_quota_writes_nothing(
    as_owner: Callable[[str], Caller], writes: list[int], small_quota: None
) -> None:
    alice = as_owner("person:alice")
    assert alice.upload(b"a" * 60).status_code == 201
    before = len(writes)
    refused = alice.upload(b"b" * 60)
    assert refused.status_code == 413
    assert len(writes) == before, "a temporary file was written for a refused upload"


def test_a_part_past_the_declared_size_writes_nothing(
    as_owner: Callable[[str], Caller], writes: list[int]
) -> None:
    alice = as_owner("person:alice")
    upload = alice.post(
        "/v1/uploads", json={"filename": "a", "purpose": "batch", "bytes": 5}
    ).json()["id"]
    assert alice.post(f"/v1/uploads/{upload}/parts", content=b"abc").status_code == 201
    before = len(writes)
    refused = alice.post(f"/v1/uploads/{upload}/parts", content=b"def")
    assert refused.status_code == 400
    assert len(writes) == before, "a part past the declared size was written to disk"


def test_an_upload_is_joined_once_at_a_time(
    client: TestClient, as_owner: Callable[[str], Caller], monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = as_owner("person:alice")
    upload = alice.post(
        "/v1/uploads", json={"filename": "a", "purpose": "batch", "bytes": 6}
    ).json()["id"]
    part = alice.post(f"/v1/uploads/{upload}/parts", content=b"abcdef").json()["id"]

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
    keeper = client.app.state.keeper  # type: ignore[attr-defined]
    results: dict[str, Any] = {}

    def first() -> None:
        try:
            results["first"] = keeper.complete_upload("person:alice", upload, part_ids=[part])
        except Exception as exc:  # recorded for the assertion below
            results["first"] = exc

    thread = threading.Thread(target=first)
    thread.start()
    assert inside.wait(10)
    second = alice.post(f"/v1/uploads/{upload}/complete", json={"partIds": [part]})
    release.set()
    thread.join(10)
    assert second.status_code == 409, second.text
    assert isinstance(results["first"], dict), results["first"]
    assert results["first"]["status"] == "completed"


def test_an_ended_file_counts_until_its_bytes_are_deleted(
    as_owner: Callable[[str], Caller], clock: Clock, settings: Settings, small_quota: None
) -> None:
    alice = as_owner("person:alice")
    for _ in range(5):
        first = alice.upload(b"x" * 60 + bytes([_]), expiresAfterSeconds=1)
        assert first.status_code == 201, first.text
        clock.advance(2)
    # However many short-lived files come and go, the disk holds at most the quota.
    assert len(blob_files(settings)) == 1
