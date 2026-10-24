"""A removed person's objects are hidden at once, deleted after the days, and back if the
root has them again first (call R53-REMOVE, Choice 2: row 7's rules, MRC27's true delete).
Nothing is hidden or deleted on a guess."""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Callable

import httpx
from fastapi.testclient import TestClient

from eugene_plexus_store.settings import Settings

from .conftest import AgentStub, Caller, Clock, blob_files

DAY = 24 * 3600


def _removal(report: dict[str, object]) -> dict[str, object]:
    removal = report["removal"]
    assert isinstance(removal, dict)
    return removal


def _owners(client: TestClient, operator: dict[str, str]) -> dict[str, dict[str, object]]:
    listed = client.get("/v1/owners", headers=operator)
    assert listed.status_code == 200, listed.text
    return {entry["owner"]: entry for entry in listed.json()["owners"]}


def _rows(settings: Settings, owner: str, path: str | None = None) -> int:
    database = sqlite3.connect(path or settings.data_folder() / "store.sqlite3")
    try:
        return sum(
            database.execute(f"SELECT COUNT(*) FROM {table} WHERE owner = ?", (owner,)).fetchone()[
                0
            ]
            for table in ("files", "uploads", "parts", "removed")
        )
    finally:
        database.close()


def test_a_removed_persons_objects_are_hidden_from_everyone_at_once(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    settings: Settings,
    clock: Clock,
) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    hers = alice.upload(b"alice only").json()["id"]
    upload = alice.post(
        "/v1/uploads", json={"filename": "x", "purpose": "batch", "bytes": 4}
    ).json()["id"]
    assert alice.post(f"/v1/uploads/{upload}/parts", content=b"ab").status_code == 201
    his = bob.upload(b"bob only").json()["id"]
    key = as_owner("key:k1").upload(b"a key's").json()["id"]
    missing = bob.get("/v1/files/file-000000000000000000000000")

    agent.removed = {"alice"}
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report) == {
        "checked": 2,
        "hidden": 1,
        "restored": 0,
        "deleted": 0,
        "lateRestored": 0,
    }
    assert agent.asked == [["alice", "bob"]]

    # The former owner, as the gateway would name her, finds nothing.
    for answer in (
        alice.get(f"/v1/files/{hers}"),
        alice.get(f"/v1/files/{hers}/content"),
        alice.delete(f"/v1/files/{hers}"),
        alice.get(f"/v1/uploads/{upload}"),
        alice.post(f"/v1/uploads/{upload}/parts", content=b"cd"),
        alice.post(f"/v1/uploads/{upload}/cancel"),
    ):
        assert answer.status_code == 404, answer.text
    assert alice.get("/v1/files").json()["data"] == []
    # Nothing new is kept for her while she is hidden.
    refused = alice.upload(b"while removed")
    assert refused.status_code == 403, refused.text
    assert "Eugene removed this person" in refused.json()["detail"]["detail"]
    started = alice.post("/v1/uploads", json={"filename": "y", "purpose": "batch", "bytes": 1})
    assert started.status_code == 403, started.text
    # Everyone else gets the answer for an id that does not exist.
    theirs = bob.get(f"/v1/files/{hers}")
    assert theirs.status_code == missing.status_code == 404
    assert theirs.text.replace(hers, "") == missing.text.replace(
        "file-000000000000000000000000", ""
    )
    assert bob.get(f"/v1/files/{his}").status_code == 200
    assert as_owner("key:k1").get(f"/v1/files/{key}").status_code == 200

    # Hidden, not deleted: her bytes and her part are still on disk.
    assert len(blob_files(settings)) == 3
    assert len(list((settings.data_folder() / "parts").iterdir())) == 1
    # The console sees her counts and her date, never her files.
    owners = _owners(client, operator)
    assert owners["person:alice"]["removedAt"] == int(clock.now)
    assert owners["person:alice"]["deleteAt"] == int(clock.now) + 30 * DAY
    assert owners["person:alice"]["files"] == 1
    assert "removedAt" not in owners["person:bob"]


def test_a_person_restored_within_the_days_gets_the_same_files_back(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    clock: Clock,
) -> None:
    alice = as_owner("person:alice")
    first = alice.upload(b"the first", filename="a.txt").json()
    second = alice.upload(b"the second", filename="b.txt").json()

    agent.removed = {"alice"}
    client.post("/v1/sweep", headers=operator)
    assert alice.get(f"/v1/files/{first['id']}").status_code == 404

    clock.advance(29 * DAY)
    agent.removed = set()
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report) == {
        "checked": 1,
        "hidden": 0,
        "restored": 1,
        "deleted": 0,
        "lateRestored": 0,
    }
    assert alice.get(f"/v1/files/{first['id']}").json() == first
    assert alice.get(f"/v1/files/{second['id']}/content").content == b"the second"
    assert {f["id"] for f in alice.get("/v1/files").json()["data"]} == {first["id"], second["id"]}
    assert "removedAt" not in _owners(client, operator)["person:alice"]
    assert alice.upload(b"back again").status_code == 201


def test_the_sweep_after_the_date_deletes_every_copy(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    settings: Settings,
    clock: Clock,
) -> None:
    alice = as_owner("person:alice")
    hers = alice.upload(b"alice only").json()["id"]
    bob = as_owner("person:bob").upload(b"bob only").json()["id"]
    agent.removed = {"alice"}
    client.post("/v1/sweep", headers=operator)
    # A later change never shortens a date already given.
    changed = client.patch("/v1/config", headers=operator, json={"removedPeopleDays": 1})
    assert changed.status_code == 200, changed.text
    # A copy kept before a migration holds her too.
    folder = settings.data_folder()
    shutil.copyfile(folder / "store.sqlite3", folder / "store.sqlite3.schema-1.bak")
    assert _rows(settings, "person:alice", str(folder / "store.sqlite3.schema-1.bak")) == 2

    clock.advance(30 * DAY - 1)
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report)["deleted"] == 0
    assert len(blob_files(settings)) == 2

    clock.advance(2)
    # The date holds even when the root cannot be asked: it was decided when it answered.
    agent.status = 503
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report)["deleted"] == 1
    assert report["files"] == 1
    assert len(blob_files(settings)) == 1
    assert _rows(settings, "person:alice") == 0
    assert _rows(settings, "person:alice", str(folder / "store.sqlite3.schema-1.bak")) == 0
    assert "person:alice" not in _owners(client, operator)

    # The root having her again later brings nothing back.
    agent.status = 200
    agent.removed = set()
    client.post("/v1/sweep", headers=operator)
    assert alice.get(f"/v1/files/{hers}").status_code == 404
    assert as_owner("person:bob").get(f"/v1/files/{bob}").status_code == 200


def test_delete_now_deletes_a_removed_person_at_once_and_for_good(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    hers = alice.upload(b"alice only").json()["id"]
    as_owner("person:bob").upload(b"bob only")
    # Only a removed person can be deleted now: a live one's files are theirs.
    live = client.post("/v1/removals/delete", headers=operator, json={"owner": "person:alice"})
    assert live.status_code == 404, live.text
    assert alice.get(f"/v1/files/{hers}").status_code == 200

    agent.removed = {"alice"}
    client.post("/v1/sweep", headers=operator)
    # Not the gateway's to ask for.
    gateway = client.post(
        "/v1/removals/delete", headers=alice.headers, json={"owner": "person:alice"}
    )
    assert gateway.status_code in (401, 403), gateway.text
    deleted = client.post("/v1/removals/delete", headers=operator, json={"owner": "person:alice"})
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["files"] == 1
    assert _removal(deleted.json())["deleted"] == 1
    assert len(blob_files(settings)) == 1
    assert _rows(settings, "person:alice") == 0

    agent.removed = set()
    client.post("/v1/sweep", headers=operator)
    assert alice.get(f"/v1/files/{hers}").status_code == 404


def test_not_at_all_deletes_a_removed_person_when_the_store_finds_out(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    settings: Settings,
) -> None:
    changed = client.patch("/v1/config", headers=operator, json={"removedPeopleRetention": "none"})
    assert changed.status_code == 200, changed.text
    hers = as_owner("person:alice").upload(b"alice only").json()["id"]
    as_owner("person:bob").upload(b"bob only")
    agent.removed = {"alice"}
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report) == {
        "checked": 2,
        "hidden": 0,
        "restored": 0,
        "deleted": 1,
        "lateRestored": 0,
    }
    assert len(blob_files(settings)) == 1
    agent.removed = set()
    client.post("/v1/sweep", headers=operator)
    assert as_owner("person:alice").get(f"/v1/files/{hers}").status_code == 404


def test_many_removed_at_once_are_hidden_like_one(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
) -> None:
    """No brake: the days are the guard against a wrong root (Troy, R53-REMOVE)."""
    ids = {name: as_owner(f"person:{name}").upload(name.encode()).json()["id"] for name in "abcd"}
    agent.removed = {"a", "b", "c"}
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report)["hidden"] == 3
    for name in "abc":
        assert as_owner(f"person:{name}").get(f"/v1/files/{ids[name]}").status_code == 404
    assert as_owner("person:d").get(f"/v1/files/{ids['d']}").status_code == 200
    agent.removed = set()
    assert _removal(client.post("/v1/sweep", headers=operator).json())["restored"] == 3
    for name in "abc":
        assert as_owner(f"person:{name}").get(f"/v1/files/{ids[name]}").status_code == 200


def test_a_root_that_cannot_be_asked_hides_nothing(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
) -> None:
    hers = as_owner("person:alice").upload(b"x").json()["id"]
    agent.removed = {"alice"}
    agent.status = 503
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report)["hidden"] == 0
    assert "the control root is locked" in str(_removal(report)["error"])
    assert as_owner("person:alice").get(f"/v1/files/{hers}").status_code == 200


def test_a_person_restored_during_the_check_keeps_what_they_keep_after_it(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
) -> None:
    """The root's answer is fenced by generation: only what is older than the question is
    deleted (review of row 53). Deleted at once, as "Not at all" does."""
    client.patch("/v1/config", headers=operator, json={"removedPeopleRetention": "none"})
    alice = as_owner("person:alice")
    old = alice.upload(b"before").json()["id"]
    keeper = client.app.state.keeper  # type: ignore[attr-defined]
    made: dict[str, str] = {}
    answer = agent.handler

    def restored_meanwhile(request: httpx.Request) -> httpx.Response:
        # Alice is restored under her id and keeps a file while the root answers.
        made["new"] = keeper.create_file(
            "person:alice",
            filename="after.txt",
            purpose="user_data",
            mime_type=None,
            expires_after=None,
            chunks=[b"after"],
        )["id"]
        return answer(request)

    client.app.state.sweeper.people._client._transport = httpx.MockTransport(restored_meanwhile)  # type: ignore[attr-defined]
    agent.removed = {"alice"}
    report = client.post("/v1/sweep", headers=operator).json()
    assert _removal(report)["deleted"] == 1
    assert alice.get(f"/v1/files/{old}").status_code == 404
    assert alice.get(f"/v1/files/{made['new']}").status_code == 200
