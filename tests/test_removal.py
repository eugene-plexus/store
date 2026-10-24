"""A removed person's objects are deleted; nothing is deleted on a guess."""

from __future__ import annotations

from collections.abc import Callable

import httpx
from fastapi.testclient import TestClient

from eugene_plexus_store.settings import Settings

from .conftest import AgentStub, Caller, blob_files


def test_a_removed_persons_files_are_deleted_at_the_sweep(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    hers = alice.upload(b"alice only").json()["id"]
    upload = alice.post(
        "/v1/uploads", json={"filename": "x", "purpose": "batch", "bytes": 4}
    ).json()["id"]
    alice.post(f"/v1/uploads/{upload}/parts", content=b"ab")
    his = bob.upload(b"bob only").json()["id"]
    key = as_owner("key:k1").upload(b"a key's").json()["id"]

    agent.removed = {"alice"}
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["removal"] == {"checked": 2, "removed": 1, "held": []}
    assert agent.asked == [["alice", "bob"]]
    assert alice.get(f"/v1/files/{hers}").status_code == 404
    assert bob.get(f"/v1/files/{his}").status_code == 200
    assert as_owner("key:k1").get(f"/v1/files/{key}").status_code == 200
    assert len(blob_files(settings)) == 2
    assert list((settings.data_folder() / "parts").iterdir()) == []
    owners = {o["owner"] for o in client.get("/v1/owners", headers=operator).json()["owners"]}
    assert owners == {"person:bob", "key:k1"}


def test_many_removed_at_once_are_held_until_released(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
) -> None:
    ids = {name: as_owner(f"person:{name}").upload(name.encode()).json()["id"] for name in "abcd"}
    agent.removed = {"a", "b", "c"}
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["removal"]["removed"] == 0
    assert report["removal"]["held"] == ["person:a", "person:b", "person:c"]
    for name in "abc":
        assert as_owner(f"person:{name}").get(f"/v1/files/{ids[name]}").status_code == 200
    assert client.get("/v1/store/status", headers=operator).json()["held"] == [
        "person:a",
        "person:b",
        "person:c",
    ]

    # One of them is known again by the time the owner releases them.
    agent.removed = {"a", "b"}
    released = client.post("/v1/removals/release", headers=operator)
    assert released.status_code == 200, released.text
    assert released.json()["removal"]["removed"] == 2
    assert as_owner("person:a").get(f"/v1/files/{ids['a']}").status_code == 404
    assert as_owner("person:b").get(f"/v1/files/{ids['b']}").status_code == 404
    assert as_owner("person:c").get(f"/v1/files/{ids['c']}").status_code == 200
    assert client.get("/v1/store/status", headers=operator).json()["held"] == []


def test_a_root_that_cannot_be_asked_deletes_nothing(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
) -> None:
    hers = as_owner("person:alice").upload(b"x").json()["id"]
    agent.removed = {"alice"}
    agent.status = 503
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["removal"]["removed"] == 0
    assert "the control root is locked" in report["removal"]["error"]
    assert as_owner("person:alice").get(f"/v1/files/{hers}").status_code == 200
    refused = client.post("/v1/removals/release", headers=operator)
    assert refused.status_code == 200  # nothing held, nothing asked


def test_a_person_restored_during_the_check_keeps_what_they_keep_after_it(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
) -> None:
    """The root's answer is fenced by generation: only what is older than the
    question is deleted (review of row 53)."""
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
    assert report["removal"]["removed"] == 1
    assert alice.get(f"/v1/files/{old}").status_code == 404
    assert alice.get(f"/v1/files/{made['new']}").status_code == 200
