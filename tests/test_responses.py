"""Row 54: stored responses and conversations, each kept for one owner and ended.

The store keeps what the gateway's `/v1/responses` and `/v1/conversations`
doors hand it, in OpenAI's item shapes; it reads only each item's `id`.
Ownership, ends, removal and the quota are row 53's, called, not copied.
"""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Callable
from typing import Any

import httpx
from fastapi.testclient import TestClient

from eugene_plexus_store.settings import Settings

from .conftest import AgentStub, Caller, Clock

DAY = 24 * 3600
MISSING = "resp_" + "0" * 32


def _rid() -> str:
    return "resp_" + secrets.token_hex(16)


def _message(role: str, text: str) -> dict[str, Any]:
    kind = "output_text" if role == "assistant" else "input_text"
    return {
        "id": "msg_" + secrets.token_hex(8),
        "type": "message",
        "role": role,
        "status": "completed",
        "content": [{"type": kind, "text": text}],
    }


def keep(
    caller: Caller,
    question: str,
    answer: str,
    *,
    previous: str | None = None,
    conversation: str | None = None,
    instructions: str | None = None,
    ident: str | None = None,
) -> httpx.Response:
    ident = ident or _rid()
    body: dict[str, Any] = {
        "id": ident,
        "model": "m",
        "status": "completed",
        "input": [_message("user", question)],
        "output": [_message("assistant", answer)],
        "response": {
            "id": ident,
            "object": "response",
            "status": "completed",
            "model": "m",
            "instructions": instructions,
            "store": True,
        },
    }
    if previous is not None:
        body["previousResponseId"] = previous
    if conversation is not None:
        body["conversationId"] = conversation
    return caller.post("/v1/responses", json=body)


def _texts(items: list[dict[str, Any]]) -> list[str]:
    return [part["text"] for item in items for part in item.get("content", [])]


def _rows(settings: Settings, table: str, owner: str | None = None) -> int:
    with sqlite3.connect(settings.data_folder() / "store.sqlite3") as connection:
        query = f"SELECT COUNT(*) FROM {table}" + (" WHERE owner = ?" if owner else "")
        return int(connection.execute(query, (owner,) if owner else ()).fetchone()[0])


def test_expired_responses_are_gone(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    clock: Clock,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    kept = keep(alice, "What is two and two?", "Four.")
    assert kept.status_code == 201, kept.text
    meta = kept.json()
    # OpenAI's retention: thirty days.
    assert meta["expiresAt"] == meta["createdAt"] + 30 * DAY
    assert meta["rootId"] == meta["id"]
    whole = alice.get(f"/v1/responses/{meta['id']}").json()
    assert _texts(whole["response"]["output"]) == ["Four."]

    clock.advance(30 * DAY + 1)
    # Ended is gone at once, before any sweep: to its owner, as a missing id.
    for answer in (
        alice.get(f"/v1/responses/{meta['id']}"),
        alice.get(f"/v1/responses/{meta['id']}/context"),
        alice.get(f"/v1/responses/{meta['id']}/input_items"),
        keep(alice, "And three?", "Five.", previous=meta["id"]),
    ):
        assert answer.status_code == 404, answer.text
    assert _rows(settings, "responses") == 1

    report = client.post("/v1/sweep", headers=operator)
    assert report.status_code == 200, report.text
    assert report.json()["responses"] == 1
    assert _rows(settings, "responses") == 0


def test_another_owners_response_is_the_missing_404(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    hers = keep(alice, "A secret question", "A secret answer").json()["id"]
    for suffix in ("", "/context", "/input_items"):
        theirs = bob.get(f"/v1/responses/{hers}{suffix}")
        none = bob.get(f"/v1/responses/{MISSING}{suffix}")
        assert theirs.status_code == none.status_code == 404
        assert theirs.json()["detail"]["detail"].replace(hers, "X") == none.json()["detail"][
            "detail"
        ].replace(MISSING, "X")
        assert "secret" not in theirs.text
    assert bob.delete(f"/v1/responses/{hers}").status_code == 404
    # Chaining from it, or keeping one with its id, is refused the same way.
    assert keep(bob, "Go on", "No", previous=hers).status_code == 404
    assert alice.get(f"/v1/responses/{hers}").status_code == 200
    taken = keep(bob, "Mine", "Mine", ident=hers)
    assert taken.status_code == 409, taken.text
    assert "secret" not in taken.text
    assert alice.get(f"/v1/responses/{hers}").json()["response"]["output"][0]["content"][0][
        "text"
    ] == ("A secret answer")


def test_a_chain_is_the_conversation_so_far_and_lives_as_long_as_its_newest(
    as_owner: Callable[[str], Caller], clock: Clock
) -> None:
    alice = as_owner("person:alice")
    first = keep(alice, "one", "uno", instructions="Answer in Spanish.").json()
    clock.advance(10 * DAY)
    second = keep(alice, "two", "dos", previous=first["id"]).json()
    clock.advance(10 * DAY)
    third = keep(alice, "three", "tres", previous=second["id"]).json()
    assert first["rootId"] == second["rootId"] == third["rootId"] == first["id"]
    assert third["previousResponseId"] == second["id"]

    context = alice.get(f"/v1/responses/{third['id']}/context").json()
    assert _texts(context["items"]) == ["one", "uno", "two", "dos", "three", "tres"]
    assert context["rootId"] == first["id"]
    assert context["hadInstructions"] is False
    assert alice.get(f"/v1/responses/{first['id']}/context").json()["hadInstructions"] is True

    # The model was given the chain, then its own input: OpenAI's input_items.
    items = alice.get(f"/v1/responses/{third['id']}/input_items", params={"order": "asc"}).json()
    assert _texts(items["data"]) == ["one", "uno", "two", "dos", "three"]
    page = alice.get(f"/v1/responses/{third['id']}/input_items", params={"limit": 2}).json()
    assert _texts(page["data"]) == ["three", "dos"] and page["hasMore"] is True
    after = page["data"][-1]["id"]
    rest = alice.get(
        f"/v1/responses/{third['id']}/input_items", params={"limit": 2, "after": after}
    ).json()
    assert _texts(rest["data"]) == ["two", "uno"] and rest["hasMore"] is True

    # The first response would have ended 30 days after it was kept; the chain
    # it starts lives 30 days after its newest.
    clock.advance(15 * DAY)
    assert alice.get(f"/v1/responses/{first['id']}").status_code == 200
    assert _texts(alice.get(f"/v1/responses/{third['id']}/context").json()["items"])[0] == "one"
    # A fork from the middle is a chain of its own, sharing the same start.
    fork = keep(alice, "four?", "cuatro", previous=second["id"]).json()
    assert fork["rootId"] == first["id"]
    assert _texts(alice.get(f"/v1/responses/{fork['id']}/context").json()["items"]) == [
        "one",
        "uno",
        "two",
        "dos",
        "four?",
        "cuatro",
    ]


def test_a_deleted_link_breaks_the_chain_by_name(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    first = keep(alice, "one", "uno").json()["id"]
    second = keep(alice, "two", "dos", previous=first).json()["id"]
    gone = alice.delete(f"/v1/responses/{first}")
    assert gone.json() == {"id": first, "deleted": True}
    assert alice.get(f"/v1/responses/{first}").status_code == 404
    broken = alice.get(f"/v1/responses/{second}/context")
    assert broken.status_code == 409, broken.text
    assert first in broken.json()["detail"]["detail"]
    assert "deleted" in broken.json()["detail"]["detail"]


def test_responses_off_keeps_nothing_and_says_where(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
) -> None:
    alice = as_owner("person:alice")
    kept = keep(alice, "one", "uno").json()["id"]
    status = client.get("/v1/store/status", headers=operator).json()
    assert status["parts"] == {"files": True, "responses": True}
    patched = client.patch("/v1/config", headers=operator, json={"responsesEnabled": False})
    assert patched.status_code == 200 and patched.json()["applied"] == ["responsesEnabled"]
    assert client.get("/v1/store/status", headers=operator).json()["parts"]["responses"] is False
    for answer in (
        keep(alice, "two", "dos"),
        alice.get(f"/v1/responses/{kept}/context"),
        alice.post("/v1/conversations", json={}),
    ):
        assert answer.status_code == 403, answer.text
        assert "Settings -> Store" in answer.json()["detail"]["detail"]


def test_a_conversation_carries_its_items_and_the_responses_kept_in_it(
    as_owner: Callable[[str], Caller], clock: Clock
) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    made = alice.post(
        "/v1/conversations",
        json={"metadata": {"topic": "numbers"}, "items": [_message("user", "Count with me.")]},
    )
    assert made.status_code == 201, made.text
    conversation = made.json()
    ident = conversation["id"]
    assert conversation["items"] == 1 and conversation["metadata"] == {"topic": "numbers"}
    # Kept until deleted, as OpenAI keeps conversations (no end by default).
    assert "expiresAt" not in conversation

    clock.advance(DAY)
    first = keep(alice, "one", "uno", conversation=ident)
    assert first.status_code == 201, first.text
    second = keep(alice, "two", "dos", conversation=ident).json()
    assert second["conversationId"] == ident
    context = alice.get(f"/v1/conversations/{ident}/context").json()
    assert _texts(context["items"]) == ["Count with me.", "one", "uno", "two", "dos"]
    assert context["conversationId"] == ident

    # input_items: the conversation's items before the response, then its own input.
    items = alice.get(f"/v1/responses/{second['id']}/input_items", params={"order": "asc"}).json()
    assert _texts(items["data"]) == ["Count with me.", "one", "uno", "two"]

    added = alice.post(
        f"/v1/conversations/{ident}/items", json={"items": [_message("user", "three")]}
    )
    assert added.status_code == 201, added.text
    item = added.json()["data"][0]
    assert alice.get(f"/v1/conversations/{ident}/items/{item['id']}").json() == item
    listed = alice.get(f"/v1/conversations/{ident}/items", params={"limit": 2}).json()
    assert _texts(listed["data"]) == ["three", "dos"] and listed["hasMore"] is True

    # Bob reads, adds to and deletes none of it: the missing id's answer.
    missing = "conv_" + "0" * 48
    for path in ("", "/items", "/context", f"/items/{item['id']}"):
        theirs = bob.get(f"/v1/conversations/{ident}{path}")
        none = bob.get(f"/v1/conversations/{missing}{path}")
        assert theirs.status_code == none.status_code == 404, theirs.text
        assert "three" not in theirs.text
    assert (
        bob.post(f"/v1/conversations/{ident}/items", json={"items": [_message("user", "x")]})
    ).status_code == 404
    assert keep(bob, "x", "y", conversation=ident).status_code == 404
    assert bob.delete(f"/v1/conversations/{ident}").status_code == 404
    assert bob.post(f"/v1/conversations/{ident}", json={"metadata": {}}).status_code == 404

    updated = alice.post(f"/v1/conversations/{ident}", json={"metadata": {"topic": "more"}})
    assert updated.json()["metadata"] == {"topic": "more"}
    after = alice.delete(f"/v1/conversations/{ident}/items/{item['id']}")
    assert after.status_code == 200 and after.json()["items"] == 5
    assert alice.delete(f"/v1/conversations/{ident}").json() == {"id": ident, "deleted": True}
    assert alice.get(f"/v1/conversations/{ident}").status_code == 404
    # The responses kept in it stay until their own end, as OpenAI keeps them.
    assert alice.get(f"/v1/responses/{second['id']}").status_code == 200


def test_a_conversation_is_kept_until_deleted_by_default(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    clock: Clock,
    settings: Settings,
) -> None:
    """OpenAI keeps a conversation until it is deleted, and so does Eugene by
    default (full compatibility); a stored response still ends after 30 days."""
    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={"items": [_message("user", "hi")]}).json()["id"]
    assert keep(alice, "kept in it", "yes", conversation=ident).status_code == 201
    clock.advance(400 * DAY)
    assert alice.get(f"/v1/conversations/{ident}").status_code == 200
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["conversations"] == 0 and report["responses"] == 1
    assert _rows(settings, "conversations") == 1
    assert _texts(alice.get(f"/v1/conversations/{ident}/context").json()["items"]) == [
        "hi",
        "kept in it",
        "yes",
    ]


def test_with_the_setting_a_conversation_ends_days_after_its_last_use(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    clock: Clock,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    idle = alice.post("/v1/conversations", json={"items": [_message("user", "idle")]}).json()["id"]
    ident = alice.post("/v1/conversations", json={"items": [_message("user", "hi")]}).json()["id"]
    clock.advance(20 * DAY)
    assert keep(alice, "still here", "yes", conversation=ident).status_code == 201
    # The expert's setting applies to every conversation, from its last use: the
    # one idle 20 days already has 10 left, whenever the setting was made.
    patched = client.patch("/v1/config", headers=operator, json={"conversationDays": 30})
    assert patched.json()["applied"] == ["conversationDays"]
    shown = alice.get(f"/v1/conversations/{ident}").json()
    assert shown["expiresAt"] == clock.now + 30 * DAY
    clock.advance(10 * DAY + 1)
    assert alice.get(f"/v1/conversations/{idle}").status_code == 404
    assert alice.get(f"/v1/conversations/{ident}").status_code == 200
    clock.advance(20 * DAY)
    assert alice.get(f"/v1/conversations/{ident}").status_code == 404
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["conversations"] == 2
    assert _rows(settings, "conversations") == 0
    assert _rows(settings, "conversation_items") == 0
    # Back to never: nothing new ends.
    client.patch("/v1/config", headers=operator, json={"conversationDays": None})
    later = alice.post("/v1/conversations", json={}).json()
    assert "expiresAt" not in later


def test_items_must_name_an_id_and_twenty_at_a_time(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={}).json()["id"]
    many = {"items": [_message("user", str(n)) for n in range(21)]}
    assert alice.post(f"/v1/conversations/{ident}/items", json=many).status_code == 400
    nameless = {"items": [{"type": "message", "role": "user", "content": []}]}
    assert alice.post(f"/v1/conversations/{ident}/items", json=nameless).status_code == 400
    twice = _message("user", "same")
    assert (
        alice.post(f"/v1/conversations/{ident}/items", json={"items": [twice]})
    ).status_code == 201
    again = alice.post(f"/v1/conversations/{ident}/items", json={"items": [twice]})
    assert again.status_code == 409, again.text


def test_a_removed_persons_responses_and_conversations_are_hidden_then_deleted(
    client: TestClient,
    as_owner: Callable[[str], Caller],
    operator: dict[str, str],
    agent: AgentStub,
    settings: Settings,
) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    hers = keep(alice, "alice asks", "alice is told").json()["id"]
    conversation = alice.post(
        "/v1/conversations", json={"items": [_message("user", "alice's own")]}
    ).json()["id"]
    his = keep(bob, "bob asks", "bob is told").json()["id"]

    agent.removed = {"alice"}
    report = client.post("/v1/sweep", headers=operator).json()
    assert report["removal"]["hidden"] == 1
    # A person who keeps only responses or conversations is asked about too.
    assert agent.asked == [["alice", "bob"]]
    for answer in (
        alice.get(f"/v1/responses/{hers}"),
        alice.get(f"/v1/responses/{hers}/context"),
        alice.get(f"/v1/conversations/{conversation}"),
        alice.get(f"/v1/conversations/{conversation}/items"),
    ):
        assert answer.status_code == 404, answer.text
    assert keep(alice, "more", "no").status_code == 403
    owners = {o["owner"]: o for o in client.get("/v1/owners", headers=operator).json()["owners"]}
    assert owners["person:alice"]["responses"] == 1
    assert owners["person:alice"]["conversations"] == 1
    assert "deleteAt" in owners["person:alice"]
    assert _rows(settings, "responses", "person:alice") == 1

    deleted = client.post(
        "/v1/removals/delete", headers=operator, json={"owner": "person:alice"}
    ).json()
    assert deleted["responses"] == 1 and deleted["conversations"] == 1
    for table in ("responses", "conversations", "conversation_items"):
        assert _rows(settings, table, "person:alice") == 0
    assert bob.get(f"/v1/responses/{his}").status_code == 200


def test_responses_count_against_the_owners_bytes(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    alice = as_owner("person:alice")
    patched = client.patch(
        "/v1/config", headers=operator, json={"ownerQuotaGiB": 1, "ownerQuotaFiles": 1}
    )
    assert patched.status_code == 200
    # A response is not a file: an owner at its file count still keeps responses.
    assert alice.upload(b"one file").status_code == 201
    big = "x" * (2 * 1024 * 1024)
    assert keep(alice, big, "ok").status_code == 201
    owners = {o["owner"]: o for o in client.get("/v1/owners", headers=operator).json()["owners"]}
    assert owners["person:alice"]["bytes"] > 2 * 1024 * 1024
    assert owners["person:alice"]["files"] == 1
    assert owners["person:alice"]["responses"] == 1
    # An upload in progress holds its declared size: nearly all of the GiB.
    client.patch("/v1/config", headers=operator, json={"ownerQuotaFiles": None})
    used = owners["person:alice"]["bytes"]
    started = alice.post(
        "/v1/uploads",
        json={"filename": "big", "purpose": "batch", "bytes": 1024**3 - used - 1024 * 1024},
    )
    assert started.status_code == 201, started.text
    over = keep(alice, big, "too much")
    assert over.status_code == 413, over.text
    assert "1073741824 bytes" in over.json()["detail"]["detail"]
    # Conversations count the same way.
    conversation = alice.post("/v1/conversations", json={}).json()["id"]
    added = alice.post(
        f"/v1/conversations/{conversation}/items", json={"items": [_message("user", big)]}
    )
    assert added.status_code == 413, added.text


def test_the_owner_console_sees_counts_and_never_content(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    alice = as_owner("person:alice")
    keep(alice, "a private question", "a private answer")
    listed = client.get("/v1/owners", headers=operator)
    assert listed.status_code == 200
    assert "private" not in listed.text
    # The owner's session may not name an owner, so it cannot read the response.
    read = client.get(
        "/v1/responses/" + MISSING,
        headers={**operator, "X-Eugene-Plexus-Owner": "person:alice"},
    )
    assert read.status_code == 403
