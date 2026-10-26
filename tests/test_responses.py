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
    # The gateway takes OpenAI's 20 at a time from a client; the store takes a turn's
    # items in one call (second-model review), so 21 is no refusal here.
    many = {"items": [_message("user", str(n)) for n in range(21)]}
    assert alice.post(f"/v1/conversations/{ident}/items", json=many).status_code == 201
    too_many = {"items": [_message("user", "x")] * 10001}
    assert alice.post(f"/v1/conversations/{ident}/items", json=too_many).status_code == 400
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


# --- the second-model review (2026-10-26) ------------------------------------------


def _decodes(monkeypatch: Any) -> list[int]:
    """Counts each JSON document the store's response module decodes (its own reads of
    kept items, not a request's body)."""
    import json as real

    from eugene_plexus_store import responses as module

    seen: list[int] = []

    class Counting:
        dumps = staticmethod(real.dumps)

        @staticmethod
        def loads(text: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(1)
            return real.loads(text, *args, **kwargs)

    monkeypatch.setattr(module, "json", Counting)
    return seen


def test_a_page_of_a_long_conversation_reads_only_that_page(
    as_owner: Callable[[str], Caller], monkeypatch: Any
) -> None:
    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={}).json()["id"]
    for start in range(0, 60, 20):
        items = [_message("user", f"item {n}") for n in range(start, start + 20)]
        added = alice.post(f"/v1/conversations/{ident}/items", json={"items": items})
        assert added.status_code == 201, added.text
    seen = _decodes(monkeypatch)
    page = alice.get(f"/v1/conversations/{ident}/items", params={"limit": 1}).json()
    assert _texts(page["data"]) == ["item 59"] and page["hasMore"] is True
    after = page["data"][0]["id"]
    nxt = alice.get(f"/v1/conversations/{ident}/items", params={"limit": 2, "after": after}).json()
    assert _texts(nxt["data"]) == ["item 58", "item 57"]
    asc = alice.get(
        f"/v1/conversations/{ident}/items", params={"limit": 2, "order": "asc", "after": after}
    ).json()
    assert asc["data"] == [] and asc["hasMore"] is False
    # One page, one decode per item on it: never the whole conversation.
    assert len(seen) <= 6, len(seen)


def test_a_history_too_large_is_refused_before_a_row_is_read(
    as_owner: Callable[[str], Caller], monkeypatch: Any
) -> None:
    from eugene_plexus_store import responses as module

    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={"items": [_message("user", "x" * 400)]}).json()[
        "id"
    ]
    lone = keep(alice, "y" * 400, "z").json()["id"]
    monkeypatch.setattr(module, "MAX_CONTEXT_BYTES", 300)
    seen = _decodes(monkeypatch)
    for path in (f"/v1/conversations/{ident}/context", f"/v1/responses/{lone}/context"):
        refused = alice.get(path)
        # The first link of a chain counts too.
        assert refused.status_code == 413, (path, refused.text)
    assert seen == []


def _bytes(client: TestClient, operator: dict[str, str], owner: str) -> int:
    listed = client.get("/v1/owners", headers=operator).json()["owners"]
    return int(next(o for o in listed if o["owner"] == owner)["bytes"])


def test_every_byte_written_counts_against_the_quota(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    alice = as_owner("person:alice")
    client.patch("/v1/config", headers=operator, json={"ownerQuotaGiB": 1})
    conversation = alice.post("/v1/conversations", json={}).json()["id"]
    room = 3 * 1024 * 1024
    started = alice.post(
        "/v1/uploads",
        json={
            "filename": "big",
            "purpose": "batch",
            "bytes": 1024**3 - _bytes(client, operator, "person:alice") - room,
        },
    )
    assert started.status_code == 201, started.text
    # 2 MiB kept with the response and 2 MiB more in its conversation: past the room.
    two = "x" * (2 * 1024 * 1024)
    over = keep(alice, two, "ok", conversation=conversation)
    assert over.status_code == 413, over.text
    # Metadata grows a conversation too: fill the room, then grow it.
    left = 1024**3 - _bytes(client, operator, "person:alice")
    filler = alice.post(
        f"/v1/conversations/{conversation}/items",
        json={"items": [_message("user", "f" * (left - 2048))]},
    )
    assert filler.status_code == 201, filler.text
    grown = alice.post(
        f"/v1/conversations/{conversation}",
        json={"metadata": {f"k{n}": "v" * 512 for n in range(16)}},
    )
    assert grown.status_code == 413, grown.text
    assert _bytes(client, operator, "person:alice") <= 1024**3


def test_item_ids_have_openais_shape_and_no_error_repeats_one(
    as_owner: Callable[[str], Caller],
) -> None:
    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={}).json()["id"]
    worded = {**_message("user", "x"), "id": "msg_my secret password"}
    refused = alice.post(f"/v1/conversations/{ident}/items", json={"items": [worded]})
    assert refused.status_code == 400, refused.text
    assert "secret" not in refused.text
    item = _message("user", "once")
    assert alice.post(f"/v1/conversations/{ident}/items", json={"items": [item]}).status_code == 201
    again = alice.post(f"/v1/conversations/{ident}/items", json={"items": [item]})
    assert again.status_code == 409
    assert item["id"] not in again.text
    twice = alice.post(f"/v1/conversations/{ident}/items", json={"items": [item, item]})
    assert twice.status_code == 400 and item["id"] not in twice.text
    cursor = alice.get(f"/v1/conversations/{ident}/items", params={"after": "msg_" + "f" * 32})
    assert cursor.status_code == 400 and "f" * 32 not in cursor.text


def test_turned_off_a_conversation_takes_no_new_metadata(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={"metadata": {"a": "b"}}).json()["id"]
    client.patch("/v1/config", headers=operator, json={"responsesEnabled": False})
    refused = alice.post(f"/v1/conversations/{ident}", json={"metadata": {"a": "c"}})
    assert refused.status_code == 403
    assert alice.get(f"/v1/conversations/{ident}").json()["metadata"] == {"a": "b"}


def test_continuing_a_response_kept_in_a_conversation_keeps_that_history(
    as_owner: Callable[[str], Caller],
) -> None:
    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={"items": [_message("user", "the start")]}).json()[
        "id"
    ]
    inside = keep(alice, "one", "uno", conversation=ident).json()["id"]
    context = alice.get(f"/v1/responses/{inside}/context").json()
    assert _texts(context["items"]) == ["the start", "one", "uno"]
    after = keep(alice, "two", "dos", previous=inside).json()["id"]
    assert _texts(alice.get(f"/v1/responses/{after}/context").json()["items"]) == [
        "the start",
        "one",
        "uno",
        "two",
        "dos",
    ]
    listed = alice.get(f"/v1/responses/{after}/input_items", params={"order": "asc"}).json()
    assert _texts(listed["data"]) == ["the start", "one", "uno", "two"]


def keep_at(caller: Caller, question: str, answer: str, conversation: str, position: int) -> str:
    ident = _rid()
    body = {
        "id": ident,
        "model": "m",
        "status": "completed",
        "conversationId": conversation,
        "conversationPosition": position,
        "input": [_message("user", question)],
        "output": [_message("assistant", answer)],
        "response": {"id": ident, "object": "response", "status": "completed"},
    }
    kept = caller.post("/v1/responses", json=body)
    assert kept.status_code == 201, kept.text
    return ident


def test_a_turn_is_given_the_conversation_as_it_read_it(
    as_owner: Callable[[str], Caller],
) -> None:
    alice = as_owner("person:alice")
    ident = alice.post("/v1/conversations", json={"items": [_message("user", "the start")]}).json()[
        "id"
    ]
    # Turns A and B read the conversation at once; A is kept first.
    read_a = alice.get(f"/v1/conversations/{ident}/context").json()
    read_b = alice.get(f"/v1/conversations/{ident}/context").json()
    assert read_a["position"] == read_b["position"] == 1
    keep_at(alice, "a asks", "a told", ident, read_a["position"])
    b = keep_at(alice, "b asks", "b told", ident, read_b["position"])
    listed = alice.get(f"/v1/responses/{b}/input_items", params={"order": "asc"}).json()
    assert _texts(listed["data"]) == ["the start", "b asks"]
    # A position past the conversation's end is no position it was read at.
    bad = {
        "id": _rid(),
        "model": "m",
        "status": "completed",
        "conversationId": ident,
        "conversationPosition": 999,
        "input": [],
        "output": [],
        "response": {},
    }
    assert alice.post("/v1/responses", json=bad).status_code == 400
    # A turn's items go in at once, however many: one call, one transaction.
    many = [_message("user", f"n{n}") for n in range(30)]
    added = alice.post(f"/v1/conversations/{ident}/items", json={"items": many})
    assert added.status_code == 201, added.text
