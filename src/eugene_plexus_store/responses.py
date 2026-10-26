"""Stored responses and conversations (row 54), each kept for one owner and ended.

The gateway's `/v1/responses` door keeps a response here when its request
says `store: true` (OpenAI's default), and reads a chain back when the next
request names it in `previous_response_id`; `/v1/conversations` keeps
conversations and their items. Items are kept as the gateway sends them, in
OpenAI's shapes: the store reads only each item's `id`.

**Row 53's rules, called and not copied.** Every lookup is `id = ? AND
owner = ?` and leaves out what has ended and what a removed person kept
(`LIVE`, `SHOWN`), so an id that is missing, ended or another owner's is
the same 404 (`not_found`). Writes are busy for their owner from
admission (the routes' `begin_write`), re-check the owner's removal in
their own transaction, and take room under the keeper's quota
(`_check_quota`, bytes only: a response is not a file). The sweep, a
removed person's deletion and Delete now reach these tables through the
keeper's own `_delete_where`, and every `.bak` copy through
`OWNER_TABLES`.

**A chain is its links.** A response kept with `previousResponseId` names
a live response of the same owner, so a chain never crosses owners and
never loops: each link is older than the one that names it. Keeping one
moves the end of every response before it to its own, so a chain lives as
long as its newest response. Reading a chain walks the links newest
first, each one read for the owner; a link that was deleted breaks the
chain, and the read says which one (409), rather than rebuild a
conversation with a hole in it.

**A conversation is kept until it is deleted**, as OpenAI keeps one (Troy's
standing rule: full compatibility by default). An expert may set
`conversationDays`: then each one ends that many days after its last use
(`used_at`: an item added, a response kept in it, its metadata changed),
counted with the setting in effect now, for the ones kept before it was set
too. Lookups read the end from `used_at` and the setting; the sweep writes it
into `expires_at` first (`Keeper.sweep`), so row 53's clauses delete what has
ended. Its items go with it.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .keeper import LIVE, SHOWN, Keeper, _removed
from .problems import StoreProblem, not_found

RESPONSE_ID = re.compile(r"^resp_[0-9a-f]{32}$")
CONVERSATION_ID = re.compile(r"^conv_[0-9a-f]{48}$")
STATUSES = ("completed", "incomplete", "failed", "cancelled")
#: OpenAI's: at most 20 items are added to a conversation at a time.
MAX_ADDED_ITEMS = 20
MAX_ITEM_ID = 128
#: A response's own items, at most: far more than one request can carry.
MAX_RESPONSE_ITEMS = 10000
#: The most a conversation so far may be when it is read back to continue it:
#: twice the gateway's 16 MiB request body.
MAX_CONTEXT_BYTES = 32 * 1024 * 1024
#: The most links a chain is walked through.
MAX_CHAIN = 10000
METADATA_PAIRS, METADATA_KEY, METADATA_VALUE = 16, 64, 512


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _bad(detail: str) -> StoreProblem:
    return StoreProblem(400, "Bad body", detail)


def _items(value: Any, field: str, *, most: int) -> list[dict[str, Any]]:
    """A list of items, each an object with an id of its own in the list."""
    if not isinstance(value, list):
        raise _bad(f"{field}: a list of items.")
    if len(value) > most:
        raise _bad(f"{field}: at most {most} items.")
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _bad(f"{field}[{index}]: an object.")
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or len(ident) > MAX_ITEM_ID:
            raise _bad(f"{field}[{index}].id: required, 1 to {MAX_ITEM_ID} characters.")
        if ident in seen:
            raise _bad(f"{field}[{index}].id: {ident} is named twice.")
        seen.add(ident)
    return value


def metadata(value: Any) -> dict[str, str]:
    """OpenAI's metadata: 16 pairs, keys of 64 characters, values of 512."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _bad("metadata: an object of strings.")
    if len(value) > METADATA_PAIRS:
        raise _bad(f"metadata: at most {METADATA_PAIRS} pairs.")
    for key, text in value.items():
        if not isinstance(key, str) or len(key) > METADATA_KEY:
            raise _bad(f"metadata: a key is at most {METADATA_KEY} characters.")
        if not isinstance(text, str) or len(text) > METADATA_VALUE:
            raise _bad(f"metadata.{key}: a string of at most {METADATA_VALUE} characters.")
    return dict(value)


def _page(
    items: Sequence[dict[str, Any]], *, limit: int, order: str, after: str | None
) -> dict[str, Any]:
    """One page of `items` (oldest first), in `order`, after the item `after`."""
    if order not in ("asc", "desc"):
        raise StoreProblem(400, "Bad order", "order: asc or desc.")
    limit = max(1, min(int(limit), 100))
    ordered = list(items) if order == "asc" else list(reversed(items))
    start = 0
    if after is not None:
        place = next((i for i, item in enumerate(ordered) if item.get("id") == after), None)
        if place is None:
            raise StoreProblem(400, "Bad cursor", f"after: no item {after} in this list.")
        start = place + 1
    page = ordered[start : start + limit]
    return {"data": page, "hasMore": start + limit < len(ordered)}


class Responses:
    """The response and conversation kinds, over the keeper's database, lock and rules."""

    def __init__(self, keeper: Keeper) -> None:
        self.keeper = keeper

    @property
    def database(self) -> Any:
        return self.keeper.database

    def _on(self) -> None:
        if not self.keeper.config.responses_enabled():
            raise StoreProblem(
                403,
                "Responses are off",
                "This install keeps no responses or conversations: the owner turned it off "
                "in Settings -> Store.",
            )

    def _life(self) -> int:
        return self.keeper.config.response_seconds()

    # -- lookups, for the owner only -------------------------------------------

    def _response(
        self, connection: sqlite3.Connection, owner: str, ident: str
    ) -> sqlite3.Row | None:
        if not RESPONSE_ID.match(ident):
            return None
        row = connection.execute(
            f"SELECT * FROM responses WHERE id = ? AND owner = ? AND {LIVE} AND {SHOWN}",
            (ident, owner, self.keeper.now()),
        ).fetchone()
        return row  # type: ignore[no-any-return]

    def _owned_response(self, owner: str, ident: str) -> sqlite3.Row:
        self.keeper.check_owner(owner)
        row = self._response(self.database.connection, owner, ident)
        if row is None:
            raise not_found("response", ident)
        return row

    def _conversation_end(self, used_at: int) -> int | None:
        """A conversation's end now: `conversationDays` after its last use, or none."""
        seconds = self.keeper.config.conversation_seconds()
        return None if seconds is None else used_at + seconds

    def _owned_conversation(self, owner: str, ident: str) -> sqlite3.Row:
        self.keeper.check_owner(owner)
        if CONVERSATION_ID.match(ident):
            seconds = self.keeper.config.conversation_seconds()
            row = self.database.connection.execute(
                f"SELECT * FROM conversations WHERE id = ? AND owner = ? AND {SHOWN} "
                "AND (? IS NULL OR used_at + ? > ?)",
                (ident, owner, seconds, seconds, self.keeper.now()),
            ).fetchone()
            if row is not None:
                return row  # type: ignore[no-any-return]
        raise not_found("conversation", ident)

    # -- shapes ----------------------------------------------------------------

    @staticmethod
    def _meta(row: sqlite3.Row) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "id": row["id"],
            "model": row["model"],
            "status": row["status"],
            "createdAt": row["created_at"],
            "expiresAt": row["expires_at"],
            "rootId": row["root_id"],
            "bytes": row["bytes"],
        }
        if row["previous_id"]:
            meta["previousResponseId"] = row["previous_id"]
        if row["conversation_id"]:
            meta["conversationId"] = row["conversation_id"]
        return meta

    def _conversation(self, row: sqlite3.Row) -> dict[str, Any]:
        count = self.database.connection.execute(
            "SELECT COUNT(*) FROM conversation_items WHERE conversation_id = ?", (row["id"],)
        ).fetchone()[0]
        shown: dict[str, Any] = {
            "id": row["id"],
            "createdAt": row["created_at"],
            "metadata": json.loads(row["metadata"]),
            "items": int(count),
        }
        ends = self._conversation_end(int(row["used_at"]))
        if ends is not None:
            shown["expiresAt"] = ends
        return shown

    # -- responses -------------------------------------------------------------

    def keep(self, owner: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """Keep one response that has ended, in its chain or its conversation."""
        self.keeper.check_owner(owner)
        self._on()
        ident = body.get("id")
        if not isinstance(ident, str) or not RESPONSE_ID.match(ident):
            raise _bad("id: resp_ and 32 hex digits.")
        model = body.get("model")
        if not isinstance(model, str) or len(model) > 512:
            raise _bad("model: a string of at most 512 characters.")
        status = body.get("status")
        if status not in STATUSES:
            raise _bad(f"status: one of {', '.join(STATUSES)}.")
        previous = body.get("previousResponseId")
        conversation = body.get("conversationId")
        for name, value in (("previousResponseId", previous), ("conversationId", conversation)):
            if value is not None and (not isinstance(value, str) or len(value) > 64):
                raise _bad(f"{name}: a string of at most 64 characters.")
        if previous is not None and conversation is not None:
            raise _bad("previousResponseId and conversationId: one or the other, not both.")
        given = _items(body.get("input"), "input", most=MAX_RESPONSE_ITEMS)
        made = _items(body.get("output"), "output", most=MAX_RESPONSE_ITEMS)
        shown = body.get("response")
        if not isinstance(shown, dict):
            raise _bad("response: the response object.")
        shown = {k: v for k, v in shown.items() if k != "output"}
        encoded = (_dumps(given), _dumps(made), _dumps(shown))
        size = sum(len(text.encode("utf-8")) for text in encoded)
        with self.database.lock:
            self.keeper._check_quota(owner, size, files=0)
            with self.database.transaction() as connection:
                if self.keeper._is_removed(owner):
                    raise _removed()
                taken = connection.execute(
                    "SELECT 1 FROM responses WHERE id = ?", (ident,)
                ).fetchone()
                if taken is not None:
                    raise StoreProblem(409, "Already kept", f"A response {ident} is already kept.")
                now = self.keeper.now()
                ends = now + self._life()
                root, depth, position = ident, 0, None
                if previous is not None:
                    link = self._response(connection, owner, previous)
                    if link is None:
                        raise not_found("response", previous)
                    root, depth = link["root_id"], int(link["depth"]) + 1
                    self._outlive(connection, owner, previous, ends)
                if conversation is not None:
                    held = self._owned_conversation(owner, conversation)
                    position = int(held["next_position"])
                connection.execute(
                    "INSERT INTO responses (id, owner, model, status, previous_id, root_id, "
                    "depth, conversation_id, conversation_position, input, output, body, "
                    "instructions, bytes, created_at, expires_at, generation) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ident,
                        owner,
                        model,
                        status,
                        previous,
                        root,
                        depth,
                        conversation,
                        position,
                        *encoded,
                        int(bool(shown.get("instructions"))),
                        size,
                        now,
                        ends,
                        self.keeper._next_generation(connection),
                    ),
                )
                if conversation is not None:
                    self._append(connection, owner, conversation, [*given, *made])
                row = connection.execute(
                    "SELECT * FROM responses WHERE id = ?", (ident,)
                ).fetchone()
                return self._meta(row)

    def _outlive(self, connection: sqlite3.Connection, owner: str, ident: str, ends: int) -> None:
        """Every response before this one in its chain ends no sooner than `ends`."""
        seen: set[str] = set()
        current: str | None = ident
        while current is not None and current not in seen and len(seen) < MAX_CHAIN:
            seen.add(current)
            connection.execute(
                "UPDATE responses SET expires_at = MAX(expires_at, ?) WHERE id = ? AND owner = ?",
                (ends, current, owner),
            )
            row = connection.execute(
                "SELECT previous_id FROM responses WHERE id = ? AND owner = ?", (current, owner)
            ).fetchone()
            current = row["previous_id"] if row is not None else None

    def get(self, owner: str, ident: str) -> dict[str, Any]:
        with self.database.lock:
            row = self._owned_response(owner, ident)
        meta = self._meta(row)
        meta["response"] = {**json.loads(row["body"]), "output": json.loads(row["output"])}
        return meta

    def delete(self, owner: str, ident: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            row = self._owned_response(owner, ident)
            connection.execute("DELETE FROM responses WHERE id = ?", (row["id"],))
        return {"id": row["id"], "deleted": True}

    def _chain(self, owner: str, row: sqlite3.Row) -> list[sqlite3.Row]:
        """`row` and every response before it, newest first, each the owner's."""
        connection = self.database.connection
        chain = [row]
        seen = {row["id"]}
        total = int(row["bytes"])
        while chain[-1]["previous_id"]:
            wanted = chain[-1]["previous_id"]
            link = self._response(connection, owner, wanted)
            if link is None:
                raise StoreProblem(
                    409,
                    "Chain broken",
                    f"Stored response {wanted}, earlier in this conversation, was deleted, so "
                    "the conversation cannot be rebuilt. Continue from a response before it, "
                    "or send the conversation in `input`.",
                )
            if link["id"] in seen or len(chain) >= MAX_CHAIN:
                raise StoreProblem(
                    413,
                    "Chain too long",
                    f"This conversation has more than {MAX_CHAIN} stored responses; send it "
                    "in `input` instead.",
                )
            total += int(link["bytes"])
            if total > MAX_CONTEXT_BYTES:
                raise self._too_large(total)
            seen.add(link["id"])
            chain.append(link)
        return chain

    @staticmethod
    def _too_large(total: int) -> StoreProblem:
        return StoreProblem(
            413,
            "Conversation too large",
            f"This conversation so far is more than {total} bytes, and one request carries at "
            f"most {MAX_CONTEXT_BYTES} ({MAX_CONTEXT_BYTES // (1024 * 1024)} MiB). Start a new "
            "one, or compact it and send it in `input`.",
        )

    @staticmethod
    def _links_items(chain: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for link in chain:
            items += json.loads(link["input"]) + json.loads(link["output"])
        return items

    def context(self, owner: str, ident: str) -> dict[str, Any]:
        """The conversation so far, to continue from response `ident`."""
        self.keeper.check_owner(owner)
        self._on()
        with self.database.lock:
            row = self._owned_response(owner, ident)
            chain = self._chain(owner, row)
        items = self._links_items(reversed(chain))
        return {
            "items": items,
            "bytes": sum(int(link["bytes"]) for link in chain),
            "rootId": row["root_id"],
            "hadInstructions": bool(row["instructions"]),
        }

    def input_items(
        self, owner: str, ident: str, *, limit: int, order: str, after: str | None
    ) -> dict[str, Any]:
        """What the model was given for response `ident`: the chain before it, or the
        conversation's items before it, then its own input."""
        with self.database.lock:
            row = self._owned_response(owner, ident)
            before: list[dict[str, Any]] = []
            if row["previous_id"]:
                link = self._response(self.database.connection, owner, row["previous_id"])
                if link is None:
                    raise StoreProblem(
                        409,
                        "Chain broken",
                        f"Stored response {row['previous_id']}, earlier in this conversation, "
                        "was deleted, so the items it gave cannot be listed.",
                    )
                before = self._links_items(reversed(self._chain(owner, link)))
            elif row["conversation_id"]:
                before = [
                    json.loads(item["item"])
                    for item in self.database.connection.execute(
                        "SELECT item FROM conversation_items "
                        "WHERE conversation_id = ? AND owner = ? AND position < ? "
                        "ORDER BY position",
                        (row["conversation_id"], owner, row["conversation_position"]),
                    )
                ]
        return _page([*before, *json.loads(row["input"])], limit=limit, order=order, after=after)

    # -- conversations ---------------------------------------------------------

    def _append(
        self,
        connection: sqlite3.Connection,
        owner: str,
        conversation: str,
        items: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Add `items` at the end; the conversation's last use is now. In a transaction."""
        held = connection.execute(
            "SELECT * FROM conversations WHERE id = ? AND owner = ?", (conversation, owner)
        ).fetchone()
        position = int(held["next_position"])
        now = self.keeper.now()
        added = 0
        for item in items:
            text = _dumps(item)
            size = len(text.encode("utf-8"))
            try:
                connection.execute(
                    "INSERT INTO conversation_items (conversation_id, id, owner, position, item, "
                    "bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (conversation, item["id"], owner, position, text, size, now),
                )
            except sqlite3.IntegrityError:
                raise StoreProblem(
                    409,
                    "Item already kept",
                    f"The conversation already holds an item {item['id']}.",
                ) from None
            position += 1
            added += size
        connection.execute(
            "UPDATE conversations SET next_position = ?, bytes = bytes + ?, used_at = ?, "
            "expires_at = ?, generation = ? WHERE id = ?",
            (
                position,
                added,
                now,
                self._conversation_end(now),
                self.keeper._next_generation(connection),
                conversation,
            ),
        )
        return list(items)

    def create_conversation(self, owner: str, *, meta: Any, items: Any) -> dict[str, Any]:
        self.keeper.check_owner(owner)
        self._on()
        pairs = metadata(meta)
        given = _items(items if items is not None else [], "items", most=MAX_ADDED_ITEMS)
        encoded = _dumps(pairs)
        size = len(encoded.encode("utf-8")) + sum(
            len(_dumps(item).encode("utf-8")) for item in given
        )
        with self.database.lock:
            self.keeper._check_quota(owner, size, files=0)
            with self.database.transaction() as connection:
                if self.keeper._is_removed(owner):
                    raise _removed()
                ident = "conv_" + secrets.token_hex(24)
                now = self.keeper.now()
                connection.execute(
                    "INSERT INTO conversations (id, owner, metadata, bytes, next_position, "
                    "created_at, used_at, expires_at, generation) "
                    "VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)",
                    (
                        ident,
                        owner,
                        encoded,
                        len(encoded.encode("utf-8")),
                        now,
                        now,
                        self._conversation_end(now),
                        self.keeper._next_generation(connection),
                    ),
                )
                self._append(connection, owner, ident, given)
                return self._conversation(
                    connection.execute(
                        "SELECT * FROM conversations WHERE id = ?", (ident,)
                    ).fetchone()
                )

    def get_conversation(self, owner: str, ident: str) -> dict[str, Any]:
        with self.database.lock:
            return self._conversation(self._owned_conversation(owner, ident))

    def update_conversation(self, owner: str, ident: str, meta: Any) -> dict[str, Any]:
        pairs = metadata(meta)
        encoded = _dumps(pairs)
        with self.database.transaction() as connection:
            row = self._owned_conversation(owner, ident)
            if self.keeper._is_removed(owner):
                raise _removed()
            was = len(row["metadata"].encode("utf-8"))
            now = self.keeper.now()
            connection.execute(
                "UPDATE conversations SET metadata = ?, bytes = bytes + ?, used_at = ?, "
                "expires_at = ? WHERE id = ?",
                (
                    encoded,
                    len(encoded.encode("utf-8")) - was,
                    now,
                    self._conversation_end(now),
                    ident,
                ),
            )
            return self._conversation(self._owned_conversation(owner, ident))

    def delete_conversation(self, owner: str, ident: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            row = self._owned_conversation(owner, ident)
            connection.execute(
                "DELETE FROM conversation_items WHERE conversation_id = ?", (row["id"],)
            )
            connection.execute("DELETE FROM conversations WHERE id = ?", (row["id"],))
        return {"id": row["id"], "deleted": True}

    def _all_items(self, conversation: str, owner: str) -> list[sqlite3.Row]:
        return list(
            self.database.connection.execute(
                "SELECT * FROM conversation_items WHERE conversation_id = ? AND owner = ? "
                "ORDER BY position",
                (conversation, owner),
            )
        )

    def list_items(
        self, owner: str, ident: str, *, limit: int, order: str, after: str | None
    ) -> dict[str, Any]:
        with self.database.lock:
            self._owned_conversation(owner, ident)
            items = [json.loads(row["item"]) for row in self._all_items(ident, owner)]
        return _page(items, limit=limit, order=order, after=after)

    def add_items(self, owner: str, ident: str, items: Any) -> dict[str, Any]:
        self.keeper.check_owner(owner)
        self._on()
        given = _items(items, "items", most=MAX_ADDED_ITEMS)
        if not given:
            raise _bad("items: at least one.")
        size = sum(len(_dumps(item).encode("utf-8")) for item in given)
        with self.database.lock:
            self.keeper._check_quota(owner, size, files=0)
            with self.database.transaction() as connection:
                self._owned_conversation(owner, ident)
                if self.keeper._is_removed(owner):
                    raise _removed()
                added = self._append(connection, owner, ident, given)
        return {"data": added, "hasMore": False}

    def get_item(self, owner: str, ident: str, item: str) -> dict[str, Any]:
        with self.database.lock:
            self._owned_conversation(owner, ident)
            row = self.database.connection.execute(
                "SELECT item FROM conversation_items "
                "WHERE conversation_id = ? AND id = ? AND owner = ?",
                (ident, item, owner),
            ).fetchone()
        if row is None:
            raise not_found("item", item)
        return json.loads(row["item"])  # type: ignore[no-any-return]

    def delete_item(self, owner: str, ident: str, item: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            self._owned_conversation(owner, ident)
            row = connection.execute(
                "SELECT bytes FROM conversation_items "
                "WHERE conversation_id = ? AND id = ? AND owner = ?",
                (ident, item, owner),
            ).fetchone()
            if row is None:
                raise not_found("item", item)
            connection.execute(
                "DELETE FROM conversation_items WHERE conversation_id = ? AND id = ?",
                (ident, item),
            )
            connection.execute(
                "UPDATE conversations SET bytes = bytes - ? WHERE id = ?", (row["bytes"], ident)
            )
            return self._conversation(self._owned_conversation(owner, ident))

    def conversation_context(self, owner: str, ident: str) -> dict[str, Any]:
        """Every item of conversation `ident`, oldest first, to continue it."""
        self.keeper.check_owner(owner)
        self._on()
        with self.database.lock:
            self._owned_conversation(owner, ident)
            rows = self._all_items(ident, owner)
        total = sum(int(row["bytes"]) for row in rows)
        if total > MAX_CONTEXT_BYTES:
            raise self._too_large(total)
        return {
            "items": [json.loads(row["item"]) for row in rows],
            "bytes": total,
            "conversationId": ident,
        }
