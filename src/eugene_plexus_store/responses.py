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

**Nothing is read whole before its size is known** (second-model review):
a page of a conversation is a page in SQL, and a chain or a conversation read
back to be continued is summed (`item_bytes`, `bytes`) before a row of it is
fetched, its first link included. Every byte written takes room under the
quota first: a response kept in a conversation counts its copy there too,
and a conversation's metadata counts when it grows. An item's id has
OpenAI's shape (a prefix and hex digits), and no error repeats one.

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
from collections.abc import Mapping, Sequence
from typing import Any

from .keeper import LIVE, SHOWN, Keeper, _removed
from .problems import StoreProblem, not_found

RESPONSE_ID = re.compile(r"^resp_[0-9a-f]{32}$")
CONVERSATION_ID = re.compile(r"^conv_[0-9a-f]{48}$")
#: OpenAI's item ids: a short prefix and hex digits, so none carries words.
ITEM_ID = re.compile(r"^[a-z]{1,16}_[0-9a-f]{16,64}$")
STATUSES = ("completed", "incomplete", "failed", "cancelled")
#: OpenAI's: at most 20 items start a conversation. The gateway takes 20 at a time
#: from a client; a turn's items go in at once (`MAX_RESPONSE_ITEMS`).
MAX_ADDED_ITEMS = 20
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
        if not isinstance(ident, str) or not ITEM_ID.match(ident):
            raise _bad(
                f"{field}[{index}].id: required, OpenAI's shape (a prefix, an underscore and "
                "16 to 64 hex digits)."
            )
        if ident in seen:
            raise _bad(f"{field}[{index}].id: named twice in this list.")
        seen.add(ident)
    return value


def _size(items: Sequence[dict[str, Any]]) -> int:
    """What `items` take as they are kept, one row each in a conversation."""
    return sum(len(_dumps(item).encode("utf-8")) for item in items)


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


def _order(order: str) -> bool:
    """True for oldest first."""
    if order not in ("asc", "desc"):
        raise StoreProblem(400, "Bad order", "order: asc or desc.")
    return order == "asc"


def _bad_cursor() -> StoreProblem:
    return StoreProblem(400, "Bad cursor", "after: no such item in this list.")


def _cursor(after: str | None) -> str | None:
    if after is not None and not ITEM_ID.match(after):
        raise _bad_cursor()
    return after


def _page(
    items: Sequence[dict[str, Any]], *, limit: int, order: str, after: str | None
) -> dict[str, Any]:
    """One page of `items` (oldest first, already bounded in size), in `order`."""
    ordered = list(items) if _order(order) else list(reversed(items))
    limit = max(1, min(int(limit), 100))
    start = 0
    if _cursor(after) is not None:
        place = next((i for i, item in enumerate(ordered) if item.get("id") == after), None)
        if place is None:
            raise _bad_cursor()
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
        read_at = body.get("conversationPosition")
        if read_at is not None and (
            conversation is None or isinstance(read_at, bool) or not isinstance(read_at, int)
        ):
            raise _bad("conversationPosition: a whole number, with conversationId.")
        encoded = (_dumps(given), _dumps(made), _dumps(shown))
        size = sum(len(text.encode("utf-8")) for text in encoded)
        item_bytes = sum(len(text.encode("utf-8")) for text in encoded[:2])
        # Every byte written takes room: the conversation's copy of the items too.
        copy = _size([*given, *made]) if conversation is not None else 0
        with self.database.lock:
            self.keeper._check_quota(owner, size + copy, files=0)
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
                    if read_at is not None:
                        # What this turn was given: the conversation as it read it,
                        # whatever another turn added since (second-model review).
                        if not 0 <= read_at <= position:
                            raise _bad("conversationPosition: not a place this conversation had.")
                        position = read_at
                connection.execute(
                    "INSERT INTO responses (id, owner, model, status, previous_id, root_id, "
                    "depth, conversation_id, conversation_position, input, output, body, "
                    "instructions, item_bytes, bytes, created_at, expires_at, generation) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        item_bytes,
                        size + copy,
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

    _LINK = (
        "SELECT id, previous_id, root_id, item_bytes, conversation_id, conversation_position "
        "FROM responses WHERE id = ? AND owner = ? AND "
    )

    def _links(self, owner: str, ident: str) -> list[sqlite3.Row]:
        """Response `ident` and every response before it, newest first, each the owner's:
        ids and sizes only, nothing of their content. Under the lock."""
        connection = self.database.connection
        now = self.keeper.now()
        row = connection.execute(self._LINK + f"{LIVE} AND {SHOWN}", (ident, owner, now)).fetchone()
        if row is None:
            raise not_found("response", ident)
        chain = [row]
        seen = {row["id"]}
        while chain[-1]["previous_id"]:
            wanted = chain[-1]["previous_id"]
            link = connection.execute(
                self._LINK + f"{LIVE} AND {SHOWN}", (wanted, owner, now)
            ).fetchone()
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
            seen.add(link["id"])
            chain.append(link)
        return chain

    def _opening(
        self, owner: str, first: sqlite3.Row, *, strict: bool
    ) -> tuple[str | None, int, int]:
        """The conversation a chain began in, if any: (its id, the place the chain's
        first response read it at, the bytes of the items before that place). A
        conversation that was deleted is a broken history when `strict`."""
        conversation = first["conversation_id"]
        if not conversation:
            return None, 0, 0
        held = self.database.connection.execute(
            f"SELECT 1 FROM conversations WHERE id = ? AND owner = ? AND {SHOWN}",
            (conversation, owner),
        ).fetchone()
        if held is None:
            if strict:
                raise StoreProblem(
                    409,
                    "Chain broken",
                    "The conversation this response was kept in was deleted, so the history "
                    "it was given cannot be rebuilt. Send the conversation in `input`.",
                )
            return None, 0, 0
        below = int(first["conversation_position"] or 0)
        size = self.database.connection.execute(
            "SELECT COALESCE(SUM(bytes), 0) FROM conversation_items "
            "WHERE conversation_id = ? AND owner = ? AND position < ?",
            (conversation, owner, below),
        ).fetchone()[0]
        return conversation, below, int(size)

    def _conversation_rows(
        self, conversation: str, owner: str, *, below: int | None = None
    ) -> list[dict[str, Any]]:
        clauses = "conversation_id = ? AND owner = ?"
        values: list[Any] = [conversation, owner]
        if below is not None:
            clauses += " AND position < ?"
            values.append(below)
        return [
            json.loads(row["item"])
            for row in self.database.connection.execute(
                f"SELECT item FROM conversation_items WHERE {clauses} ORDER BY position", values
            )
        ]

    def _content(self, links: Sequence[sqlite3.Row], *, last_output: bool) -> list[dict[str, Any]]:
        """The items of `links` (oldest first): each one's input then its output; the
        last one's output only when `last_output`."""
        items: list[dict[str, Any]] = []
        for place, link in enumerate(links):
            row = self.database.connection.execute(
                "SELECT input, output FROM responses WHERE id = ?", (link["id"],)
            ).fetchone()
            items += json.loads(row["input"])
            if last_output or place < len(links) - 1:
                items += json.loads(row["output"])
        return items

    def _history(
        self, owner: str, ident: str, *, last_output: bool, strict: bool
    ) -> tuple[list[sqlite3.Row], list[dict[str, Any]], int]:
        """Response `ident`'s history, oldest first: the conversation its chain began
        in, up to where it was read, then each link's items. Its size is summed and
        checked before a row of it is read. Under the lock."""
        links = self._links(owner, ident)
        first = links[-1]
        conversation, below, opening = self._opening(owner, first, strict=strict)
        total = opening + sum(int(link["item_bytes"]) for link in links)
        if total > MAX_CONTEXT_BYTES:
            raise self._too_large(total)
        items = self._conversation_rows(conversation, owner, below=below) if conversation else []
        items += self._content(list(reversed(links)), last_output=last_output)
        return links, items, total

    @staticmethod
    def _too_large(total: int) -> StoreProblem:
        return StoreProblem(
            413,
            "Conversation too large",
            f"This conversation so far is more than {total} bytes, and one request carries at "
            f"most {MAX_CONTEXT_BYTES} ({MAX_CONTEXT_BYTES // (1024 * 1024)} MiB). Start a new "
            "one, or compact it and send it in `input`.",
        )

    def context(self, owner: str, ident: str) -> dict[str, Any]:
        """The conversation so far, to continue from response `ident`."""
        self.keeper.check_owner(owner)
        self._on()
        with self.database.lock:
            links, items, total = self._history(owner, ident, last_output=True, strict=True)
            had = self.database.connection.execute(
                "SELECT instructions FROM responses WHERE id = ?", (links[0]["id"],)
            ).fetchone()[0]
        return {
            "items": items,
            "bytes": total,
            "rootId": links[0]["root_id"],
            "hadInstructions": bool(had),
        }

    def input_items(
        self, owner: str, ident: str, *, limit: int, order: str, after: str | None
    ) -> dict[str, Any]:
        """What the model was given for response `ident`: the conversation its chain
        began in, up to where it was read, every earlier link's items, then its own
        input. Bounded as a continuation is (`MAX_CONTEXT_BYTES`)."""
        _order(order)
        with self.database.lock:
            _links, items, _total = self._history(owner, ident, last_output=False, strict=False)
        return _page(items, limit=limit, order=order, after=after)

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
                    "The conversation already holds an item with one of these ids.",
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

    def _place(self, connection: sqlite3.Connection, conversation: str, item: str) -> int:
        row = connection.execute(
            "SELECT position FROM conversation_items WHERE conversation_id = ? AND id = ?",
            (conversation, item),
        ).fetchone()
        if row is None:
            raise _bad_cursor()
        return int(row["position"])

    def create_conversation(self, owner: str, *, meta: Any, items: Any) -> dict[str, Any]:
        self.keeper.check_owner(owner)
        self._on()
        pairs = metadata(meta)
        given = _items(items if items is not None else [], "items", most=MAX_ADDED_ITEMS)
        encoded = _dumps(pairs)
        size = len(encoded.encode("utf-8")) + _size(given)
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
        self.keeper.check_owner(owner)
        # A write: refused while the owner has turned responses off (review).
        self._on()
        pairs = metadata(meta)
        encoded = _dumps(pairs)
        with self.database.lock:
            row = self._owned_conversation(owner, ident)
            growth = len(encoded.encode("utf-8")) - len(row["metadata"].encode("utf-8"))
            if growth > 0:
                # Metadata that grows takes room like any other byte (review).
                self.keeper._check_quota(owner, growth, files=0)
            with self.database.transaction() as connection:
                self._owned_conversation(owner, ident)
                if self.keeper._is_removed(owner):
                    raise _removed()
                now = self.keeper.now()
                connection.execute(
                    "UPDATE conversations SET metadata = ?, bytes = bytes + ?, used_at = ?, "
                    "expires_at = ? WHERE id = ?",
                    (encoded, growth, now, self._conversation_end(now), ident),
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

    def list_items(
        self, owner: str, ident: str, *, limit: int, order: str, after: str | None
    ) -> dict[str, Any]:
        """One page, read as one page: never the whole conversation (review)."""
        ascending = _order(order)
        limit = max(1, min(int(limit), 100))
        with self.database.lock:
            self._owned_conversation(owner, ident)
            clauses = "conversation_id = ? AND owner = ?"
            values: list[Any] = [ident, owner]
            if _cursor(after) is not None:
                assert after is not None
                clauses += " AND position > ?" if ascending else " AND position < ?"
                values.append(self._place(self.database.connection, ident, after))
            rows = self.database.connection.execute(
                f"SELECT item FROM conversation_items WHERE {clauses} "
                f"ORDER BY position {'ASC' if ascending else 'DESC'} LIMIT ?",
                (*values, limit + 1),
            ).fetchall()
            page = [json.loads(row["item"]) for row in rows[:limit]]
        return {"data": page, "hasMore": len(rows) > limit}

    def add_items(self, owner: str, ident: str, items: Any) -> dict[str, Any]:
        self.keeper.check_owner(owner)
        self._on()
        # One call, one transaction, a turn's items however many (review); a client's
        # 20 at a time is the gateway's door.
        given = _items(items, "items", most=MAX_RESPONSE_ITEMS)
        if not given:
            raise _bad("items: at least one.")
        size = _size(given)
        with self.database.lock:
            self.keeper._check_quota(owner, size, files=0)
            with self.database.transaction() as connection:
                self._owned_conversation(owner, ident)
                if self.keeper._is_removed(owner):
                    raise _removed()
                added = self._append(connection, owner, ident, given)
        return {"data": added, "hasMore": False}

    @staticmethod
    def _no_item() -> StoreProblem:
        """A missing item, its id not repeated."""
        return StoreProblem(404, "Not found", "No such item in this conversation.")

    def get_item(self, owner: str, ident: str, item: str) -> dict[str, Any]:
        with self.database.lock:
            self._owned_conversation(owner, ident)
            row = self.database.connection.execute(
                "SELECT item FROM conversation_items "
                "WHERE conversation_id = ? AND id = ? AND owner = ?",
                (ident, item, owner),
            ).fetchone()
        if row is None:
            raise self._no_item()
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
                raise self._no_item()
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
            held = self._owned_conversation(owner, ident)
            # Summed before a row is read (review): a conversation near the quota is
            # refused, not loaded.
            total = int(
                self.database.connection.execute(
                    "SELECT COALESCE(SUM(bytes), 0) FROM conversation_items "
                    "WHERE conversation_id = ? AND owner = ?",
                    (ident, owner),
                ).fetchone()[0]
            )
            if total > MAX_CONTEXT_BYTES:
                raise self._too_large(total)
            items = self._conversation_rows(ident, owner)
        return {
            "items": items,
            "bytes": total,
            "conversationId": ident,
            # Sent back when the turn is kept: what it was given (review).
            "position": int(held["next_position"]),
        }
