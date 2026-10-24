"""An object is seen only by its owner, and only the gateway may say who that is."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from .conftest import Caller, FakeInstall


def _without_id(body: dict, ident: str) -> dict:
    detail = body["detail"]
    return {**detail, "detail": detail["detail"].replace(ident, "<id>")}


def test_another_owners_id_is_a_plain_404(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    kept = alice.upload(b"alice's notes")
    assert kept.status_code == 201, kept.text
    ident = kept.json()["id"]
    missing = "file-" + "0" * 24

    for path in ("/v1/files/{}", "/v1/files/{}/content"):
        hers = bob.get(path.format(ident))
        none = bob.get(path.format(missing))
        assert hers.status_code == none.status_code == 404
        # The same words for an id that exists elsewhere and one that never did.
        assert _without_id(hers.json(), ident) == _without_id(none.json(), missing)

    deleted = bob.delete(f"/v1/files/{ident}")
    assert deleted.status_code == 404
    assert _without_id(deleted.json(), ident) == _without_id(
        bob.delete(f"/v1/files/{missing}").json(), missing
    )
    assert bob.get("/v1/files").json() == {"data": [], "hasMore": False}
    # Bob's attempts changed nothing of Alice's.
    assert alice.get(f"/v1/files/{ident}/content").content == b"alice's notes"
    assert [f["id"] for f in alice.get("/v1/files").json()["data"]] == [ident]


def test_another_owners_id_is_no_cursor(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    ident = alice.upload(b"a").json()["id"]
    bob.upload(b"b")
    assert bob.get("/v1/files", params={"after": ident}).status_code == 404


def test_only_a_gateway_token_may_name_the_owner(
    client: TestClient, install: FakeInstall, as_owner: Callable[[str], Caller]
) -> None:
    ident = as_owner("person:alice").upload(b"secret").json()["id"]
    owner = {"X-Eugene-Plexus-Owner": "person:alice"}

    def read(token: str | None) -> int:
        headers = dict(owner)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        return client.get(f"/v1/files/{ident}/content", headers=headers).status_code

    # The header alone, a client key, the owner's own session, and another
    # component's token from this machine: none may name an owner.
    assert read(None) == 401
    assert read(install.client_key()) == 403
    assert read(install.session()) == 403
    assert read(install.service("library")) == 403
    assert read(install.service("store")) == 403
    # Another machine's gateway token, from a key without the gateway grant.
    assert read(install.foreign_service("gateway")) == 401
    # This machine's gateway, and the owner it names.
    assert read(install.service("gateway")) == 200
    # The gateway with no owner is refused before anything is looked up.
    no_owner = client.get(
        f"/v1/files/{ident}", headers={"Authorization": f"Bearer {install.service('gateway')}"}
    )
    assert no_owner.status_code == 400
    bad = client.get(
        f"/v1/files/{ident}",
        headers={
            "Authorization": f"Bearer {install.service('gateway')}",
            "X-Eugene-Plexus-Owner": "person:../alice",
        },
    )
    assert bad.status_code == 400


def test_a_granted_machines_gateway_may_name_the_owner(
    make_app: Callable[..., object], install: FakeInstall
) -> None:
    install.far_grants = ("gateway",)
    install.publish()
    with TestClient(make_app()) as client:  # type: ignore[arg-type]
        kept = client.post(
            "/v1/files",
            params={"filename": "a.txt", "purpose": "user_data"},
            content=b"x",
            headers={
                "Authorization": f"Bearer {install.foreign_service('gateway')}",
                "X-Eugene-Plexus-Owner": "key:k1",
            },
        )
        assert kept.status_code == 201, kept.text


def test_the_owners_console_sees_counts_and_never_content(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    as_owner("person:alice").upload(b"12345", filename="diary.txt")
    as_owner("key:k1").upload(b"abc")
    answer = client.get("/v1/owners", headers=operator)
    assert answer.status_code == 200
    body = answer.json()
    assert {o["owner"]: (o["files"], o["bytes"]) for o in body["owners"]} == {
        "person:alice": (1, 5),
        "key:k1": (1, 3),
    }
    assert "diary" not in answer.text
    # The counts are the owner's console's alone.
    gateway = as_owner("person:alice")
    assert (
        client.get(
            "/v1/owners", headers={"Authorization": gateway.headers["Authorization"]}
        ).status_code
        == 401
    )


def test_only_the_four_owner_kinds_are_taken(as_owner: Callable[[str], Caller]) -> None:
    for owner in ("install", "person:p-1", "key:k_1", "component:agent"):
        assert as_owner(owner).upload(b"x").status_code == 201, owner
    # `service:` was the audience prefix retired with the shared key.
    for owner in ("service:agent", "person:", "person:a/b", "nobody"):
        assert as_owner(owner).upload(b"x").status_code == 400, owner
