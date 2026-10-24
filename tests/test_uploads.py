"""Uploads: parts joined in order, checked, and only then a file."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from eugene_plexus_store.settings import Settings

from .conftest import Caller


def _start(caller: Caller, size: int, **extra: object) -> str:
    made = caller.post(
        "/v1/uploads",
        json={"filename": "big.bin", "purpose": "batch", "bytes": size, **extra},
    )
    assert made.status_code == 201, made.text
    assert made.json()["status"] == "pending"
    return str(made.json()["id"])


def _part(caller: Caller, upload: str, data: bytes) -> str:
    added = caller.post(
        f"/v1/uploads/{upload}/parts",
        content=data,
        headers={**caller.headers, "Content-Type": "application/octet-stream"},
    )
    assert added.status_code == 201, added.text
    return str(added.json()["id"])


def test_parts_assemble_and_a_bad_hash_is_refused(
    as_owner: Callable[[str], Caller], settings: Settings
) -> None:
    alice = as_owner("person:alice")
    pieces = [b"first-" * 1000, b"second-" * 1000, b"third"]
    whole = b"".join(pieces)
    upload = _start(alice, len(whole))
    parts = [_part(alice, upload, piece) for piece in pieces]

    # Out of order: the size adds up but the digest does not.
    wrong = alice.post(
        f"/v1/uploads/{upload}/complete",
        json={
            "partIds": [parts[1], parts[0], parts[2]],
            "sha256": hashlib.sha256(whole).hexdigest(),
        },
    )
    assert wrong.status_code == 400
    assert "sha256" in wrong.json()["detail"]["detail"]
    bad_md5 = alice.post(f"/v1/uploads/{upload}/complete", json={"partIds": parts, "md5": "0" * 32})
    assert bad_md5.status_code == 400
    assert "md5" in bad_md5.json()["detail"]["detail"]
    assert alice.get("/v1/files").json()["data"] == []

    done = alice.post(
        f"/v1/uploads/{upload}/complete",
        json={"partIds": parts, "md5": hashlib.md5(whole).hexdigest()},
    )
    assert done.status_code == 200, done.text
    body = done.json()
    assert body["status"] == "completed"
    assert body["file"]["bytes"] == len(whole)
    assert body["file"]["sha256"] == hashlib.sha256(whole).hexdigest()
    assert alice.get(f"/v1/files/{body['file']['id']}/content").content == whole
    # The parts are gone once joined; only the blob remains.
    assert list((settings.data_folder() / "parts").iterdir()) == []
    assert list((settings.data_folder() / "tmp").iterdir()) == []
    # A completed upload takes no more parts.
    again = alice.post(f"/v1/uploads/{upload}/parts", content=b"more")
    assert again.status_code == 400


def test_a_part_never_crosses_uploads_or_owners(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    bob = as_owner("person:bob")
    hers = _start(alice, 4)
    her_part = _part(alice, hers, b"abcd")
    his = _start(bob, 4)

    # Bob cannot add to, complete or cancel Alice's upload: it is not there for him.
    assert bob.post(f"/v1/uploads/{hers}/parts", content=b"zzzz").status_code == 404
    assert bob.post(f"/v1/uploads/{hers}/complete", json={"partIds": [her_part]}).status_code == 404
    assert bob.post(f"/v1/uploads/{hers}/cancel").status_code == 404
    # Nor can he join her part into his own upload.
    assert bob.post(f"/v1/uploads/{his}/complete", json={"partIds": [her_part]}).status_code == 404
    # And Alice cannot join her part into a second upload of her own.
    second = _start(alice, 4)
    assert (
        alice.post(f"/v1/uploads/{second}/complete", json={"partIds": [her_part]}).status_code
        == 404
    )
    done = alice.post(f"/v1/uploads/{hers}/complete", json={"partIds": [her_part]})
    assert done.status_code == 200
    assert bob.get("/v1/files").json()["data"] == []


def test_parts_past_the_declared_size_are_refused(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    upload = _start(alice, 5)
    _part(alice, upload, b"abc")
    over = alice.post(f"/v1/uploads/{upload}/parts", content=b"def")
    assert over.status_code == 400
    assert "declared 5" in over.json()["detail"]["detail"]


def test_a_cancelled_upload_leaves_no_parts(
    as_owner: Callable[[str], Caller], settings: Settings
) -> None:
    alice = as_owner("person:alice")
    upload = _start(alice, 3)
    _part(alice, upload, b"ab")
    cancelled = alice.post(f"/v1/uploads/{upload}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert list((settings.data_folder() / "parts").iterdir()) == []


def test_a_body_over_16_mib_is_refused(as_owner: Callable[[str], Caller]) -> None:
    alice = as_owner("person:alice")
    upload = _start(alice, 32 * 1024 * 1024)
    too_big = alice.post(
        f"/v1/uploads/{upload}/parts",
        content=b"x" * (16 * 1024 * 1024 + 1),
        headers={**alice.headers, "Content-Type": "application/octet-stream"},
    )
    assert too_big.status_code == 413
