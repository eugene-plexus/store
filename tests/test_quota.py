"""Quotas per owner: over them is a 413 that says the numbers."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from .conftest import Caller


def test_over_quota_is_413_with_the_numbers(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    patched = client.patch("/v1/config", json={"ownerQuotaFiles": 2}, headers=operator)
    assert patched.json()["applied"] == ["ownerQuotaFiles"]
    alice = as_owner("person:alice")
    assert alice.upload(b"one").status_code == 201
    assert alice.upload(b"two").status_code == 201
    third = alice.upload(b"three")
    assert third.status_code == 413
    assert "keeps 2 files" in third.json()["detail"]["detail"]
    assert "most it may keep is 2" in third.json()["detail"]["detail"]
    # Another owner's quota is its own.
    assert as_owner("person:bob").upload(b"one").status_code == 201

    # Bytes: an upload's declared size counts from the moment it starts.
    client.patch("/v1/config", json={"ownerQuotaFiles": None}, headers=operator)
    big = alice.post(
        "/v1/uploads",
        json={"filename": "huge", "purpose": "batch", "bytes": 11 * 1024**3},
    )
    assert big.status_code == 413
    said = big.json()["detail"]["detail"]
    assert "11811160064 bytes" in said
    assert "10737418240 bytes" in said
