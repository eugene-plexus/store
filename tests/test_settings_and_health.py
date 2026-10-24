"""A bad config never stops the store, and every setting does what it says."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_store.settings import Settings

from .conftest import Caller, FakeInstall


def test_a_bad_config_file_degrades_and_the_config_stays_reachable(
    make_app: Callable[..., FastAPI], settings: Settings, install: FakeInstall
) -> None:
    settings.config_file.parent.mkdir(parents=True, exist_ok=True)
    settings.config_file.write_text("fileDefaultDays: [not, a, number]\n", encoding="utf-8")
    with TestClient(make_app()) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert "fileDefaultDays" in health["details"]["error"]
        headers = {"Authorization": f"Bearer {install.session()}"}
        assert client.get("/v1/config", headers=headers).json()["fileDefaultDays"] == 30
        fixed = client.patch("/v1/config", json={"fileDefaultDays": 7}, headers=headers)
        assert fixed.json()["applied"] == ["fileDefaultDays"]


def test_files_turned_off_are_refused_naming_the_setting(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    alice = as_owner("person:alice")
    kept = alice.upload(b"before").json()["id"]
    client.patch("/v1/config", json={"filesEnabled": False}, headers=operator)
    refused = alice.upload(b"after")
    assert refused.status_code == 403
    assert "Settings -> Store" in refused.json()["detail"]["detail"]
    # What was kept stays readable until it ends.
    assert alice.get(f"/v1/files/{kept}/content").content == b"before"
    status = client.get("/v1/store/status", headers=operator).json()
    assert status["parts"] == {"files": False}


def test_the_default_life_follows_the_setting(
    client: TestClient, as_owner: Callable[[str], Caller], operator: dict[str, str]
) -> None:
    client.patch("/v1/config", json={"fileDefaultDays": 2}, headers=operator)
    kept = as_owner("install").upload(b"x").json()
    assert kept["expiresAt"] - kept["createdAt"] == 2 * 86400
    schema = client.get("/v1/config/schema", headers=operator).json()
    assert schema["component"] == "store"
    assert {f["key"] for f in schema["fields"]} >= {"filesEnabled", "fileDefaultDays"}


def test_the_data_folder_sits_beside_the_config(tmp_path: Path) -> None:
    settings = Settings(config_file=tmp_path / "agent" / "store.yaml")
    assert settings.data_folder() == tmp_path / "agent" / "store"


def test_a_migration_keeps_a_copy_first(tmp_path: Path) -> None:
    import sqlite3

    from eugene_plexus_store import database

    folder = tmp_path / "store"
    first = database.Database(folder)
    first.open()
    first.close()

    def schema_2(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE extra (owner TEXT)")

    migrations = database.MIGRATIONS
    database.MIGRATIONS = (*migrations, schema_2)
    try:
        second = database.Database(folder)
        second.open()
        second.close()
    finally:
        database.MIGRATIONS = migrations
    assert (folder / "store.sqlite3.schema-1.bak").is_file()
