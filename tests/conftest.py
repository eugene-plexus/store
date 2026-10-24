"""Fixtures: a store on a temporary folder, a trust bundle, and the tokens callers send."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_store import tokens
from eugene_plexus_store.app import create_app
from eugene_plexus_store.auth_state import AuthState, load_auth_state
from eugene_plexus_store.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `EUGENE_PLEXUS_*` from the developer's own install reaches a test."""
    for key in [k for k in os.environ if k.startswith("EUGENE_PLEXUS_")]:
        monkeypatch.delenv(key, raising=False)


@dataclass
class FakeInstall:
    """A trust bundle as the store's agent keeps it. The store runs on node
    `root`; `far` is another machine of the install."""

    directory: Path
    name: str = "root"
    far_grants: tuple[str, ...] = ()
    identity: Ed25519PrivateKey = field(default_factory=tokens.generate_private_key)
    root: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    )
    node: tokens.Signer | None = None
    far: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="node:far")
    )
    version: int = 0

    def __post_init__(self) -> None:
        if self.node is None:
            self.node = tokens.Signer(key=tokens.generate_private_key(), issuer=self.recipient)
        self.publish()

    @property
    def recipient(self) -> str:
        return tokens.node_recipient(self.name)

    @property
    def authority(self) -> str:
        return tokens.public_b64(self.identity)

    @property
    def bundle_path(self) -> Path:
        return self.directory / "trust_bundle.json"

    def publish(self) -> None:
        assert self.node is not None
        self.version += 1
        bundle = tokens.build_bundle(
            authority=self.identity,
            version=self.version,
            epoch=1,
            keys=[
                self.root.trust_key(["authority"]),
                self.node.trust_key(["node", "gateway"]),
                self.far.trust_key(["node", *self.far_grants]),
            ],
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        tokens.write_bundle_file(self.bundle_path, bundle)

    def auth_state(self) -> AuthState:
        return load_auth_state(
            trust_bundle_file=str(self.bundle_path),
            trust_authority=self.authority,
            auth_recipient=self.recipient,
            service_token=self.service("store", ttl=365 * 24 * 3600),
        )

    def session(self, *, sub: str = "operator") -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_SESSION, sub=sub, aud=[self.recipient, "control"], ttl_seconds=3600
        )
        return token

    def client_key(self) -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_CLIENT, sub="app", aud=["gateway"], ttl_seconds=3600, jti="key-1"
        )
        return token

    def service(self, sub: str = "gateway", *, ttl: int = 3600) -> str:
        """A token this machine's agent minted for one of its own children."""
        assert self.node is not None
        token, _ = self.node.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=ttl
        )
        return token

    def foreign_service(self, sub: str = "gateway") -> str:
        """Another machine's token, addressed here and correctly signed."""
        token, _ = self.far.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=600
        )
        return token


@pytest.fixture
def install(tmp_path: Path) -> FakeInstall:
    return FakeInstall(tmp_path / "node")


class Clock:
    def __init__(self) -> None:
        self.now = float(int(time.time()))

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


@dataclass
class AgentStub:
    """This machine's agent, answering `POST /v1/people/known` as the root would."""

    removed: set[str] = field(default_factory=set)
    status: int = 200
    asked: list[list[str]] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json

        assert request.url.path == "/v1/people/known"
        subjects = json.loads(request.content)["subjects"]
        self.asked.append(subjects)
        if self.status != 200:
            return httpx.Response(
                self.status, json={"detail": {"detail": "the control root is locked"}}
            )
        return httpx.Response(
            200,
            json={
                "known": [s for s in subjects if s not in self.removed],
                "unknown": [s for s in subjects if s in self.removed],
            },
        )


@pytest.fixture
def agent() -> AgentStub:
    return AgentStub()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(config_file=tmp_path / "data" / "store.yaml", agent_url="http://127.0.0.1:9")


@pytest.fixture
def make_app(
    settings: Settings, install: FakeInstall, clock: Clock, agent: AgentStub
) -> Callable[..., FastAPI]:
    def build(*, auth: bool = True, **state: Any) -> FastAPI:
        app = create_app(settings=settings)
        if auth:
            app.state.auth_state = install.auth_state()
        app.state.clock = clock
        app.state.manual_sweeps = True
        app.state.agent_transport = httpx.MockTransport(agent.handler)
        for key, value in state.items():
            setattr(app.state, key, value)
        return app

    return build


@pytest.fixture
def client(make_app: Callable[..., FastAPI]) -> Iterator[TestClient]:
    with TestClient(make_app()) as c:
        yield c


class Caller:
    """The gateway, naming an owner on each call."""

    def __init__(self, client: TestClient, token: str, owner: str) -> None:
        self.client = client
        self.headers = {"Authorization": f"Bearer {token}", "X-Eugene-Plexus-Owner": owner}

    def upload(
        self,
        content: bytes,
        *,
        filename: str = "notes.txt",
        purpose: str = "user_data",
        **params: Any,
    ) -> httpx.Response:
        return self.client.post(
            "/v1/files",
            params={"filename": filename, "purpose": purpose, **params},
            content=content,
            headers={**self.headers, "Content-Type": "application/octet-stream"},
        )

    def get(self, path: str, **kw: Any) -> httpx.Response:
        return self.client.get(path, headers=self.headers, **kw)

    def post(self, path: str, **kw: Any) -> httpx.Response:
        headers = {**self.headers, **kw.pop("headers", {})}
        return self.client.post(path, headers=headers, **kw)

    def delete(self, path: str) -> httpx.Response:
        return self.client.delete(path, headers=self.headers)


@pytest.fixture
def as_owner(client: TestClient, install: FakeInstall) -> Callable[[str], Caller]:
    def caller(owner: str) -> Caller:
        return Caller(client, install.service("gateway"), owner)

    return caller


@pytest.fixture
def operator(client: TestClient, install: FakeInstall) -> dict[str, str]:
    return {"Authorization": f"Bearer {install.session()}"}


def blob_files(settings: Settings) -> list[Path]:
    root = settings.data_folder() / "blobs"
    return sorted(p for p in root.rglob("*") if p.is_file())
