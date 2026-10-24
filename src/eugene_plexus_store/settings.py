"""Startup-time settings, sourced from environment variables.

Distinct from the runtime config (`config.py`), which is editable through
`PATCH /v1/config`. These only say where things are: the config file, the
data folder, the interface to bind, and the auth wiring the agent hands
every child it spawns.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_STORE_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("store.yaml")
    """Where runtime config is persisted. PATCH /v1/config writes here."""

    data_dir: Path | None = None
    """Where the database and the blobs live. Unset, `store/` beside the
    config file, which under the agent is Eugene's own data folder: the
    store's data sits with the library's and the agent's, and is backed up
    and upgraded with them."""

    agent_url: str | None = None
    """This machine's agent, supplied by the supervisor: the store asks it
    which people the root still has."""

    bind_host: str = "127.0.0.1"
    """The gateway reaches the store through its agent, on this machine or
    through the machine port of the one that holds it, so the store stays
    on loopback."""

    safe_mode: bool = False
    """Skip the persisted config at startup and run on built-in defaults.
    Set by the agent after a boot failed on bad config, so the operator can
    reach /v1/config to fix it."""

    trust_bundle_file: str | None = None
    """The trust bundle the agent keeps beside `node.yaml`."""

    trust_authority: str | None = None
    """The public key that bundle must be signed by."""

    auth_recipient: str | None = None
    """This machine as a token's audience names it: `node:<name>`."""

    service_token: str | None = None
    """This store's own token, addressed to its machine alone."""

    def data_folder(self) -> Path:
        return self.data_dir if self.data_dir is not None else self.config_file.parent / "store"


def load_settings() -> Settings:
    return Settings()
