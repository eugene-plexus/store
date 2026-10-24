"""Entrypoint: `python -m eugene_plexus_store`."""

from __future__ import annotations

import contextlib
import logging
import os

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings

#: The contract's `servers` default. Under the agent the topology owns the
#: port and passes it here; this is for a standalone run.
_DEFAULT_PORT = 8084


def _resolve_port() -> int:
    env_port = os.environ.get("EUGENE_PLEXUS_STORE_BIND_PORT")
    return int(env_port) if env_port else _DEFAULT_PORT


def main() -> None:
    settings = load_settings()
    bootstrap = ConfigStore(settings.config_file)
    if not settings.safe_mode:
        # Only the log level is read here; the app reports a bad file itself.
        with contextlib.suppress(Exception):
            bootstrap.load()
    log_level = str(bootstrap.get("logLevel") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    uvicorn.run(
        create_app(settings),
        host=settings.bind_host,
        port=_resolve_port(),
        log_level=log_level.lower(),
        # Trust no forwarding header from anyone: every caller is the
        # gateway through an agent, and none of them may set the peer.
        forwarded_allow_ips=[],
    )


if __name__ == "__main__":
    main()
