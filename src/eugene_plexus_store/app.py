"""FastAPI app factory."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .auth_state import load_auth_state
from .config import ConfigStore
from .dependencies import require_operator
from .keeper import Keeper
from .removal import PeopleCheck
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import files as files_routes
from .routes import health as health_routes
from .routes import store as store_routes
from .settings import Settings, load_settings
from .sweeper import Sweeper

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            trust_bundle_file=settings.trust_bundle_file,
            trust_authority=settings.trust_authority,
            auth_recipient=settings.auth_recipient,
            service_token=settings.service_token,
        )
    config = ConfigStore(settings.config_file)
    app.state.config_store = config
    app.state.safe_mode = settings.safe_mode
    app.state.config_error = None
    app.state.keeper = None
    app.state.keeper_error = None
    app.state.sweeper = None
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_STORE_SAFE_MODE=1); ignoring %s and keeping "
            "nothing. Fix the config via /v1/config, then restart without the variable.",
            settings.config_file,
        )
    else:
        try:
            config.load()
        except Exception as exc:
            # A bad config must not stop the store: its endpoints are how it is fixed.
            app.state.config_error = f"config file {settings.config_file} could not be read: {exc}"
            log.error("%s; running on defaults", app.state.config_error)
    keeper: Keeper | None = None
    sweeper: Sweeper | None = None
    if not settings.safe_mode:
        folder = settings.data_folder()
        keeper = Keeper(folder, config, clock=getattr(app.state, "clock", None) or time.time)
        try:
            keeper.open()
        except Exception as exc:
            app.state.keeper_error = f"the data folder {folder} could not be opened: {exc}"
            log.error("%s", app.state.keeper_error)
            keeper = None
    if keeper is not None:
        app.state.keeper = keeper
        people = PeopleCheck(
            settings.agent_url,
            app.state.auth_state.service_token,
            transport=getattr(app.state, "agent_transport", None),
        )
        sweeper = Sweeper(keeper, people)
        app.state.sweeper = sweeper
        if not getattr(app.state, "manual_sweeps", False):
            sweeper.start()
    try:
        yield
    finally:
        if sweeper is not None:
            await sweeper.stop()
        if keeper is not None:
            keeper.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(
        title="Eugene Plexus — store",
        description=(
            "What the gateway's doors keep for people, each object with an owner and an end."
        ),
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's default echoes the values sent; a file name is the caller's.
        fields = sorted({".".join(str(p) for p in error["loc"][1:]) for error in exc.errors()})
        return JSONResponse(
            status_code=400,
            content={
                "detail": {
                    "type": "https://github.com/eugene-plexus/store#bad-request",
                    "title": "Bad request",
                    "status": 400,
                    "detail": f"{', '.join(fields) or 'request'}: missing or not valid.",
                    "component": "store",
                }
            },
        )

    app.include_router(health_routes.router)
    app.include_router(files_routes.router)
    app.include_router(store_routes.router)
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)
    return app
