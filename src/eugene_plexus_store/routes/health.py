"""GET /healthz: liveness, and whether the data folder and the last sweep are well."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.models import Health, Status
from ..sweeper import Sweeper

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    state = request.app.state
    safe_mode = bool(getattr(state, "safe_mode", False))
    problems = [
        message
        for message in (
            getattr(state, "config_error", None),
            getattr(state, "keeper_error", None),
            "running in safe mode" if safe_mode else None,
        )
        if message
    ]
    sweeper: Sweeper | None = getattr(state, "sweeper", None)
    if sweeper is not None and sweeper.error:
        problems.append(sweeper.error)
    if problems:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="store",
            safeMode=safe_mode,
            details={"error": "; ".join(problems)},
        )
    return Health(status=Status.ok, version=__version__, component="store")
