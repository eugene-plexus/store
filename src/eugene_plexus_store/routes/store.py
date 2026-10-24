"""The owner's console: counts per owner (never content), the sweep, Delete now."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from .._generated.models import RemovalDeleteRequest
from ..config import ConfigStore
from ..dependencies import require_operator, require_reader
from ..problems import StoreProblem, problem
from ..sweeper import Sweeper
from .files import _keeper

router = APIRouter(tags=["store"])


def _sweeper(request: Request) -> Sweeper:
    sweeper: Sweeper | None = getattr(request.app.state, "sweeper", None)
    if sweeper is None:
        _keeper(request)
        raise problem(503, "Store not open", "The store's sweep is not running.")
    return sweeper


@router.get("/v1/owners", dependencies=[Depends(require_operator)])
async def owners(request: Request) -> Any:
    return await run_in_threadpool(_keeper(request).owners)


@router.get("/v1/store/status", dependencies=[Depends(require_reader)])
async def store_status(request: Request) -> Any:
    keeper = _keeper(request)
    config: ConfigStore = request.app.state.config_store
    report = await run_in_threadpool(keeper.last_report)
    answer: dict[str, Any] = {"parts": {"files": config.files_enabled()}}
    if report is not None:
        answer["lastSweep"] = report
    return answer


@router.post("/v1/sweep", dependencies=[Depends(require_operator)])
async def sweep_now(request: Request) -> Any:
    return await _sweeper(request).sweep()


@router.post("/v1/removals/delete", dependencies=[Depends(require_operator)])
async def delete_now(body: RemovalDeleteRequest, request: Request) -> Any:
    """Delete now: a removed person's objects, at once and for good (MRC27)."""
    try:
        return await _sweeper(request).delete_now(body.owner)
    except StoreProblem as exc:
        raise exc.http() from exc
