"""POST /v1/admin/restart: exit shortly after answering; the agent starts us again."""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter

from .._generated.models import RestartResult

router = APIRouter(tags=["meta"])

_EXIT_DELAY_MS = 500


@router.post("/v1/admin/restart", response_model=RestartResult, status_code=202)
async def restart() -> RestartResult:
    logging.getLogger(__name__).warning(
        "restart requested via /v1/admin/restart; exiting in %dms", _EXIT_DELAY_MS
    )
    asyncio.get_running_loop().call_later(_EXIT_DELAY_MS / 1000.0, lambda: os._exit(0))
    return RestartResult(
        scheduled=True,
        delayMs=_EXIT_DELAY_MS,
        message=f"The store exits in {_EXIT_DELAY_MS} ms; its agent starts it again.",
    )
