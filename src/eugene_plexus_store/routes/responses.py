"""The object routes for stored responses and conversations (row 54), for the gateway."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from ..dependencies import Caller, require_owner
from ..problems import problem
from ..responses import Responses
from .files import _admitted, _call, _keeper, read_capped

router = APIRouter(tags=["responses"])

#: A kept response: the request's own items (at most the gateway's 16 MiB body),
#: what the model made and the response object, with room to spare.
RESPONSE_BODY_BYTES = 64 * 1024 * 1024
#: Up to 20 items, or a conversation's start: the gateway's 16 MiB body and a margin.
ITEMS_BODY_BYTES = 20 * 1024 * 1024


def _responses(request: Request) -> Responses:
    keeper = _keeper(request)
    kept: Responses | None = getattr(request.app.state, "responses", None)
    if kept is None or kept.keeper is not keeper:
        kept = Responses(keeper)
        request.app.state.responses = kept
    return kept


async def _json(request: Request, limit: int) -> dict[str, Any]:
    chunks = await read_capped(request, limit)
    try:
        body = json.loads(b"".join(chunks) or b"null")
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise problem(400, "Bad body", "The body must be a JSON object.")
    return body


@router.post("/v1/responses", status_code=201)
async def keep_response(request: Request, caller: Caller = Depends(require_owner)) -> JSONResponse:
    kept = _responses(request)
    async with _admitted(request, caller.owner):
        body = await _json(request, RESPONSE_BODY_BYTES)
        answer = await _call(kept.keep, caller.owner, body)
    return JSONResponse(answer, status_code=201)


@router.get("/v1/responses/{response_id}")
async def get_response(
    request: Request, response_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).get, caller.owner, response_id)


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    request: Request, response_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).delete, caller.owner, response_id)


@router.get("/v1/responses/{response_id}/input_items")
async def response_input_items(
    request: Request,
    response_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    order: str = Query(default="desc"),
    after: str | None = Query(default=None, max_length=128),
    caller: Caller = Depends(require_owner),
) -> Any:
    return await _call(
        _responses(request).input_items,
        caller.owner,
        response_id,
        limit=limit,
        order=order,
        after=after,
    )


@router.get("/v1/responses/{response_id}/context")
async def response_context(
    request: Request, response_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).context, caller.owner, response_id)


@router.post("/v1/conversations", status_code=201)
async def create_conversation(
    request: Request, caller: Caller = Depends(require_owner)
) -> JSONResponse:
    kept = _responses(request)
    async with _admitted(request, caller.owner):
        body = await _json(request, ITEMS_BODY_BYTES)
        unknown = sorted(set(body) - {"metadata", "items"})
        if unknown:
            raise problem(400, "Bad body", f"{unknown[0]}: not a conversation field.")
        answer = await _call(
            kept.create_conversation,
            caller.owner,
            meta=body.get("metadata"),
            items=body.get("items"),
        )
    return JSONResponse(answer, status_code=201)


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(
    request: Request, conversation_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).get_conversation, caller.owner, conversation_id)


@router.post("/v1/conversations/{conversation_id}")
async def update_conversation(
    request: Request, conversation_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    kept = _responses(request)
    async with _admitted(request, caller.owner):
        body = await _json(request, 1024 * 1024)
        if "metadata" not in body or set(body) != {"metadata"}:
            raise problem(400, "Bad body", "metadata: required, and the only field.")
        return await _call(
            kept.update_conversation, caller.owner, conversation_id, body["metadata"]
        )


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(
    request: Request, conversation_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).delete_conversation, caller.owner, conversation_id)


@router.get("/v1/conversations/{conversation_id}/items")
async def list_conversation_items(
    request: Request,
    conversation_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    order: str = Query(default="desc"),
    after: str | None = Query(default=None, max_length=128),
    caller: Caller = Depends(require_owner),
) -> Any:
    return await _call(
        _responses(request).list_items,
        caller.owner,
        conversation_id,
        limit=limit,
        order=order,
        after=after,
    )


@router.post("/v1/conversations/{conversation_id}/items", status_code=201)
async def add_conversation_items(
    request: Request, conversation_id: str, caller: Caller = Depends(require_owner)
) -> JSONResponse:
    kept = _responses(request)
    async with _admitted(request, caller.owner):
        body = await _json(request, ITEMS_BODY_BYTES)
        if set(body) != {"items"}:
            raise problem(400, "Bad body", "items: required, and the only field.")
        answer = await _call(kept.add_items, caller.owner, conversation_id, body["items"])
    return JSONResponse(answer, status_code=201)


@router.get("/v1/conversations/{conversation_id}/items/{item_id}")
async def get_conversation_item(
    request: Request, conversation_id: str, item_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).get_item, caller.owner, conversation_id, item_id)


@router.delete("/v1/conversations/{conversation_id}/items/{item_id}")
async def delete_conversation_item(
    request: Request, conversation_id: str, item_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).delete_item, caller.owner, conversation_id, item_id)


@router.get("/v1/conversations/{conversation_id}/context")
async def conversation_context(
    request: Request, conversation_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_responses(request).conversation_context, caller.owner, conversation_id)
