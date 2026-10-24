"""The object routes: files and uploads, for the gateway's doors."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from ..dependencies import Caller, require_owner
from ..keeper import MAX_BODY_BYTES, Keeper
from ..problems import StoreProblem, problem

router = APIRouter(tags=["files"])


def _keeper(request: Request) -> Keeper:
    keeper: Keeper | None = getattr(request.app.state, "keeper", None)
    if keeper is None:
        cause = getattr(request.app.state, "keeper_error", None) or "it is in safe mode"
        raise problem(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Store not open",
            f"The store cannot keep or read anything: {cause}.",
        )
    return keeper


async def read_capped(request: Request, limit: int = MAX_BODY_BYTES) -> list[bytes]:
    """The body, refused once it passes `limit`: a declared length first,
    then what actually arrives (a chunked body declares nothing)."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.strip().isdigit() and int(declared) > limit:
        raise _too_large(limit)
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise _too_large(limit)
        if chunk:
            chunks.append(chunk)
    return chunks


def _too_large(limit: int) -> Exception:
    return problem(
        status.HTTP_413_CONTENT_TOO_LARGE,
        "Too large",
        f"A request body here is at most {limit // (1024 * 1024)} MiB; send a larger file "
        "through an upload, in parts.",
    )


async def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return await run_in_threadpool(function, *args, **kwargs)
    except StoreProblem as exc:
        raise exc.http() from exc


@router.post("/v1/files", status_code=201)
async def create_file(
    request: Request,
    filename: str = Query(..., min_length=1, max_length=255),
    purpose: str = Query(...),
    mimeType: str | None = Query(default=None, max_length=255),
    expiresAfterSeconds: int | None = Query(default=None),
    caller: Caller = Depends(require_owner),
) -> JSONResponse:
    keeper = _keeper(request)
    chunks = await read_capped(request)
    kept = await _call(
        keeper.create_file,
        caller.owner,
        filename=filename,
        purpose=purpose,
        mime_type=mimeType,
        expires_after=expiresAfterSeconds,
        chunks=chunks,
    )
    return JSONResponse(kept, status_code=201)


@router.get("/v1/files")
async def list_files(
    request: Request,
    purpose: str | None = Query(default=None),
    limit: int = Query(default=10000, ge=1, le=10000),
    order: str = Query(default="desc"),
    after: str | None = Query(default=None, max_length=64),
    before: str | None = Query(default=None, max_length=64),
    caller: Caller = Depends(require_owner),
) -> Any:
    keeper = _keeper(request)
    return await _call(
        keeper.list_files,
        caller.owner,
        purpose=purpose,
        limit=limit,
        order=order,
        after=after,
        before=before,
    )


@router.get("/v1/files/{file_id}")
async def get_file(request: Request, file_id: str, caller: Caller = Depends(require_owner)) -> Any:
    return await _call(_keeper(request).get_file, caller.owner, file_id)


@router.delete("/v1/files/{file_id}")
async def delete_file(
    request: Request, file_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_keeper(request).delete_file, caller.owner, file_id)


@router.get("/v1/files/{file_id}/content")
async def file_content(
    request: Request, file_id: str, caller: Caller = Depends(require_owner)
) -> StreamingResponse:
    found, chunks = await _call(_keeper(request).content, caller.owner, file_id)
    return StreamingResponse(
        iterate_in_threadpool(chunks),
        media_type=found["mimeType"],
        headers={
            "Content-Length": str(found["bytes"]),
            "X-Eugene-Plexus-Sha256": found["sha256"],
        },
    )


@router.post("/v1/uploads", status_code=201)
async def create_upload(request: Request, caller: Caller = Depends(require_owner)) -> JSONResponse:
    keeper = _keeper(request)
    body = await _json(request)
    kept = await _call(
        keeper.create_upload,
        caller.owner,
        filename=_text(body, "filename"),
        purpose=_text(body, "purpose"),
        size=_integer(body, "bytes"),
        mime_type=_optional_text(body, "mimeType"),
        expires_after=_optional_integer(body, "expiresAfterSeconds"),
    )
    return JSONResponse(kept, status_code=201)


@router.post("/v1/uploads/{upload_id}/parts", status_code=201)
async def add_part(
    request: Request, upload_id: str, caller: Caller = Depends(require_owner)
) -> JSONResponse:
    keeper = _keeper(request)
    chunks = await read_capped(request)
    part = await _call(keeper.add_part, caller.owner, upload_id, chunks)
    return JSONResponse(part, status_code=201)


@router.post("/v1/uploads/{upload_id}/complete")
async def complete_upload(
    request: Request, upload_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    keeper = _keeper(request)
    body = await _json(request)
    parts = body.get("partIds")
    if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
        raise problem(400, "Bad parts", "partIds: a list of part ids, in order.")
    return await _call(
        keeper.complete_upload,
        caller.owner,
        upload_id,
        part_ids=parts,
        md5=_optional_text(body, "md5"),
        sha256=_optional_text(body, "sha256"),
    )


@router.post("/v1/uploads/{upload_id}/cancel")
async def cancel_upload(
    request: Request, upload_id: str, caller: Caller = Depends(require_owner)
) -> Any:
    return await _call(_keeper(request).cancel_upload, caller.owner, upload_id)


async def _json(request: Request) -> dict[str, Any]:
    chunks = await read_capped(request, 1024 * 1024)
    try:
        body = json.loads(b"".join(chunks) or b"null")
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise problem(400, "Bad body", "The body must be a JSON object.")
    return body


def _text(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value:
        raise problem(400, "Bad field", f"{key}: required, a string.")
    return value


def _optional_text(body: dict[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise problem(400, "Bad field", f"{key}: a string.")
    return value


def _integer(body: dict[str, Any], key: str) -> int:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise problem(400, "Bad field", f"{key}: required, an integer.")
    return value


def _optional_integer(body: dict[str, Any], key: str) -> int | None:
    if body.get(key) is None:
        return None
    return _integer(body, key)
