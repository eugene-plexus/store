"""Refusals, as RFC 7807 problems the gateway turns into its doors' envelopes."""

from __future__ import annotations

from fastapi import HTTPException

from ._generated.models import Problem


class StoreProblem(Exception):
    """A refusal that names its cause. `kind` is the slug of its type."""

    def __init__(self, status: int, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail

    def http(self) -> HTTPException:
        return problem(self.status, self.title, self.detail)


def problem(status: int, title: str, detail: str) -> HTTPException:
    slug = title.replace(" ", "-").lower()
    return HTTPException(
        status_code=status,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/store#{slug}",
            title=title,
            status=status,
            detail=detail,
            component="store",
        ).model_dump(exclude_none=True),
    )


def not_found(kind: str, ident: str) -> StoreProblem:
    """The one answer for an id that does not exist, has ended, or is
    another owner's: the same status, title and words, so a caller learns
    nothing of what others keep."""
    return StoreProblem(404, "Not found", f"No such {kind}: {ident}")
