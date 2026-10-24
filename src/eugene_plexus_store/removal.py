"""Which people the root still has: asked through this machine's agent.

The root removes a person (row 7); the store learns it at its next sweep.
It sends the ids of the people who own something here to its agent's
`POST /v1/people/known`, and the agent asks the root with its own node
token. A store's own token cannot reach the root: a node may send only an
agent's or a granted gateway's token off its machine.

**Nothing is hidden or deleted on a guess.** A root that cannot be
asked, or does not answer as it should, changes nothing; a person it
names in neither list is a failed answer. What an answer does is the
sweep's (`sweeper.py`): hide, restore, delete on the date.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from ._generated.models import PeopleKnownResult
from ._http import internal_client

log = logging.getLogger(__name__)

PERSON = "person:"
BATCH = 500
TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class AskFailed(Exception):
    """Why the root could not say; the message is the observed cause."""


class PeopleCheck:
    """One client for the store's life (one HTTP client per instance)."""

    def __init__(
        self,
        agent_url: str | None,
        service_token: str | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.agent_url = agent_url.rstrip("/") if agent_url else None
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}
        self._client = internal_client(headers=headers, timeout=TIMEOUT, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def unknown(self, owners: Sequence[str]) -> tuple[list[str], list[str]]:
        """(known, unknown) of these `person:<id>` owners, as the root says."""
        if not owners:
            return [], []
        if self.agent_url is None:
            raise AskFailed(
                "this store has no agent to ask (it was not started by one), so nobody "
                "can be checked"
            )
        by_subject = {owner[len(PERSON) :]: owner for owner in owners if owner.startswith(PERSON)}
        subjects = list(by_subject)
        known: list[str] = []
        unknown: list[str] = []
        for start in range(0, len(subjects), BATCH):
            batch = subjects[start : start + BATCH]
            answer = await self._ask(batch)
            said_known = set(answer.get("known", []))
            said_unknown = set(answer.get("unknown", []))
            for subject in batch:
                if subject in said_unknown and subject not in said_known:
                    unknown.append(by_subject[subject])
                elif subject in said_known:
                    known.append(by_subject[subject])
                else:
                    raise AskFailed(
                        f"the root's answer left out {subject!r}, so nothing is deleted"
                    )
        return known, unknown

    async def _ask(self, subjects: list[str]) -> dict[str, Any]:
        url = f"{self.agent_url}/v1/people/known"
        try:
            response = await self._client.post(url, json={"subjects": subjects})
        except httpx.HTTPError as exc:
            raise AskFailed(
                f"this machine's agent at {self.agent_url} did not answer: "
                f"{str(exc) or type(exc).__name__}"
            ) from exc
        if response.status_code != 200:
            detail = ""
            try:
                body = response.json()
                if isinstance(body, dict):
                    inner = body.get("detail", body)
                    detail = str(inner.get("detail") if isinstance(inner, dict) else inner)
            except ValueError:
                detail = response.text[:300]
            raise AskFailed(
                f"the root could not be asked ({response.status_code}): {detail}".rstrip(": ")
            )
        try:
            answer = PeopleKnownResult.model_validate_json(response.content)
        except ValueError as exc:
            raise AskFailed("the agent's answer did not say known and unknown") from exc
        return {"known": list(answer.known), "unknown": list(answer.unknown)}
