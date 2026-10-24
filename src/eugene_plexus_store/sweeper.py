"""The sweep: what has ended is deleted, and removed people's objects with it.

It runs every `sweepIntervalMinutes`, and when the owner asks
(`POST /v1/sweep`). One sweep at a time: a timed one and an asked one
wait for each other.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from starlette.concurrency import run_in_threadpool

from .keeper import Keeper, SweepCounts, report_of
from .removal import AskFailed, PeopleCheck

log = logging.getLogger(__name__)


class Sweeper:
    def __init__(self, keeper: Keeper, people: PeopleCheck) -> None:
        self.keeper = keeper
        self.people = people
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="store-sweep")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.people.aclose()

    async def _loop(self) -> None:
        while True:
            try:
                await self.sweep()
            except Exception:  # the loop must outlive any one sweep
                log.exception("the store's sweep failed")
            await asyncio.sleep(self.keeper.config.sweep_seconds())

    async def sweep(self) -> dict[str, Any]:
        async with self._lock:
            started = time.time()
            try:
                counts = await run_in_threadpool(self.keeper.sweep)
            except Exception as exc:
                self.error = f"the sweep could not delete what ended: {exc}"
                report = report_of(SweepCounts(), at=started, removal=None)
                report["error"] = self.error
                await run_in_threadpool(self.keeper.save_report, report)
                raise
            removal = await self._removals(counts)
            report = report_of(counts, at=started, removal=removal)
            self.error = None
            await run_in_threadpool(self.keeper.save_report, report)
            if counts.files or counts.uploads or counts.blobs:
                log.info(
                    "swept: %d files, %d uploads, %d blobs (%d bytes) deleted",
                    counts.files,
                    counts.uploads,
                    counts.blobs,
                    counts.freed_bytes,
                )
            return report

    async def _removals(self, counts: SweepCounts) -> dict[str, Any]:
        owners = await run_in_threadpool(self.keeper.person_owners)
        held = set(await run_in_threadpool(self.keeper.held))
        removal: dict[str, Any] = {"checked": len(owners), "removed": 0, "held": sorted(held)}
        if not owners:
            return removal
        try:
            known, unknown = await self.people.unknown(owners)
        except AskFailed as exc:
            removal["error"] = str(exc)
            log.warning("removed people were not checked: %s", exc)
            return removal
        back = [owner for owner in known if owner in held]
        if back:
            await run_in_threadpool(self.keeper.unhold, back)
            held -= set(back)
        new = [owner for owner in unknown if owner not in held]
        brake = self.keeper.config.removal_brake()
        if brake and len(new) >= brake:
            await run_in_threadpool(self.keeper.hold, new)
            held |= set(new)
            log.warning(
                "the root no longer knows %d people who own files here; holding them until "
                "the owner releases them in Settings -> Store (removalBrake %d)",
                len(new),
                brake,
            )
        elif new:
            removed = await run_in_threadpool(self.keeper.remove_owners, new)
            counts.add(removed)
            removal["removed"] = len(new)
            log.info("deleted everything %d removed people kept here", len(new))
        removal["held"] = sorted(held)
        return removal

    async def release(self) -> dict[str, Any]:
        """Delete what the held people own, after asking the root once more."""
        async with self._lock:
            started = time.time()
            held = await run_in_threadpool(self.keeper.held)
            removal: dict[str, Any] = {"checked": len(held), "removed": 0, "held": []}
            counts = SweepCounts()
            if held:
                known, unknown = await self.people.unknown(held)
                if known:
                    await run_in_threadpool(self.keeper.unhold, known)
                if unknown:
                    counts = await run_in_threadpool(self.keeper.remove_owners, unknown)
                    removal["removed"] = len(unknown)
            report = report_of(counts, at=started, removal=removal)
            await run_in_threadpool(self.keeper.save_report, report)
            return report
