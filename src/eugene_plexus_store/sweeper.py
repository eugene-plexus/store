"""The sweep: what has ended is deleted; removed people are hidden, restored or deleted.

It runs every `sweepIntervalMinutes`, and when the owner asks
(`POST /v1/sweep`). One sweep at a time: a timed one, an asked one and a
Delete now wait for each other.

**Removed people** (call R53-REMOVE, Choice 2: row 7's rules). Each sweep
asks the root about every person who owns something here, the hidden ones
included:

- a person it no longer has is hidden at once, with a delete date
  `removedPeopleDays` from now; with `removedPeopleRetention` "none" their
  objects are deleted at once instead;
- a hidden person it has again is restored: the same ids and bytes, but only
  before their date: a date that has passed is final;
- a root that cannot be asked hides nobody and restores nobody;
- a hidden person whose date has passed is deleted for good, every copy,
  whether or not the root could be asked this time: the date was decided
  when it answered. The markers are read again after the answer, and a
  deletion is compare-and-delete on the marker seen, so an answer that is
  stale by the time it is used cannot delete a person restored meanwhile;
- a copy that cannot be purged keeps the marker, and the sweep names it.

Each deletion is fenced by the generation read before the root was asked.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import time
from typing import Any

from starlette.concurrency import run_in_threadpool

from .keeper import Keeper, SweepCounts, report_of
from .problems import StoreProblem
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
        keeper = self.keeper
        owners = await run_in_threadpool(keeper.person_owners)
        removal: dict[str, Any] = {
            "checked": len(owners),
            "hidden": 0,
            "restored": 0,
            "deleted": 0,
            "lateRestored": 0,
        }
        errors: list[str] = []
        # Read before the root is asked: only what is older than its answer is
        # deleted (review of row 53).
        fence = await run_in_threadpool(keeper.generation)
        before = await run_in_threadpool(keeper.removed_people)
        if owners:
            try:
                known, unknown = await self.people.unknown(owners)
            except AskFailed as exc:
                errors.append(str(exc))
                log.warning("removed people were not checked: %s", exc)
            else:
                # What is hidden is read again now: the answer may be stale by the time
                # it is used, and a person restored meanwhile must not be acted on.
                fresh = await run_in_threadpool(keeper.removed_people)
                back = await run_in_threadpool(keeper.restore, [o for o in known if o in fresh])
                removal["restored"] = len(back)
                removal["lateRestored"] = len([o for o in known if o in fresh]) - len(back)
                if back:
                    log.info(
                        "the root has %d removed people again: their files are back", len(back)
                    )
                if removal["lateRestored"]:
                    log.info(
                        "the root has %d removed people again, but their date had passed: "
                        "their files are deleted",
                        removal["lateRestored"],
                    )
                # A person already hidden when the question was asked is not hidden again
                # by that answer: it may be the one that was restored since.
                new = [o for o in unknown if o not in before and o not in fresh]
                if new:
                    at_once = keeper.config.removed_at_once()
                    seconds = 0 if at_once else keeper.config.removed_seconds()
                    done = await run_in_threadpool(keeper.hide, new, seconds)
                    if not at_once:
                        removal["hidden"] = len(done)
                        log.info(
                            "the root no longer has %d people: their files are hidden, and "
                            "deleted in %d days",
                            len(done),
                            seconds // 86400,
                        )
        # Whoever's date has passed is deleted, read fresh and compare-and-delete on the
        # marker seen: a marker restored or replaced since is left alone.
        now = keeper.now()
        due = await run_in_threadpool(keeper.removed_people)
        for owner, dates in sorted(due.items()):
            if dates["deleteAt"] > now:
                continue
            try:
                result = await run_in_threadpool(
                    functools.partial(
                        keeper.delete_removed, owner, marker=dates["marker"], fence=fence
                    )
                )
            except StoreProblem as exc:
                errors.append(exc.detail)
                log.warning("could not finish deleting %s: %s", owner, exc.detail)
                continue
            if result is not None:
                counts.add(result)
                removal["deleted"] += 1
                log.info("deleted everything a removed person kept here")
        if errors:
            removal["error"] = " ".join(errors)
        return removal

    async def delete_now(self, owner: str) -> dict[str, Any]:
        """The owner's Delete now: a removed person's objects, at once and for good."""
        async with self._lock:
            started = time.time()
            counts = await run_in_threadpool(self.keeper.delete_now, owner)
            removal = {"checked": 0, "hidden": 0, "restored": 0, "deleted": 1, "lateRestored": 0}
            report = report_of(counts, at=started, removal=removal)
            await run_in_threadpool(self.keeper.save_report, report)
            log.info("the owner deleted everything a removed person kept here")
            return report
