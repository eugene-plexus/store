"""One HTTP client per instance, and one clock for durations."""

from __future__ import annotations

from pathlib import Path

from eugene_plexus_store import _http
from eugene_plexus_store.removal import PeopleCheck


def test_the_people_check_keeps_one_client_and_no_proxy() -> None:
    check = PeopleCheck("http://127.0.0.1:8079", "token")
    first = check._client
    assert check._client is first
    assert first.trust_env is False
    assert _http.is_internal("http://127.0.0.1:8079")


def test_no_duration_in_this_component_is_measured_with_monotonic() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_store"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "_generated" not in path.parts and "time.monotonic()" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these measure with monotonic(): {offenders}"
