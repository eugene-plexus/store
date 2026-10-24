"""Runtime configuration: the shared config trio, for the install's store.

Settings -> Store in the console is this schema, rendered by the generic
config page. Every field takes effect on the next request or the next
sweep; only the log level needs a restart.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

import yaml

from . import _private_files
from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)

log = logging.getLogger(__name__)

GIB = 1024 * 1024 * 1024
DAY = 24 * 3600

CATEGORY_LABELS: dict[str, str] = {
    "files": "Files",
    "quotas": "Space per owner",
    "people": "Removed people",
    "logging": "Logging",
}

FIELDS: list[ConfigField] = [
    ConfigField(
        key="filesEnabled",
        label="Keep files",
        description=(
            "Apps can upload files and use them in chats (`/v1/files` and `/v1/uploads`, "
            "as OpenAI's and Anthropic's apps expect).\n\n"
            "Off: those requests are refused with a message that says the owner turned "
            "files off. Files already kept stay until they end."
        ),
        category="files",
        valueType=ConfigValueType.boolean,
        default=True,
    ),
    ConfigField(
        key="fileDefaultDays",
        label="Delete a file after (days)",
        description=(
            "How long a file is kept when the app that uploaded it does not say, then "
            "deleted for good.\n\n"
            "0 keeps it until it is deleted, as OpenAI does. An app can always ask for an "
            "end of its own (OpenAI's `expires_after`, one hour to 30 days)."
        ),
        category="files",
        valueType=ConfigValueType.integer,
        default=0,
        minimum=0,
        maximum=3650,
    ),
    ConfigField(
        key="ownerQuotaGiB",
        label="Space per owner (GiB)",
        description=(
            "The most one person, app key or console may keep, files and uploads in "
            "progress together. An upload past it is refused with the numbers."
        ),
        category="quotas",
        valueType=ConfigValueType.integer,
        default=10,
        minimum=1,
        maximum=100000,
    ),
    ConfigField(
        key="ownerQuotaFiles",
        label="Files per owner",
        description="The most files one person, app key or console may keep.",
        category="quotas",
        valueType=ConfigValueType.integer,
        default=10000,
        minimum=1,
        maximum=1000000,
    ),
    ConfigField(
        key="sweepIntervalMinutes",
        label="Delete what has ended every (minutes)",
        description=(
            "How often ended files and uploads are deleted, and the root is asked which "
            "people were removed. An ended file can no longer be read in between."
        ),
        category="people",
        valueType=ConfigValueType.integer,
        default=60,
        minimum=1,
        maximum=1440,
    ),
    ConfigField(
        key="removalBrake",
        label="Hold when this many are removed at once",
        description=(
            "When the root no longer knows this many people or more at one check, nothing "
            "of theirs is deleted until you release them here. It guards against a "
            "restored or wrong root emptying the store. 0 never holds."
        ),
        category="people",
        valueType=ConfigValueType.integer,
        default=3,
        minimum=0,
        maximum=1000,
    ),
    ConfigField(
        key="logLevel",
        label="Log level",
        description="How chatty this process's log is.",
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
]
_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def as_schema(*, pending: dict[str, Any] | None = None) -> ConfigSchema:
    fields: list[ConfigField] = []
    for field in FIELDS:
        if pending and field.key in pending:
            fields.append(
                field.model_copy(update={"pendingRestart": True, "inEffect": pending[field.key]})
            )
        else:
            fields.append(field)
    return ConfigSchema(component="store", fields=fields, categories=CATEGORY_LABELS)


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    """None if valid, otherwise why not."""
    if value is None:
        return None
    vt = field.valueType
    if vt == ConfigValueType.boolean:
        return None if isinstance(value, bool) else f"expected true or false, got {value!r}"
    if vt == ConfigValueType.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if not math.isfinite(value):
            return "must be a finite number"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum:g}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum:g}"
        return None
    if vt == ConfigValueType.enum:
        if not isinstance(value, str) or value not in (field.enumValues or []):
            return f"must be one of {field.enumValues}"
        return None
    return f"unsupported valueType: {vt}"


class ConfigStore:
    """File-backed config. A file that does not load leaves the defaults in
    effect and the error on `/healthz`; the endpoints stay reachable."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = _defaults()
        self._started: dict[str, Any] = dict(self._values)

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for key, value in raw.items():
                    field = _FIELDS_BY_KEY.get(key)
                    if field is None:
                        continue
                    if value is None:
                        continue
                    problem = _validate_value(field, value)
                    if problem is not None:
                        raise ValueError(f"config field {key!r} in {self._path}: {problem}")
                    merged[key] = value
                self._values = merged
            else:
                self._values = _defaults()
            self._started = dict(self._values)

    def as_document(self) -> ConfigDocument:
        with self._lock:
            return ConfigDocument.model_validate(dict(self._values))

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        pending: list[str] = []
        patch: dict[str, Any] = request.model_dump()
        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue
                err = _validate_value(field, new_value)
                if err is not None:
                    rejected.append(ConfigFieldError(key=key, message=err))
                    continue
                self._values[key] = field.default if new_value is None else new_value
                applied.append(key)
                if field.requiresRestart and self._values.get(key) != self._started.get(key):
                    pending.append(key)
            if applied:
                self._write_locked()
            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=bool(pending),
                pendingRestart=pending,
            )

    def pending_restart(self) -> dict[str, Any]:
        with self._lock:
            return {
                f.key: self._started.get(f.key)
                for f in FIELDS
                if f.requiresRestart and self._values.get(f.key) != self._started.get(f.key)
            }

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    # -- what the rest of the store reads ---------------------------------

    def files_enabled(self) -> bool:
        return bool(self.get("filesEnabled"))

    def default_seconds(self) -> int | None:
        """A file's life when its app names none; None keeps it until deleted."""
        days = int(self.get("fileDefaultDays"))
        return days * DAY if days > 0 else None

    def quota_bytes(self) -> int:
        return int(self.get("ownerQuotaGiB")) * GIB

    def quota_files(self) -> int:
        return int(self.get("ownerQuotaFiles"))

    def sweep_seconds(self) -> float:
        return float(self.get("sweepIntervalMinutes")) * 60.0

    def removal_brake(self) -> int:
        return int(self.get("removalBrake"))

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _private_files.write_private_text(
            self._path,
            yaml.safe_dump(dict(self._values), sort_keys=True, default_flow_style=False),
        )
