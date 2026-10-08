"""Pure offline contracts for idempotent incremental minute ingestion.

This module does not fetch, aggregate, score, or claim live operation. Raw responses
remain immutable; the merge result is only a current-view index with revision audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time

import pandas as pd

from .io import DataContractError
from .providers import ProviderError, parse_vci_minutes


KEY_COLUMNS = ["symbol", "timestamp"]
VALUE_COLUMNS = ["source", "open", "high", "low", "close", "volume", "accumulated_volume"]
REQUIRED_COLUMNS = KEY_COLUMNS + ["retrieved_at_utc"] + VALUE_COLUMNS


@dataclass(frozen=True)
class IncrementalMergeResult:
    minutes: pd.DataFrame
    revisions: pd.DataFrame
    report: dict


def _utc_series(values: pd.Series, name: str) -> pd.Series:
    parsed = []
    for value in values:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"{name} must contain valid UTC timestamps") from exc
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise DataContractError(f"{name} must contain explicit UTC timestamps")
        parsed.append(stamp)
    return pd.to_datetime(parsed, utc=True)


def _same(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return left == right


def _prepare(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, int]:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DataContractError(f"{name} missing incremental columns: {', '.join(sorted(missing))}")
    result = frame.copy().reset_index(drop=True)
    result["timestamp"] = _utc_series(result["timestamp"], "timestamp")
    result["retrieved_at_utc"] = _utc_series(result["retrieved_at_utc"], "retrieved_at_utc")
    valid_symbol = result.symbol.map(lambda value: isinstance(value, str) and value.isalnum() and value == value.upper())
    if not valid_symbol.all() or not result.source.eq("VCI").all():
        raise DataContractError(f"{name} symbol/source differs from the configured VCI scope")

    duplicates_removed = 0
    kept = []
    for _, group in result.groupby(KEY_COLUMNS, sort=False, dropna=False):
        if len(group) == 1:
            kept.append(group.iloc[-1])
            continue
        for column in VALUE_COLUMNS:
            values = group[column].tolist()
            if any(not _same(values[0], value) for value in values[1:]):
                raise DataContractError(f"{name} has a conflicting duplicate incremental key")
        duplicates_removed += len(group) - 1
        kept.append(group.sort_values("retrieved_at_utc").iloc[-1])
    collapsed = pd.DataFrame(kept, columns=result.columns) if kept else result.iloc[0:0].copy()
    return collapsed.reset_index(drop=True), duplicates_removed


def merge_incremental_minutes(existing: pd.DataFrame, incoming: pd.DataFrame) -> IncrementalMergeResult:
    """Merge a newly retrieved VCI minute view without rollback or silent revision.

    A newer observation replaces the current value and emits a revision record when
    OHLCV changed. Older observations never roll state back. Repeating the same input
    is idempotent. Downstream validation remains responsible for market quality and
    closed-bar eligibility.
    """
    if set(existing.columns) != set(incoming.columns):
        raise DataContractError("Existing and incoming incremental schemas differ")
    current, existing_duplicates = _prepare(existing, "existing")
    update, incoming_duplicates = _prepare(incoming, "incoming")
    by_key = {(row.symbol, row.timestamp): row.copy() for _, row in current.iterrows()}
    revision_rows = []
    appended = refreshed = revisions_applied = stale_ignored = duplicates_ignored = 0

    for _, row in update.iterrows():
        key = (row.symbol, row.timestamp)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = row.copy()
            appended += 1
            continue
        changed = [column for column in VALUE_COLUMNS if not _same(previous[column], row[column])]
        if row.retrieved_at_utc < previous.retrieved_at_utc:
            stale_ignored += 1
            continue
        if row.retrieved_at_utc == previous.retrieved_at_utc:
            if changed:
                raise DataContractError("Equal retrieval timestamps contain conflicting source values")
            duplicates_ignored += 1
            continue
        by_key[key] = row.copy()
        if changed:
            record = {
                "symbol": row.symbol,
                "timestamp": row.timestamp,
                "previous_retrieved_at_utc": previous.retrieved_at_utc,
                "incoming_retrieved_at_utc": row.retrieved_at_utc,
                "changed_fields": "|".join(changed),
            }
            for column in VALUE_COLUMNS:
                record[f"previous_{column}"] = previous[column]
                record[f"incoming_{column}"] = row[column]
            revision_rows.append(record)
            revisions_applied += 1
        else:
            refreshed += 1

    minutes = pd.DataFrame(list(by_key.values()), columns=current.columns)
    if not minutes.empty:
        minutes = minutes.sort_values(KEY_COLUMNS).reset_index(drop=True)
    revisions = pd.DataFrame(revision_rows)
    report = {
        "existing_rows": len(existing),
        "incoming_rows": len(incoming),
        "existing_exact_duplicates_removed": existing_duplicates,
        "incoming_exact_duplicates_removed": incoming_duplicates,
        "appended_rows": appended,
        "refreshed_rows": refreshed,
        "revisions_applied": revisions_applied,
        "stale_observations_ignored": stale_ignored,
        "same_observations_ignored": duplicates_ignored,
        "output_rows": len(minutes),
        "state_changed": bool(appended or refreshed or revisions_applied),
    }
    return IncrementalMergeResult(minutes=minutes, revisions=revisions, report=report)


@dataclass(frozen=True)
class IncrementalStoreResult:
    minutes: pd.DataFrame
    revisions: pd.DataFrame
    report: dict
    pointer: dict


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise DataContractError(f"Immutable incremental artifact differs: {path.name}")
        return
    # A short, exclusively-created sibling avoids MAX_PATH failures on Windows
    # without global long-path settings, while retaining same-directory replace.
    descriptor, name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    """Publish a complete JSON document; tolerate brief Windows access conflicts.

    Three replacement attempts, at most150ms requested wait. Persistent access
    errors still propagate and the old document is never deleted first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(3):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                # Win32 access denied/sharing/lock conflict; do not retry generic
                # permission, disk, path, or serialization errors on other systems.
                code = getattr(exc, "winerror", None)
                if code not in {5, 32, 33} or attempt == 2:
                    raise
                logging.getLogger(__name__).warning(
                    "Atomic JSON Windows access error%s; retry%d/2", code, attempt + 1)
                time.sleep(0.05 * (2 ** attempt))
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DataContractError("Incremental state path escapes its store")
    return path


def _json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_incremental_current(store_root) -> tuple[pd.DataFrame, dict | None]:
    """Load and hash-check the atomically selected current view."""
    root = Path(store_root).resolve()
    pointer_path = root / "current.json"
    if not pointer_path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS), None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataContractError("Incremental current pointer is unreadable") from exc
    if pointer.get("schema_version") != 1 or pointer.get("mode") != "offline_incremental_store" or pointer.get("live_evidence") is not False:
        raise DataContractError("Incremental current pointer has an incompatible contract")
    manifest_path = _safe_relative(root, pointer.get("version_manifest", ""))
    view_path = _safe_relative(root, pointer.get("view_file", ""))
    if not manifest_path.is_file() or _sha256_bytes(manifest_path.read_bytes()) != pointer.get("version_manifest_sha256"):
        raise DataContractError("Incremental version manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("view_file") != pointer.get("view_file") or manifest.get("view_sha256") != pointer.get("view_sha256"):
        raise DataContractError("Incremental pointer and version manifest disagree")
    if not view_path.is_file() or _sha256_bytes(view_path.read_bytes()) != pointer.get("view_sha256"):
        raise DataContractError("Incremental current view checksum mismatch")
    frame = pd.read_csv(view_path)
    prepared, duplicates = _prepare(frame, "stored current view")
    if duplicates or len(prepared) != pointer.get("rows"):
        raise DataContractError("Incremental current view row contract differs")
    return prepared, pointer


def load_incremental_poll(store_root, poll_id: str, symbol: str) -> tuple[bytes, pd.Timestamp] | None:
    """Verify a durable poll's original bytes/receipt before any restarted fetch.

    A raw file without its receipt manifest cannot be safely recovered. Materialized
    versions also anchor the manifest hash, including orphan versions after a crash.
    """
    if not isinstance(poll_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", poll_id):
        raise DataContractError("poll_id must be a safe nonempty identifier")
    root = Path(store_root).resolve()
    directory = _safe_relative(root, f"polls/{poll_id}")
    _, pointer = load_incremental_current(root)
    if not directory.exists():
        if pointer and poll_id in pointer.get("applied_poll_ids", []):
            raise DataContractError("Applied incremental poll is missing")
        return None
    try:
        manifest_path = directory / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        expected = {"schema_version": 1, "kind": "immutable_incremental_raw_poll",
                    "poll_id": poll_id, "symbol": symbol, "source": "VCI",
                    "raw_file": f"polls/{poll_id}/response.json", "source_rights_status": "pending",
                    "live_evidence": False}
        if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
            raise DataContractError("Saved incremental poll manifest has an incompatible contract")
        raw_body = (directory / "response.json").read_bytes()
        if _sha256_bytes(raw_body) != manifest.get("raw_sha256"):
            raise DataContractError("Saved incremental poll raw checksum mismatch")
        retrieved = _utc_series(pd.Series([manifest.get("retrieved_at_utc")]), "retrieved_at_utc").item()
        incoming = parse_vci_minutes(json.loads(raw_body), symbol)
        incoming["retrieved_at_utc"] = retrieved
        incoming, _ = _prepare(incoming[REQUIRED_COLUMNS], "saved incremental poll")
        if type(manifest.get("rows")) is not int or len(incoming) != manifest["rows"]:
            raise DataContractError("Saved incremental poll row contract differs")
        # Existing immutable view identities bind the receipt manifest to its
        # historical hash, even if this poll is no longer the latest applied poll.
        for version_path in (root / "views").glob("*/manifest.json"):
            version = json.loads(version_path.read_bytes())
            if version.get("poll_id") != poll_id:
                continue
            view_path = _safe_relative(root, version["view_file"])
            view_bytes = view_path.read_bytes()
            identity_content = {key: value for key, value in version.items() if key != "view_file"}
            if (_sha256_bytes(_json_bytes(identity_content) + view_bytes) != version_path.parent.name
                    or version.get("poll_manifest_sha256") != _sha256_bytes(manifest_bytes)):
                raise DataContractError("Saved poll manifest differs from its immutable view provenance")
    except (OSError, ValueError, TypeError, KeyError, ProviderError) as exc:
        raise DataContractError("Saved incremental poll is incomplete or unreadable") from exc
    return raw_body, retrieved


def persist_incremental_poll(store_root, poll_id: str, raw_body: bytes, symbol: str,
                             retrieved_at_utc) -> IncrementalStoreResult:
    """Persist one raw VCI poll and atomically select its merged current view.

    This offline storage primitive never performs network I/O. Raw poll bytes and
    version directories are immutable; only the verified current pointer is replaced.
    """
    if not isinstance(poll_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", poll_id):
        raise DataContractError("poll_id must be a safe nonempty identifier")
    if not isinstance(raw_body, bytes):
        raise DataContractError("raw_body must be exact response bytes")
    root = Path(store_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    retrieved = _utc_series(pd.Series([retrieved_at_utc]), "retrieved_at_utc").item()
    try:
        payload = json.loads(raw_body)
    except (TypeError, ValueError) as exc:
        raise DataContractError("Incremental raw body is not valid JSON") from exc
    incoming = parse_vci_minutes(payload, symbol)
    incoming["retrieved_at_utc"] = retrieved
    incoming = incoming[REQUIRED_COLUMNS]
    incoming, _ = _prepare(incoming, "incremental poll")

    poll_directory = root / "polls" / poll_id
    raw_path = poll_directory / "response.json"
    raw_sha256 = _sha256_bytes(raw_body)
    _write_once(raw_path, raw_body)
    poll_manifest = {
        "schema_version": 1,
        "kind": "immutable_incremental_raw_poll",
        "poll_id": poll_id,
        "symbol": symbol,
        "source": "VCI",
        "retrieved_at_utc": retrieved.isoformat(),
        "raw_file": str(raw_path.relative_to(root)).replace("\\", "/"),
        "raw_sha256": raw_sha256,
        "rows": len(incoming),
        "source_rights_status": "pending",
        "live_evidence": False,
    }
    poll_manifest_path = poll_directory / "manifest.json"
    _write_once(poll_manifest_path, _json_bytes(poll_manifest))
    poll_manifest_sha256 = _sha256_bytes(poll_manifest_path.read_bytes())

    existing, previous_pointer = load_incremental_current(root)
    applied = [] if previous_pointer is None else list(previous_pointer.get("applied_poll_ids", []))
    if poll_id in applied:
        return IncrementalStoreResult(existing, pd.DataFrame(), {
            "already_applied": True, "state_changed": False, "output_rows": len(existing),
            "raw_sha256": raw_sha256,
        }, previous_pointer)

    merged = merge_incremental_minutes(existing, incoming)
    view_bytes = merged.minutes.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S%z").encode("utf-8")
    view_sha256 = _sha256_bytes(view_bytes)
    applied.append(poll_id)
    revisions = [{key: _json_value(value) for key, value in record.items()}
                 for record in merged.revisions.to_dict(orient="records")]
    version_manifest = {
        "schema_version": 1,
        "kind": "incremental_materialized_view",
        "poll_id": poll_id,
        "poll_manifest": str(poll_manifest_path.relative_to(root)).replace("\\", "/"),
        "poll_manifest_sha256": poll_manifest_sha256,
        "parent_pointer_sha256": None if previous_pointer is None else _sha256_bytes((root / "current.json").read_bytes()),
        "view_sha256": view_sha256,
        "rows": len(merged.minutes),
        "applied_poll_ids": applied,
        "merge_report": merged.report,
        "revisions": revisions,
        "source_rights_status": "pending",
        "live_evidence": False,
    }
    identity = _sha256_bytes(_json_bytes(version_manifest) + view_bytes)
    version_directory = root / "views" / identity
    view_path = version_directory / "minutes.csv"
    version_manifest["view_file"] = str(view_path.relative_to(root)).replace("\\", "/")
    version_manifest_path = version_directory / "manifest.json"
    _write_once(view_path, view_bytes)
    _write_once(version_manifest_path, _json_bytes(version_manifest))

    pointer = {
        "schema_version": 1,
        "mode": "offline_incremental_store",
        "live_evidence": False,
        "source_rights_status": "pending",
        "latest_poll_id": poll_id,
        "applied_poll_ids": applied,
        "version_manifest": str(version_manifest_path.relative_to(root)).replace("\\", "/"),
        "version_manifest_sha256": _sha256_bytes(version_manifest_path.read_bytes()),
        "view_file": version_manifest["view_file"],
        "view_sha256": view_sha256,
        "rows": len(merged.minutes),
    }
    _atomic_json(root / "current.json", pointer)
    report = dict(merged.report, already_applied=False, raw_sha256=raw_sha256,
                  version_id=identity, poll_manifest_sha256=poll_manifest_sha256)
    return IncrementalStoreResult(merged.minutes, merged.revisions, report, pointer)
