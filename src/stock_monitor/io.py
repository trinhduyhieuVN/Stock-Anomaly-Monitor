"""CSV/Parquet schema checks and immutable local snapshot acquisition.

This module does not convert raw VCI minutes to the standard bar contract.
Source normalization and complete market-data validation belong to prompt 4.
"""

import hashlib
import json
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import project_path

BAR_COLUMNS = (
    "symbol", "timestamp", "open", "high", "low", "close", "volume",
    "interval", "source", "session_id", "segment_id", "quality_status",
)
QUALITY_STATUSES = {"ok", "rejected", "unverified"}


class DataContractError(ValueError):
    """Input is not a standard CSV/Parquet bar table."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_project_reference(path, project_root):
    """Prefer the same artifact/data path inside a relocated project copy."""
    declared = Path(path)
    root = Path(project_root).resolve()
    lowered = [part.lower() for part in declared.parts]
    for marker in ("artifacts", "data"):
        if marker in lowered:
            candidate = (root / Path(*declared.parts[lowered.index(marker):])).resolve()
            if candidate.is_relative_to(root) and candidate.exists():
                return candidate
    return declared.resolve()


def read_table(path):
    """Load the declared schema without losing invalid rows before validation."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={c: "string" for c in
            ("symbol", "timestamp", "interval", "source", "session_id", "segment_id", "quality_status")})
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise DataContractError("Only .csv and .parquet standard bar files are supported")
    missing = sorted(set(BAR_COLUMNS) - set(frame.columns))
    if missing:
        raise DataContractError(f"Missing standard columns: {', '.join(missing)}")
    if frame.empty:
        raise DataContractError("No bars in the input file")
    return frame


def read_bars(path, *, symbols, interval="5m", source="VCI"):
    frame = read_table(path)
    for col in ("symbol", "interval", "source", "session_id", "segment_id", "quality_status"):
        if frame[col].isna().any() or not frame[col].map(lambda v: isinstance(v, str) and bool(v.strip()) and v == v.strip()).all():
            raise DataContractError(f"{col} must contain nonempty strings without surrounding spaces")
    if not set(frame.symbol).issubset(set(symbols)) or not frame.interval.eq(interval).all() or not frame.source.eq(source).all():
        raise DataContractError("Source, symbol or interval differs from the configuration")
    if not set(frame.quality_status).issubset(QUALITY_STATUSES):
        raise DataContractError("Unknown quality_status")
    timestamps = []
    for value in frame.timestamp:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise DataContractError("Invalid timestamp") from exc
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise DataContractError("timestamp must explicitly include UTC; naive/local timestamps are not inferred")
        timestamps.append(stamp)
    frame["timestamp"] = pd.to_datetime(timestamps, utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        if pd.api.types.is_bool_dtype(frame[col]):
            raise DataContractError(f"{col} must be numeric, not boolean")
        try:
            values = pd.to_numeric(frame[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"{col} must be numeric") from exc
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise DataContractError(f"{col} contains null or nonfinite values")
        if col == "volume" and ((values < 0).any() or (values % 1 != 0).any()):
            raise DataContractError("volume must contain nonnegative integer counts")
        frame[col] = values
    return frame


def _write_manifest_once(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise DataContractError("An existing immutable snapshot manifest differs")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    return path


def collect_probe_snapshot(settings):
    """Verify and register already-downloaded raw evidence; never fetch a network URL."""
    evidence = project_path(settings, "probe_manifest")
    manifest = json.loads(evidence.read_text(encoding="utf-8"))
    selected = []
    data = settings["data"]
    for symbol in data["symbols"]:
        candidates = [r for r in manifest.get("runs", []) if r.get("symbol") == symbol
            and r.get("requested_start") == data["start"] and r.get("requested_end") == data["end"]]
        if len(candidates) != 1:
            raise DataContractError(f"Expected one audited source snapshot for {symbol} and the exact date range")
        run = candidates[0]
        directory = project_path(settings, "raw_dir") / "vci" / run["run"]
        for file, expected in ((directory / "response_01.json", run["raw_sha256"]),
                               (directory / "metadata.json", run["metadata_sha256"])):
            if not file.resolve().is_relative_to(project_path(settings, "raw_dir")):
                raise DataContractError("Source snapshot path escapes the raw directory")
            if sha256(file) != expected:
                raise DataContractError(f"Source checksum mismatch: {symbol}/{file.name}")
        selected.append({"symbol": symbol, "raw_path": str((directory / "response_01.json").relative_to(settings["_root"])).replace("\\", "/"),
            "raw_sha256": run["raw_sha256"], "metadata_sha256": run["metadata_sha256"],
            "source_minutes": run["requested_range_minutes"], "observed_session_dates": run["session_dates"]})
    digest = sha256(evidence)
    snapshot = {"schema_version": 1, "kind": "raw_source_snapshot", "source": data["source"],
        "source_interval": data["source_interval"], "target_interval": data["interval"],
        "requested_start": data["start"], "requested_end": data["end"], "symbols": data["symbols"],
        "evidence_sha256": digest, "files": selected, "normalized": False}
    return _write_manifest_once(project_path(settings, "raw_dir") / "snapshots" / f"vci-{digest[:16]}", snapshot)


def import_bar_file(settings, path):
    """Read the declared schema, then preserve the supplied file byte-for-byte."""
    path = Path(path).resolve()
    frame = read_bars(path, symbols=settings["data"]["symbols"], interval=settings["data"]["interval"], source=settings["data"]["source"])
    digest = sha256(path)
    directory = project_path(settings, "raw_dir") / "imports" / digest
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"bars{path.suffix.lower()}"
    if destination.exists():
        if sha256(destination) != digest:
            raise DataContractError("Existing imported snapshot checksum mismatch")
    else:
        with path.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if sha256(destination) != digest:
            raise DataContractError("Imported snapshot checksum mismatch")
    return _write_manifest_once(directory, {"schema_version": 1, "kind": "standard_bar_import",
        "file": destination.name, "sha256": digest, "rows": len(frame),
        "source_units_verified": False, "market_quality_validated": False})
