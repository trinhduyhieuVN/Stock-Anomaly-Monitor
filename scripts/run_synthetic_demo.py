"""Validate and summarize the small, distributable synthetic demo fixture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any


EXPECTED_ROLES = {"runner", "synthetic_fixture"}
REQUIRED_COLUMNS = {
    "symbol", "timestamp", "open", "high", "low", "close", "volume",
    "interval", "source", "session_id", "segment_id", "quality_status",
    "synthetic_case",
}
FORBIDDEN_PREFIXES = ("artifacts/", "data/raw/", "data/processed/", "reports/evaluation/")


class SyntheticDemoError(ValueError):
    """Raised when the distributable synthetic demo contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_member(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise SyntheticDemoError(f"absolute path is forbidden: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SyntheticDemoError(f"path escapes project root: {relative}") from exc
    return resolved


def verify_manifest(root: Path, manifest_path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("bundle_kind") != "synthetic_demo":
        raise SyntheticDemoError("manifest must be schema 1 and bundle_kind=synthetic_demo")
    policy = manifest.get("data_policy", {})
    expected_policy = {
        "classification": "synthetic_generated",
        "distributable": True,
        "contains_real_source_data": False,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise SyntheticDemoError(f"data_policy.{key} must be {expected!r}")

    files = manifest.get("files", [])
    roles = [item.get("role") for item in files]
    if set(roles) != EXPECTED_ROLES or len(roles) != len(EXPECTED_ROLES):
        raise SyntheticDemoError("manifest must contain exactly runner and synthetic_fixture")
    fixture_path: Path | None = None
    for item in files:
        relative = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise SyntheticDemoError("each file needs string path and sha256")
        normalized = relative.replace("\\", "/").lower()
        if normalized.startswith(FORBIDDEN_PREFIXES):
            raise SyntheticDemoError(f"real/local artifact path is forbidden: {relative}")
        path = resolve_member(root, relative)
        if not path.is_file() or sha256_file(path) != expected_hash.lower():
            raise SyntheticDemoError(f"missing file or sha256 mismatch: {relative}")
        if item["role"] == "synthetic_fixture":
            fixture_path = path
    assert fixture_path is not None
    return manifest, fixture_path


def validate_fixture(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise SyntheticDemoError("fixture is missing required columns")
        rows = list(reader)
    expected_rows = manifest.get("fixture", {}).get("row_count")
    if len(rows) != expected_rows:
        raise SyntheticDemoError(f"fixture row count must be {expected_rows}")

    by_symbol: dict[str, list[datetime]] = defaultdict(list)
    cases: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        if row["source"] != "SYNTHETIC" or row["interval"] != "5m":
            raise SyntheticDemoError(f"row {row_number}: source must be SYNTHETIC and interval 5m")
        if row["quality_status"] != "ok":
            raise SyntheticDemoError(f"row {row_number}: quality_status must be ok")
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        if timestamp.utcoffset() != timedelta(0):
            raise SyntheticDemoError(f"row {row_number}: timestamp must be UTC")
        by_symbol[row["symbol"]].append(timestamp)
        values = [float(row[name]) for name in ("open", "high", "low", "close")]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise SyntheticDemoError(f"row {row_number}: prices must be finite and positive")
        open_, high, low, close = values
        if high < max(open_, close) or low > min(open_, close) or high < low:
            raise SyntheticDemoError(f"row {row_number}: invalid OHLC relation")
        volume = int(row["volume"])
        if volume < 0 or str(volume) != row["volume"]:
            raise SyntheticDemoError(f"row {row_number}: volume must be a non-negative integer")
        cases[row["synthetic_case"]] += 1

    expected_symbols = set(manifest.get("fixture", {}).get("symbols", []))
    if set(by_symbol) != expected_symbols:
        raise SyntheticDemoError("fixture symbols do not match manifest")
    for symbol, timestamps in by_symbol.items():
        if timestamps != sorted(timestamps) or any(
            later - earlier != timedelta(minutes=5)
            for earlier, later in zip(timestamps, timestamps[1:])
        ):
            raise SyntheticDemoError(f"{symbol}: timestamps must be strictly contiguous 5m bars")
    if cases != Counter(manifest.get("fixture", {}).get("case_counts", {})):
        raise SyntheticDemoError("synthetic case counts do not match manifest")
    return {
        "rows": len(rows),
        "symbols": sorted(by_symbol),
        "bars_per_symbol": {symbol: len(values) for symbol, values in sorted(by_symbol.items())},
        "case_counts": dict(sorted(cases.items())),
        "first_timestamp": min(min(values) for values in by_symbol.values()).isoformat(),
        "last_timestamp": max(max(values) for values in by_symbol.values()).isoformat(),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest, fixture_path = verify_manifest(root, manifest_path)
    fixture = validate_fixture(fixture_path, manifest)
    return {
        "status": "passed",
        "criteria_met": False,
        "check": "prompt12_distributable_synthetic_fixture",
        "demo_id": manifest["demo_id"],
        "data_policy": manifest["data_policy"],
        "fixture": fixture,
        "environment": {
            "python": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "in_venv": sys.prefix != sys.base_prefix,
        },
        "note": "Synthetic fixture smoke only; not market data, model evaluation, or the final prompt 12 application demo.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("demo/manifest.synthetic.json"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    try:
        result = run(root, manifest_path)
    except (OSError, json.JSONDecodeError, SyntheticDemoError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    if args.report:
        report_path = args.report if args.report.is_absolute() else root / args.report
        write_json_atomic(report_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
