"""Fail-closed offline readiness checks for closed-bar incremental scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
from time import perf_counter

import numpy as np
import pandas as pd

from .io import BAR_COLUMNS, DataContractError, resolve_project_reference, sha256
from .validation import clock_minutes, normalize_minutes


FROZEN_SCORING_ROLES = {
    "feature_scalers",
    "baseline_models",
    "baseline_thresholds",
    "lstm_checkpoint",
    "scoring_thresholds",
}


@dataclass(frozen=True)
class ClosedBarReadiness:
    bars: pd.DataFrame
    status: dict
    frozen_resources: dict


@dataclass(frozen=True)
class IncrementalScoreResult:
    scores: pd.DataFrame
    timing: dict
    frozen_resources: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{name} must be a valid UTC timestamp") from exc
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise DataContractError(f"{name} must explicitly use UTC")
    return stamp


def verify_frozen_scoring_manifest(project_root, manifest_path) -> dict:
    """Verify the exact local-only scaler/model/threshold files before scoring."""
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path)
    manifest_file = manifest_file.resolve() if manifest_file.is_absolute() else (root / manifest_file).resolve()
    if not manifest_file.is_relative_to(root) or not manifest_file.is_file():
        raise DataContractError("Frozen scoring manifest must stay inside the project")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataContractError("Frozen scoring manifest is unreadable") from exc
    policy = manifest.get("data_policy", {})
    if (manifest.get("schema_version") != 1 or policy.get("classification") != "local_only_real_data"
            or policy.get("source_rights_status") != "pending" or policy.get("distributable") is not False):
        raise DataContractError("Frozen scoring manifest must preserve local-only pending-rights policy")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise DataContractError("Frozen scoring manifest files must be a list")
    selected = {}
    for item in files:
        role = item.get("role")
        if role not in FROZEN_SCORING_ROLES:
            continue
        relative, expected = item.get("path"), item.get("sha256")
        if role in selected or not isinstance(relative, str) or not isinstance(expected, str):
            raise DataContractError("Frozen scoring roles must be unique with path and sha256")
        if Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
            raise DataContractError("Frozen scoring artifact path must be relative")
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise DataContractError(f"Frozen scoring artifact is missing or unsafe: {role}")
        actual = _sha256(artifact)
        if actual != expected.lower():
            raise DataContractError(f"Frozen scoring artifact checksum mismatch: {role}")
        selected[role] = {"path": relative.replace("\\", "/"), "sha256": actual}
    missing = sorted(FROZEN_SCORING_ROLES - set(selected))
    if missing:
        raise DataContractError("Frozen scoring manifest missing roles: " + ", ".join(missing))
    return {
        "manifest_path": str(manifest_file.relative_to(root)).replace("\\", "/"),
        "manifest_sha256": _sha256(manifest_file),
        "source_rights_status": "pending",
        "distributable": False,
        "resources": selected,
    }


def _scheduled_bar_ends(calendar: dict, day: str, cutoff: pd.Timestamp) -> tuple[list[pd.Timestamp], bool]:
    timezone = calendar.get("timezone")
    if timezone != "Asia/Ho_Chi_Minh":
        raise DataContractError("Incremental calendar timezone must be Asia/Ho_Chi_Minh")
    midnight = pd.Timestamp(day, tz=timezone)
    local_cutoff = cutoff.tz_convert(timezone)
    ends = []
    active = False
    sessions = calendar.get("overrides", {}).get(day, calendar.get("sessions", []))
    for session in sessions:
        start, end = clock_minutes(session["start"]), clock_minutes(session["end"])
        local_minute = local_cutoff.hour * 60 + local_cutoff.minute + local_cutoff.second / 60
        active = active or start <= local_minute <= end
        for minute in range(start + 5, end + 1, 5):
            stamp = (midnight + pd.Timedelta(minutes=minute)).tz_convert("UTC")
            if stamp <= cutoff:
                ends.append(stamp)
    return sorted(ends), active


def closed_bar_readiness(minutes: pd.DataFrame, profile: dict, *, symbols, observed_at_utc,
                         provider_delay_seconds, stale_after_seconds, provider_delay_evidence,
                         project_root, manifest_path="demo/manifest.local.json",
                         delay_measured_live=False) -> ClosedBarReadiness:
    """Return only causally closed 5m bars plus gap/stale and frozen-artifact status.

    No delay is inferred. The caller must provide both a delay and its evidence label;
    `live_update_allowed` stays false until the delay is explicitly marked measured live.
    """
    if (isinstance(provider_delay_seconds, bool) or not isinstance(provider_delay_seconds, (int, float))
            or not 0 <= provider_delay_seconds <= 3600):
        raise DataContractError("provider_delay_seconds must be explicitly set to 0..3600")
    if (isinstance(stale_after_seconds, bool) or not isinstance(stale_after_seconds, (int, float))
            or not provider_delay_seconds <= stale_after_seconds <= 86400):
        raise DataContractError("stale_after_seconds must be explicit and at least the provider delay")
    if not isinstance(provider_delay_evidence, str) or not provider_delay_evidence.strip():
        raise DataContractError("provider_delay_evidence is required; delay is never inferred")
    if not isinstance(delay_measured_live, bool):
        raise DataContractError("delay_measured_live must be boolean")
    symbols = list(symbols)
    if not symbols or len(symbols) != len(set(symbols)) or any(not isinstance(s, str) or not s.isalnum() or s != s.upper() for s in symbols):
        raise DataContractError("symbols must be unique uppercase identifiers")

    frozen = verify_frozen_scoring_manifest(project_root, manifest_path)
    observed = _utc(observed_at_utc, "observed_at_utc")
    cutoff = observed - pd.Timedelta(seconds=provider_delay_seconds)
    calendar = profile.get("calendar", {})
    timezone = calendar.get("timezone")
    if not timezone:
        raise DataContractError("Incremental profile requires a calendar timezone")
    day = cutoff.tz_convert(timezone).date().isoformat()
    common = {
        "observed_at_utc": observed.isoformat(),
        "eligibility_cutoff_utc": cutoff.isoformat(),
        "provider_delay_seconds": provider_delay_seconds,
        "provider_delay_evidence": provider_delay_evidence,
        "delay_measured_live": delay_measured_live,
        "stale_after_seconds": stale_after_seconds,
        "calendar_date": day,
        "calendar_status": calendar.get("status"),
        "source_rights_status": frozen["source_rights_status"],
    }
    if day not in set(calendar.get("dates", [])):
        status = dict(common, state="BLOCKED_UNKNOWN_CALENDAR", calendar_date_known=False,
                      scoring_allowed=False, live_update_allowed=False, gap_count=0,
                      missing_symbols=symbols, latest_bar_end_utc=None, stale=False)
        return ClosedBarReadiness(pd.DataFrame(), status, frozen)

    expected_ends, session_active = _scheduled_bar_ends(calendar, day, cutoff)
    if minutes.empty:
        day_frame = minutes.copy()
    else:
        stamps = pd.to_datetime(minutes.timestamp, utc=True, errors="raise")
        day_frame = minutes.loc[stamps.dt.tz_convert(timezone).dt.date.astype(str).eq(day)].copy()
    present_symbols = set(day_frame.symbol) if "symbol" in day_frame else set()
    missing_symbols = sorted(set(symbols) - present_symbols)
    if missing_symbols:
        state = "WAITING_SYMBOLS" if expected_ends else ("WAITING_FIRST_CLOSED_BAR" if session_active else "OUTSIDE_SESSION")
        status = dict(common, state=state, calendar_date_known=True, session_active=session_active,
                      scoring_allowed=False, live_update_allowed=False,
                      expected_latest_bar_end_utc=expected_ends[-1].isoformat() if expected_ends else None,
                      latest_bar_end_utc=None, gap_count=len(expected_ends) * len(symbols),
                      missing_symbols=missing_symbols, stale=bool(session_active and expected_ends))
        return ClosedBarReadiness(pd.DataFrame(), status, frozen)

    validation = normalize_minutes(day_frame, profile, start=day, end=day, symbols=symbols, as_of=cutoff)
    bars = validation.bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    expected = {(symbol, stamp) for symbol in symbols for stamp in expected_ends}
    accepted = set(zip(bars.symbol, bars.timestamp))
    gaps = sorted(expected - accepted, key=lambda item: (item[0], item[1]))
    latest = {symbol: (group.timestamp.max() if not group.empty else None)
              for symbol, group in bars.groupby("symbol")}
    for symbol in symbols:
        latest.setdefault(symbol, None)
    expected_latest = expected_ends[-1] if expected_ends else None
    stale_symbols = []
    if session_active and expected_latest is not None:
        for symbol in symbols:
            value = latest[symbol]
            age = None if value is None else (observed - value).total_seconds()
            if value is None or value < expected_latest or age > stale_after_seconds:
                stale_symbols.append(symbol)
    latest_values = [value for value in latest.values() if value is not None]
    latest_bar = max(latest_values) if latest_values else None
    complete_latest = expected_latest is not None and all(latest[symbol] == expected_latest for symbol in symbols)
    scoring_allowed = bool(complete_latest and not stale_symbols)
    if stale_symbols:
        state = "STALE"
    elif not expected_ends:
        state = "WAITING_FIRST_CLOSED_BAR" if session_active else "OUTSIDE_SESSION"
    elif gaps:
        state = "READY_WITH_GAPS" if scoring_allowed else "GAPS_BLOCK_LATEST"
    else:
        state = "READY" if session_active else "READY_OUTSIDE_SESSION"
    status = dict(common, state=state, calendar_date_known=True, session_active=session_active,
                  scoring_allowed=scoring_allowed,
                  live_update_allowed=bool(scoring_allowed and delay_measured_live),
                  expected_latest_bar_end_utc=expected_latest.isoformat() if expected_latest is not None else None,
                  latest_bar_end_utc=latest_bar.isoformat() if latest_bar is not None else None,
                  latest_by_symbol={key: value.isoformat() if value is not None else None for key, value in latest.items()},
                  latest_age_seconds=None if latest_bar is None else (observed - latest_bar).total_seconds(),
                  gap_count=len(gaps),
                  gaps=[{"symbol": symbol, "bar_end_utc": stamp.isoformat()} for symbol, stamp in gaps],
                  missing_symbols=[], stale=bool(stale_symbols), stale_symbols=stale_symbols,
                  validation_rejected_rows=validation.report["rejected_source_rows"],
                  validation_rejected_bars=validation.report["rejected_bar_groups"])
    return ClosedBarReadiness(bars, status, frozen)


def _checked_scoring_directories(root: Path, frozen: dict) -> tuple[Path, Path, Path]:
    resources = frozen["resources"]
    feature_dir = (root / resources["feature_scalers"]["path"]).resolve().parent
    baseline_dir = (root / resources["baseline_models"]["path"]).resolve().parent
    scoring_dir = (root / resources["scoring_thresholds"]["path"]).resolve().parent
    lstm_dir = (root / resources["lstm_checkpoint"]["path"]).resolve().parent
    if (root / resources["baseline_thresholds"]["path"]).resolve().parent != baseline_dir:
        raise DataContractError("Frozen baseline models and thresholds must share one bundle")
    baseline = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))
    scoring = json.loads((scoring_dir / "metadata.json").read_text(encoding="utf-8"))
    if (resolve_project_reference(baseline.get("input_feature_dir", ""), root) != feature_dir
            or resolve_project_reference(scoring.get("input_feature_dir", ""), root) != feature_dir
            or resolve_project_reference(scoring.get("input_lstm_dir", ""), root) != lstm_dir):
        raise DataContractError("Frozen scoring paths do not match artifact provenance")
    return feature_dir, baseline_dir, scoring_dir


def score_ready_closed_bars(readiness: ClosedBarReadiness, historical_bars: pd.DataFrame, profile: dict, *,
                            project_root, received_at_utc, manifest_path="demo/manifest.local.json") \
        -> IncrementalScoreResult:
    """Score the latest readiness-approved endpoints with frozen artifacts only.

    `received_at_utc` is recorded as input provenance. Inference duration uses a
    separate monotonic clock and is not represented as provider/live latency.
    """
    if not isinstance(readiness, ClosedBarReadiness) or readiness.status.get("scoring_allowed") is not True:
        raise DataContractError("Incremental scoring requires readiness-approved closed bars")
    root = Path(project_root).resolve()
    verified = verify_frozen_scoring_manifest(root, manifest_path)
    if verified != readiness.frozen_resources:
        raise DataContractError("Frozen scoring manifest changed after readiness approval")
    received = _utc(received_at_utc, "received_at_utc")
    cutoff = _utc(readiness.status.get("eligibility_cutoff_utc"), "eligibility_cutoff_utc")
    latest_by_symbol = {symbol: _utc(value, f"latest_by_symbol[{symbol}]")
                        for symbol, value in readiness.status.get("latest_by_symbol", {}).items()}
    if not latest_by_symbol or readiness.bars.empty:
        raise DataContractError("Readiness approval must include closed bars and latest endpoints")
    if received < max(latest_by_symbol.values()):
        raise DataContractError("received_at_utc cannot precede the latest closed bar")

    symbols = set(latest_by_symbol)
    for label, frame in (("readiness", readiness.bars), ("historical", historical_bars)):
        if set(BAR_COLUMNS) - set(frame.columns):
            raise DataContractError(f"{label} bars do not follow the standard bar contract")
        stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if (stamps > cutoff).any():
            raise DataContractError(f"{label} bars contain data beyond the readiness cutoff")
        if not set(frame["symbol"]).issubset(symbols):
            raise DataContractError(f"{label} bars contain symbols outside the readiness scope")
    ready_latest = readiness.bars.groupby("symbol")["timestamp"].max().to_dict()
    if ready_latest != latest_by_symbol:
        raise DataContractError("Readiness bars do not match the approved latest endpoints")

    combined = pd.concat([historical_bars[list(BAR_COLUMNS)], readiness.bars[list(BAR_COLUMNS)]],
                         ignore_index=True)
    duplicate_rows = combined.loc[combined.duplicated(["symbol", "timestamp"], keep=False)]
    for _, rows in duplicate_rows.groupby(["symbol", "timestamp"]):
        if len(rows.drop_duplicates(list(BAR_COLUMNS))) != 1:
            raise DataContractError("Historical and readiness bars conflict at the same timestamp")
    combined = combined.drop_duplicates(["symbol", "timestamp"], keep="last")

    feature_dir, baseline_dir, scoring_dir = _checked_scoring_directories(root, verified)
    feature_metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    config_path = root / "configs" / ("research.yaml" if feature_metadata["source_profile_mode"] == "research"
                                      else "default.yaml")
    if sha256(config_path) != feature_metadata.get("config_sha256"):
        raise DataContractError("Frozen feature configuration checksum mismatch")
    from .config import load_config
    from .features import compute_features
    from .services import frozen_model_resources
    from .windows import make_windows, transform_features
    from .baselines import score_baselines
    from .scoring import score_reconstructions
    from .training import reconstruct

    settings = load_config(config_path)
    started_wall = datetime.now(timezone.utc)
    started_clock = perf_counter()
    resources = frozen_model_resources(baseline_dir, scoring_dir)
    features = compute_features(
        combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True), profile["calendar"],
        history_sessions=settings["features"]["volume_history_sessions"],
        min_history_sessions=settings["features"]["volume_min_history_sessions"],
        warmup_sessions=settings["features"]["warmup_sessions"],
    ).frame
    features["split"] = "test"
    scaled = transform_features(features, resources["scalers"])
    windows = make_windows(scaled, window_length=feature_metadata["window_length"])
    endpoint_mask = np.array([
        symbol in latest_by_symbol and timestamp == latest_by_symbol[symbol]
        for symbol, timestamp in zip(windows.index["symbol"], windows.index["timestamp"])
    ], dtype=bool)
    selected_values = windows.values[endpoint_mask]
    selected_index = windows.index.loc[endpoint_mask].reset_index(drop=True)
    selected_index["window_id"] = np.arange(len(selected_index))
    if set(selected_index["symbol"]) != symbols or selected_index["symbol"].duplicated().any():
        raise DataContractError("Every approved symbol requires exactly one contiguous frozen-feature window")

    baseline_scores = score_baselines(resources["baseline"], selected_values, selected_index)
    baseline_thresholds = json.loads((baseline_dir / "thresholds.json").read_text(encoding="utf-8"))
    lstm_thresholds = json.loads((scoring_dir / "thresholds.json").read_text(encoding="utf-8"))
    lstm_scores = score_reconstructions(
        selected_values, reconstruct(resources["lstm"], selected_values), selected_index,
        feature_metadata["feature_order"], expected_feature_order=feature_metadata["feature_order"])
    result = baseline_scores[["symbol", "timestamp", "window_start", "robust_zscore",
                              "isolation_forest"]].copy()
    result["lstm"] = lstm_scores["score"]
    for model in ("robust_zscore", "isolation_forest"):
        result[f"{model}_threshold"] = result["symbol"].map(
            lambda symbol: baseline_thresholds[symbol][model]["threshold"])
        result[f"{model}_alert"] = result[model] > result[f"{model}_threshold"]
    result["lstm_threshold"] = result["symbol"].map(lambda symbol: lstm_thresholds[symbol]["threshold"])
    result["lstm_alert"] = result["lstm"] > result["lstm_threshold"]
    finished_wall = datetime.now(timezone.utc)
    timing = {
        "received_at_utc": received.isoformat(),
        "score_started_at_utc": started_wall.isoformat(),
        "score_finished_at_utc": finished_wall.isoformat(),
        "score_duration_ms": round((perf_counter() - started_clock) * 1000, 3),
        "latency_classification": "offline_process_timing_not_live_provider_latency",
    }
    return IncrementalScoreResult(result.sort_values("symbol").reset_index(drop=True), timing, verified)
