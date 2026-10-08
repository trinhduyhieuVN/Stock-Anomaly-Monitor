"""Bounded history and a separate, observed-only research trial context."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import pandas as pd
import yaml

from .config import load_config, pending_confirmations
from .io import DataContractError, sha256
from .providers import ENDPOINT, fetch_vci_raw, load_snapshot, parse_vci_minutes
from .validation import load_profile, normalize_minutes


def _inside(root, value, folder):
    path = (root / value).resolve()
    if not path.is_relative_to(root / folder):
        raise DataContractError(f"Trial input must stay inside {folder}")
    return path


def _json_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def history_plan(project_root, symbol, start, end, *, now):
    root = Path(project_root).resolve()
    settings = load_config(root / "configs/research.yaml")
    profile = load_profile(settings)
    current = pd.Timestamp(now)
    if pd.isna(current) or current.tzinfo is None:
        raise DataContractError("History planning requires a timezone-aware clock")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if (first.tzinfo is not None or last.tzinfo is not None or pd.isna(first) or pd.isna(last)
            or first.isoformat()[:10] != start or last.isoformat()[:10] != end
            or first > last or (last - first).days > 6
            or start <= settings["data"]["end"]
            or end >= current.tz_convert("Asia/Ho_Chi_Minh").date().isoformat()):
        raise DataContractError("Use at most seven fully historical dates after frozen history")
    if symbol not in settings["data"]["symbols"] or not profile["research"]:
        raise DataContractError("History symbol/profile is outside the authorized research scope")
    return {"symbol": symbol, "start": start, "end": end, "count_back": 500,
            "max_attempts": 1, "timeout": 10, "source_rights_status": "pending",
            "source_confirmations_pending": pending_confirmations(settings), "scoring_called": False}


def fetch_trial_history(project_root, symbol, start, end, *, transport,
                        now=None):
    root = Path(project_root).resolve()
    plan = history_plan(root, symbol, start, end, now=now or datetime.now(timezone.utc))
    raw, metadata = fetch_vci_raw(root / "data/raw", symbol, start, end, plan["count_back"],
                                 transport=transport, max_attempts=1, timeout=10)
    snapshot = {"kind": "raw_source_snapshot", "source": "VCI", "source_interval": "1m",
                "symbols": [symbol], "requested_start": start, "requested_end": end,
                "source_rights_status": "pending", "distributable": False,
                "files": [{"symbol": symbol, "raw_path": raw.relative_to(root).as_posix(),
                           "raw_sha256": sha256(raw), "metadata_path": metadata.relative_to(root).as_posix(),
                           "metadata_sha256": sha256(metadata)}]}
    path = raw.parent / "trial_snapshot.json"
    _json_new(path, snapshot)
    return path


def _observation(root, report_path, report_sha256, symbol):
    report_path = _inside(root, report_path, "reports")
    if sha256(report_path) != report_sha256:
        raise DataContractError("Observation report checksum mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = report["source_observation"]
    if (report.get("status") != "PASS" or meta.get("http_status") != 200
            or meta.get("endpoint") != ENDPOINT or meta.get("symbol") != symbol
            or meta.get("kind") != "source_observation_not_live_trial"
            or meta.get("query", {}).get("symbols") != [symbol]
            or meta.get("query", {}).get("timeFrame") != "ONE_MINUTE"
            or meta.get("source_rights_status") != "pending"):
        raise DataContractError("Observation source/scope/status mismatch")
    raw = _inside(root, meta["raw_path"], "data/raw")
    metadata = raw.with_name("metadata.json")
    pins = {item["path"]: item["sha256"] for item in report["artifacts"]}
    for path in (raw, metadata):
        if pins.get(path.relative_to(root).as_posix()) != sha256(path):
            raise DataContractError("Observation raw/metadata checksum mismatch")
    if json.loads(metadata.read_text(encoding="utf-8")) != meta or sha256(raw) != meta["raw_sha256"]:
        raise DataContractError("Observation metadata differs from pinned report")
    frame = parse_vci_minutes(json.loads(raw.read_bytes()), symbol)
    receipt = pd.Timestamp(meta["received_at_utc"])
    requested = pd.Timestamp(meta["request_started_at_utc"])
    if (receipt.tzinfo is None or requested.tzinfo is None or receipt < requested
            or frame.timestamp.isna().any() or frame.timestamp.gt(receipt).any()
            or receipt.tz_convert("Asia/Ho_Chi_Minh").date().isoformat() != meta["expected_local_date"]):
        raise DataContractError("Observation clock or timestamps are invalid")
    return frame, meta, report_path


def prepare_trial_history(project_root, snapshot_path, *, observation_report=None,
                          observation_sha256=None):
    """Verify inputs, derive dates from accepted bars, publish a fresh local bundle.

    No network or scoring. Missing days remain unknown, not declared holidays.
    """
    root = Path(project_root).resolve()
    settings = load_config(root / "configs/research.yaml")
    profile = load_profile(settings)
    snapshot_path = _inside(root, snapshot_path, "data/raw")
    minutes, snapshot, provenance = load_snapshot(snapshot_path, root)
    symbols = snapshot["symbols"]
    start, end = snapshot["requested_start"], snapshot["requested_end"]
    if (len(symbols) != 1 or symbols[0] not in settings["data"]["symbols"]
            or start <= settings["data"]["end"] or start > end or not profile["research"]):
        raise DataContractError("Trial history must extend frozen history in the approved symbol scope")
    if any(not 1 <= item["query"].get("countBack", 0) <= 500 for item in provenance):
        raise DataContractError("Trial history exceeds the 500-minute request budget")
    if minutes.timestamp.isna().any() or minutes.empty:
        raise DataContractError("Trial history contains invalid or no timestamps")
    cutoff = pd.Timestamp(end, tz="Asia/Ho_Chi_Minh") + pd.Timedelta(days=1)
    if minutes.timestamp.ge(cutoff).any():
        raise DataContractError("Historical response contains minutes beyond its requested cutoff")
    dates = minutes.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh").dt.strftime("%Y-%m-%d")
    candidate_days = sorted(set(dates.loc[dates.between(start, end)]))
    if not candidate_days:
        raise DataContractError("No history rows in the requested date range")
    calendar = deepcopy(profile["calendar"])
    calendar.update(status="observed_only", complete=False,
                    dates=sorted(set(calendar["dates"]) | set(candidate_days)))
    profile["calendar"] = calendar
    result = normalize_minutes(minutes, profile, start=start, end=end, symbols=symbols, as_of=cutoff)
    accepted_days = sorted(set(result.bars.session_id))
    if accepted_days != candidate_days:
        raise DataContractError("An observed date has no quality-approved closed bar")
    evidence = [{"path": snapshot_path.relative_to(root).as_posix(), "sha256": sha256(snapshot_path)}]
    observation_check = None
    if observation_report is not None:
        observed, meta, report_path = _observation(root, observation_report, observation_sha256, symbols[0])
        day = meta["expected_local_date"]
        if day <= end:
            raise DataContractError("Current-date observation must follow the historical extension")
        calendar["dates"] = sorted(set(calendar["dates"]) | {day})
        candidate = normalize_minutes(observed, profile, start=day, end=day, symbols=symbols,
                                      as_of=meta["received_at_utc"])
        if candidate.bars.empty:
            raise DataContractError("Observation date has no quality-approved closed bar")
        observation_check = {"date": day, "receipt": meta["received_at_utc"],
                             "accepted_closed_bars": len(candidate.bars), "validation": candidate.report}
        evidence.append({"path": report_path.relative_to(root).as_posix(), "sha256": sha256(report_path)})
    added_dates = sorted(set(calendar["dates"]) - set(load_profile(settings)["calendar"]["dates"]))
    calendar.update(trial_symbol_scope=symbols, trial_evidence=evidence)
    directory = root / "reports/trials" / ("history_" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    calendar_path = directory / "observed_calendar.json"
    _json_new(calendar_path, calendar)
    config = {key: deepcopy(value) for key, value in settings.items() if not key.startswith("_")}
    config["project"]["root"] = os.path.relpath(root, directory).replace("\\", "/")
    config["data"]["end"] = max(calendar["dates"])
    config["data"]["research_profile"]["calendar_file"] = calendar_path.relative_to(root).as_posix()
    config["data"]["research_profile"]["evidence"]["trial_context"] = (directory / "manifest.json").relative_to(root).as_posix()
    config["data"]["http"].update(max_attempts=1, count_back=500, timeout=10)
    with (directory / "research.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    result.bars.to_parquet(directory / "history_extension.parquet", index=False)
    result.rejected_bars.to_parquet(directory / "history_rejected_bars.parquet", index=False)
    result.rejected_rows.to_parquet(directory / "history_rejected_rows.parquet", index=False)
    _json_new(directory / "validation.json", result.report)
    loaded = load_profile(load_config(directory / "research.yaml"))
    assert loaded["calendar"] == calendar and loaded["confirmations_pending"] == profile["confirmations_pending"]
    complete = result.report["missing_bar_groups"] == 0 and result.report["rejected_bar_groups"] == 0
    output = {"schema_version": 1, "kind": "local_research_trial_history", "criteria_met": False,
              "source_rights_status": "pending", "distributable": False, "symbol_scope": symbols,
              "source_confirmations_pending": pending_confirmations(settings),
              "history_start": start, "history_end": end, "accepted_bars": len(result.bars),
              "observed_history_dates": accepted_days, "added_calendar_dates": added_dates,
              "coverage_of_observed_history_dates_complete": complete,
              "calendar_complete": False, "observation_check": observation_check,
              "config_path": (directory / "research.yaml").relative_to(root).as_posix(),
              "history_path": (directory / "history_extension.parquet").relative_to(root).as_posix(),
              "inputs": evidence, "provenance": provenance,
              "provider_delay_seconds": None, "live_evidence": False, "scoring_called": False,
              "files": [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)}
                        for p in sorted(directory.iterdir())]}
    _json_new(directory / "manifest.json", output)
    return directory / "manifest.json", output
