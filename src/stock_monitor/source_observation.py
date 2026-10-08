"""One bounded source observation to gather evidence before live-trial eligibility.

This does not approve a calendar date, infer provider delay, or call a scorer.
The transport is explicitly injected; the CLI must opt in to a real request.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd

from .config import load_config, pending_confirmations
from .io import DataContractError
from .providers import ENDPOINT, ProviderError, parse_vci_minutes
from .validation import load_profile


def observation_plan(project_root, expected_local_date, symbol, *, now):
    root = Path(project_root).resolve()
    settings = load_config(root / "configs/research.yaml")
    profile = load_profile(settings)
    if settings["data"]["source"] != "VCI" or not profile["research"]:
        raise DataContractError("Source observation requires the approved VCI research profile")
    if symbol not in settings["data"]["symbols"]:
        raise DataContractError("Observation symbol is outside the approved scope")
    stamp = pd.Timestamp(now)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise DataContractError("Observation clock must be timezone-aware")
    stamp = stamp.tz_convert("UTC")
    local_date = stamp.tz_convert("Asia/Ho_Chi_Minh").date().isoformat()
    if expected_local_date != local_date:
        raise DataContractError("Expected local date differs from the observation clock")
    return {
        "schema_version": 1, "kind": "source_observation_not_live_trial",
        "expected_local_date": local_date, "symbol": symbol,
        "request_started_at_utc": stamp.isoformat(), "endpoint": ENDPOINT,
        "query": {"symbols": [symbol], "timeFrame": "ONE_MINUTE",
                  "to": int(stamp.timestamp()), "countBack": 500},
        "timeout_seconds": 10, "max_requests": 1, "max_retries": 0,
        "calendar_date_previously_observed": local_date in profile["calendar"]["dates"],
        "source_rights_status": "pending", "confirmations_pending": pending_confirmations(settings),
        "calendar_changed": False, "scoring_called": False, "criteria_met": False,
        "provider_delay_seconds": None, "receive_to_score_latency_ms": None,
        "live_completion_evidence": False,
    }


def observe_source_once(project_root, expected_local_date, symbol, *, transport,
                        clock=lambda: datetime.now(timezone.utc)):
    if not callable(transport) or not callable(clock):
        raise DataContractError("Observation requires an explicit transport and clock")
    root = Path(project_root).resolve()
    report = observation_plan(root, expected_local_date, symbol, now=clock())
    name = f"source_observation_{symbol}_{uuid.uuid4().hex}"
    directory = root / "data/raw/vci" / name
    directory.mkdir(parents=True, exist_ok=False)
    report.update(network_attempted=True, state="BLOCKED_TRANSPORT", http_status=None,
                  received_at_utc=None, raw_path=None, raw_sha256=None)
    try:
        status, headers, body = transport(report["query"], report["timeout_seconds"])
        received = pd.Timestamp(clock())
        if pd.isna(received) or received.tzinfo is None:
            raise DataContractError("Receipt clock must be timezone-aware")
        received = received.tz_convert("UTC")
        report.update(http_status=status, received_at_utc=received.isoformat())
        if not isinstance(body, bytes):
            raise DataContractError("Source body must be exact bytes")
        raw_path = directory / "response.json"
        with raw_path.open("xb") as handle:
            handle.write(body)
        report.update(raw_path=raw_path.relative_to(root).as_posix(),
                      raw_sha256=hashlib.sha256(body).hexdigest(), raw_bytes=len(body),
                      response_headers={str(key).lower(): str(value) for key, value in headers.items()
                                        if str(key).lower() in {"date", "content-type", "retry-after"}})
        if status != 200:
            report["state"] = ("BLOCKED_PROVIDER_ACCESS" if status in {401, 403} else
                               "BLOCKED_PROVIDER_QUOTA" if status == 429 else "BLOCKED_PROVIDER_HTTP")
        else:
            frame = parse_vci_minutes(json.loads(body), symbol)
            invalid = int(frame.timestamp.isna().sum())
            future = int(frame.timestamp.gt(received).sum())
            today = frame.timestamp.dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.astype(str).eq(expected_local_date)
            latest = frame.timestamp.max()
            report.update(rows=len(frame), invalid_timestamps=invalid, future_timestamp_rows=future,
                          current_date_rows=int(today.sum()),
                          latest_source_timestamp_utc=None if pd.isna(latest) else latest.isoformat(),
                          latest_minute_age_seconds=None if pd.isna(latest) else (received - latest).total_seconds())
            candidate = bool(today.any() and not invalid and not future)
            report["state"] = "CURRENT_DATE_OBSERVED_CANDIDATE" if candidate else "BLOCKED_NO_VALID_CURRENT_DATE"
            report["current_date_candidate"] = candidate
    except (OSError, TimeoutError) as exc:
        report.update(state="BLOCKED_TRANSPORT", error_type=type(exc).__name__)
    except (ValueError, TypeError, ProviderError, DataContractError) as exc:
        report.update(state="BLOCKED_SOURCE_CONTRACT", error_type=type(exc).__name__)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["limitations"] = [
        "A single source observation is not an official session calendar or measured provider delay.",
        "Latest-minute age is source timestamp age, not measured provider release latency.",
        "No scoring, calendar edit, automatic approval of source rights or live completion claim."]
    metadata_path = directory / "metadata.json"
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report, metadata_path
