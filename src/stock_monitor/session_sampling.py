"""Bounded source observations and an auditable empirical closed-bar delay basis."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import pandas as pd

from .incremental import _atomic_json
from .io import DataContractError, sha256
from .live_trial import _trial_context
from .providers import ENDPOINT, parse_vci_minutes, public_transport
from .source_observation import observe_source_once
from .validation import clock_minutes, normalize_minutes


def utc_now():
    return datetime.now(timezone.utc)


def session_gate(profile, observed_at):
    stamp = pd.Timestamp(observed_at)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise DataContractError("Session clock must include a timezone")
    local = stamp.tz_convert("Asia/Ho_Chi_Minh")
    day = local.date().isoformat()
    calendar = profile["calendar"]
    if day not in calendar["dates"]:
        return "BLOCKED_UNKNOWN_CALENDAR", day
    minute = local.hour * 60 + local.minute + local.second / 60
    active = any(clock_minutes(s["start"]) <= minute < clock_minutes(s["end"])
                 for s in calendar.get("overrides", {}).get(day, calendar["sessions"]))
    return ("SESSION_OPEN" if active else "OUTSIDE_SESSION"), day


def _bounds(max_polls, interval_seconds):
    if type(max_polls) is not int or not 2 <= max_polls <= 12:
        raise DataContractError("Sampler max_polls must be 2..12")
    if type(interval_seconds) not in (int, float) or not 15 <= interval_seconds <= 60:
        raise DataContractError("Sampler interval_seconds must be 15..60")


def sample_session(project_root, config_path, symbol, *, transport,
                   clock=utc_now, sleep=time.sleep, max_polls=12, interval_seconds=60):
    """At most 12 single-attempt requests; persist each receipt before waiting.

    Injected clocks/transports are always labeled offline, even if their payloads
    were originally taken from a real source.
    """
    _bounds(max_polls, interval_seconds)
    root = Path(project_root).resolve()
    settings, profile, context = _trial_context(root, config_path)
    if symbol not in settings["data"]["symbols"]:
        raise DataContractError("Sampler symbol is outside the approved scope")
    directory = root / "reports/session_observations" / uuid.uuid4().hex
    path = directory / "session.json"
    real = transport is public_transport and clock is utc_now and sleep is time.sleep
    report = {"schema_version": 1, "kind": "bounded_session_observations", "context": context,
              "symbol": symbol, "capture_kind": "real_public_transport" if real else "offline_injected",
              "max_polls": max_polls, "interval_seconds": interval_seconds,
              "maximum_requested_wait_seconds": (max_polls - 1) * interval_seconds,
              "requested_wait_seconds": 0, "observations": [], "status": "RUNNING",
              "source_rights_status": "pending", "criteria_met": False}
    _atomic_json(path, report)
    last_receipt = None
    for index in range(max_polls):
        _, current_profile, current_context = _trial_context(root, config_path)
        if current_context != context or current_profile != profile:
            report["status"] = "BLOCKED_CONTEXT_CHANGED"
            break
        started = pd.Timestamp(clock())
        gate, day = session_gate(profile, started)
        if gate != "SESSION_OPEN":
            report["status"] = gate
            break
        if report.get("calendar_date", day) != day or (last_receipt is not None and started <= last_receipt):
            report["status"] = "BLOCKED_CLOCK"
            break
        report["calendar_date"] = day
        report.setdefault("capture_started_at_utc", started.isoformat())
        if (started - pd.Timestamp(report["capture_started_at_utc"])).total_seconds() > 900:
            report["status"] = "BLOCKED_DURATION_BUDGET"
            break
        # Recheck the actual request time inside observe_source_once as well.
        def guarded_clock():
            now = clock()
            if session_gate(profile, now) != ("SESSION_OPEN", day):
                raise DataContractError("Observation crossed its market session/date")
            return now
        try:
            observed, meta_path = observe_source_once(root, day, symbol, transport=transport, clock=guarded_clock)
        except DataContractError:
            report["status"] = "BLOCKED_SESSION_BOUNDARY"
            break
        item = {"metadata_path": meta_path.relative_to(root).as_posix(), "metadata_sha256": sha256(meta_path)}
        report["observations"].append(item)
        _atomic_json(path, report)
        if observed.get("state") != "CURRENT_DATE_OBSERVED_CANDIDATE":
            report["status"] = observed.get("state", "BLOCKED_OBSERVATION")
            break
        last_receipt = pd.Timestamp(observed["received_at_utc"])
        if last_receipt < pd.Timestamp(observed["request_started_at_utc"]):
            report["status"] = "BLOCKED_CLOCK"
            break
        if index + 1 < max_polls:
            sleep(interval_seconds)
            report["requested_wait_seconds"] += interval_seconds
    else:
        report["status"] = "CAPTURE_COMPLETE"
    _atomic_json(path, report)
    if report["status"] == "CAPTURE_COMPLETE":
        try:
            report["analysis"] = analyze_observations(root, path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            report.update(status="BLOCKED_ANALYSIS", error_type=type(exc).__name__)
        _atomic_json(path, report)
    return path, report


def analyze_observations(project_root, session_path):
    """Recompute evidence from pinned bytes; never use latest timestamp age as delay.

    Only bars closing after capture began qualify. A candidate needs the same
    five-minute payload on at least two later polls. Revisions reset that count.
    The empirical delay is the latest observed change age plus one poll interval;
    it is a finite-sample basis, not a guarantee against all future revisions.
    """
    root = Path(project_root).resolve()
    path = (root / session_path).resolve()
    if not path.is_relative_to(root / "reports/session_observations"):
        raise DataContractError("Session evidence must stay inside session observation reports")
    capture = json.loads(path.read_text(encoding="utf-8"))
    if capture.get("kind") != "bounded_session_observations" or capture.get("status") != "CAPTURE_COMPLETE":
        raise DataContractError("Incomplete observation capture cannot certify delay")
    _bounds(capture["max_polls"], capture["interval_seconds"])
    if len(capture["observations"]) != capture["max_polls"]:
        raise DataContractError("Observation count differs from the bounded capture plan")
    _, profile, context = _trial_context(root, capture["context"]["config_path"])
    if context != capture["context"] or capture.get("source_rights_status") != "pending":
        raise DataContractError("Observation context or source status changed")
    bars, prior_receipt, capture_started, revisions = {}, None, None, []
    seen_paths = set()
    for item in capture["observations"]:
        metadata_path = (root / item["metadata_path"]).resolve()
        if (not metadata_path.is_relative_to(root / "data/raw") or metadata_path in seen_paths
                or sha256(metadata_path) != item["metadata_sha256"]):
            raise DataContractError("Duplicate or changed observation metadata")
        seen_paths.add(metadata_path)
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = (root / meta["raw_path"]).resolve()
        if (not raw.is_relative_to(metadata_path.parent) or sha256(raw) != meta["raw_sha256"]
                or meta.get("endpoint") != ENDPOINT or meta.get("symbol") != capture["symbol"]
                or meta.get("http_status") != 200 or meta.get("state") != "CURRENT_DATE_OBSERVED_CANDIDATE"
                or meta.get("query", {}).get("symbols") != [capture["symbol"]]
                or meta.get("query", {}).get("timeFrame") != "ONE_MINUTE"):
            raise DataContractError("Observation raw/source/scope mismatch")
        request, receipt = pd.Timestamp(meta["request_started_at_utc"]), pd.Timestamp(meta["received_at_utc"])
        day = capture["calendar_date"]
        if (session_gate(profile, request) != ("SESSION_OPEN", day)
                or session_gate(profile, receipt) != ("SESSION_OPEN", day) or receipt < request
                or (prior_receipt is not None and (request - prior_receipt).total_seconds() < capture["interval_seconds"])):
            raise DataContractError("Observation timestamps/cadence are invalid")
        if capture_started is None:
            capture_started = request
        prior_receipt = receipt
        frame = parse_vci_minutes(json.loads(raw.read_bytes()), capture["symbol"])
        if frame.timestamp.isna().any() or frame.timestamp.gt(receipt).any():
            raise DataContractError("Invalid/future observation timestamp")
        accepted = normalize_minutes(frame, profile, start=day, end=day,
                                     symbols=[capture["symbol"]], as_of=receipt).bars
        present = set()
        for bar in accepted.itertuples():
            end = bar.timestamp
            if end <= capture_started:
                continue  # Historical/left-censored bars cannot measure release delay.
            present.add(end)
            stamps = frame.timestamp - (pd.Timedelta(minutes=1) if profile["bar_label"] == "bar_end" else pd.Timedelta(0))
            raw_bar = frame.loc[stamps.ge(end - pd.Timedelta(minutes=5)) & stamps.lt(end),
                                ["timestamp", "open", "high", "low", "close", "volume", "accumulated_volume"]]
            digest = hashlib.sha256(raw_bar.sort_values("timestamp").to_json(orient="records", date_format="iso").encode()).hexdigest()
            previous = bars.get(end)
            if previous is None:
                bars[end] = {"bar_end_utc": end.isoformat(), "first_seen_at_utc": receipt.isoformat(),
                             "last_change_at_utc": receipt.isoformat(), "fingerprint": digest,
                             "stable_followups": 0, "missing_after_seen": False}
            elif previous["fingerprint"] != digest:
                revisions.append({"bar_end_utc": end.isoformat(), "received_at_utc": receipt.isoformat(),
                                  "previous_fingerprint": previous["fingerprint"], "fingerprint": digest})
                previous.update(fingerprint=digest, last_change_at_utc=receipt.isoformat(), stable_followups=0)
            else:
                previous["stable_followups"] += 1
        for end in set(bars) - present:
            bars[end]["missing_after_seen"] = True
    eligible = [b for b in bars.values() if b["stable_followups"] >= 2 and not b["missing_after_seen"]]
    if (prior_receipt - capture_started).total_seconds() > 900:
        raise DataContractError("Observation duration exceeds the finite sampling budget")
    delay = (math.ceil(max((pd.Timestamp(b["last_change_at_utc"]) - pd.Timestamp(b["bar_end_utc"])).total_seconds()
                          for b in eligible) + capture["interval_seconds"]) if len(eligible) >= 2 else None)
    enough = delay is not None and delay <= 3600
    return {"eligible_new_closed_bars": len(eligible), "bars": list(bars.values()), "revisions": revisions,
            "empirical_delay_seconds": delay if enough else None, "basis_sufficient": enough,
            "capture_started_at_utc": capture_started.isoformat(), "last_received_at_utc": prior_receipt.isoformat(),
            "measured_live_basis": bool(enough and capture.get("capture_kind") == "real_public_transport"),
            "limitations": "At least two newly closed bars and two stable followups each; finite sample only. "
                           "Source bar-label assumptions remain pending; later revisions remain possible."}


def verified_delay_basis(project_root, session_path, expected_sha256, *, config_path, symbol, now):
    root = Path(project_root).resolve()
    path = (root / session_path).resolve()
    if not path.is_relative_to(root / "reports/session_observations") or sha256(path) != expected_sha256:
        raise DataContractError("Delay evidence path/checksum mismatch")
    capture = json.loads(path.read_text(encoding="utf-8"))
    _, profile, context = _trial_context(root, config_path)
    gate, day = session_gate(profile, now)
    if (gate != "SESSION_OPEN" or capture["context"] != context or capture["symbol"] != symbol
            or capture.get("calendar_date") != day):
        raise DataContractError("Delay evidence must match the current session, context and symbol")
    analysis = analyze_observations(root, path)
    age = (pd.Timestamp(now) - pd.Timestamp(analysis["last_received_at_utc"])).total_seconds()
    if not analysis["measured_live_basis"] or not 0 <= age <= 1800:
        raise DataContractError("Fresh real in-session measured delay evidence is required")
    return analysis
