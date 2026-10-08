"""Offline gap diagnosis from pinned observations; no filling or scoring."""
import json
from pathlib import Path

import pandas as pd

from .incremental import REQUIRED_COLUMNS, merge_incremental_minutes
from .io import DataContractError, sha256
from .live_trial import _trial_context
from .providers import ENDPOINT, parse_vci_minutes
from .realtime import _scheduled_bar_ends
from .trial_history import _observation
from .validation import normalize_minutes


def _stamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise DataContractError("Diagnostic times must include a timezone")
    return stamp.tz_convert("UTC")


def _details(frame, profile, day, symbol, cutoff, expected):
    result = normalize_minutes(frame, profile, start=day, end=day,
                               symbols=[symbol], as_of=cutoff)
    accepted = set(result.bars.timestamp)
    shift = pd.Timedelta(minutes=1) if profile["bar_label"] == "bar_end" else pd.Timedelta(0)
    present = set(frame.timestamp)
    gaps = []
    for end in expected:
        if end in accepted:
            continue
        begin = end - pd.Timedelta(minutes=5)
        wanted = [begin + pd.Timedelta(minutes=i) + shift for i in range(5)]
        bad = result.rejected_rows.loc[result.rejected_rows.timestamp.ge(begin)
                                      & result.rejected_rows.timestamp.lt(end)]
        rejected_bar = result.rejected_bars.loc[result.rejected_bars.timestamp.eq(end)]
        reasons = sorted({reason for value in rejected_bar.reasons for reason in value.split("|") if reason})
        gaps.append({"bar_end_utc": end.isoformat(),
            "missing_source_minutes_utc": [value.isoformat() for value in wanted if value not in present],
            "bar_reasons": reasons or ["absent_bar"],
            "invalid_minutes": [{"source_timestamp_utc": (row.timestamp + shift).isoformat(),
                                  "reasons": sorted(set(filter(None, row.reasons.split("|"))))}
                                 for row in bad.itertuples()]})
    return gaps


def diagnose_frames(observations, profile, *, symbol, day, cutoff_utc):
    """Compare a fixed closed prefix and a revision-aware union of actual rows.

    Recovery here is retrospective data coverage only. A late row never becomes
    an on-time observation or a live delay/scoring certificate.
    """
    if not 1 <= len(observations) <= 24:
        raise DataContractError("Use between one and24 observations")
    cutoff = _stamp(cutoff_utc)
    if (day not in profile["calendar"]["dates"]
            or cutoff.tz_convert(profile["calendar"]["timezone"]).date().isoformat() != day):
        raise DataContractError("Diagnostic cutoff requires the observed calendar date")
    expected, _ = _scheduled_bar_ends(profile["calendar"], day, cutoff)
    if not expected:
        raise DataContractError("Diagnostic prefix has no closed scheduled bars")
    ordered = sorted(observations, key=lambda item: _stamp(item["received_at_utc"]))
    if cutoff > _stamp(ordered[0]["received_at_utc"]):
        raise DataContractError("Cutoff must not exceed the earliest observation receipt")
    snapshots, merged, revisions = [], None, 0
    seen = set()
    for item in ordered:
        if item["id"] in seen:
            raise DataContractError("Duplicate diagnostic observation")
        seen.add(item["id"])
        receipt = _stamp(item["received_at_utc"])
        frame = item["frame"].copy()
        if (frame.empty or set(frame.symbol) != {symbol} or not frame.source.eq("VCI").all()
                or frame.timestamp.isna().any() or frame.timestamp.gt(receipt).any()):
            raise DataContractError("Observation symbol/source/timestamps are invalid")
        gaps = _details(frame, profile, day, symbol, cutoff, expected)
        snapshots.append({"id": item["id"], "received_at_utc": receipt.isoformat(),
                          "gap_count": len(gaps), "gaps": gaps})
        frame["retrieved_at_utc"] = receipt
        incoming = frame[REQUIRED_COLUMNS].copy()
        update = merge_incremental_minutes(incoming.iloc[:0] if merged is None else merged, incoming)
        merged = update.minutes
        revisions += update.report["revisions_applied"]
    merged_gaps = _details(merged, profile, day, symbol, cutoff, expected)
    first = {row["bar_end_utc"] for row in snapshots[0]["gaps"]}
    last = {row["bar_end_utc"] for row in merged_gaps}
    return {"kind": "offline_gap_diagnostic", "criteria_met": False,
        "live_evidence": False, "scoring_called": False, "source_rights_status": "pending",
        "symbol": symbol, "calendar_date": day, "cutoff_utc": cutoff.isoformat(),
        "cutoff_basis": "explicit common closed prefix; no measured provider delay implied",
        "source_bar_label_assumption": profile["bar_label"],
        "source_confirmations_pending": profile["confirmations_pending"],
        "expected_closed_bars": len(expected), "observations": snapshots,
        "merged_gap_count": len(merged_gaps), "merged_gaps": merged_gaps,
        "recovered_bar_ends_from_raw": sorted(first - last), "revisions_applied": revisions,
        "filled_or_interpolated_rows": 0,
        "operator_action": ("Source evidence is still missing/invalid; keep the live gap guard."
                            if merged_gaps else "Coverage recovered from real rows; fresh delay/session guards are still required.")}


def diagnose_capture(project_root, capture_path, capture_sha256, *, cutoff_utc,
                     additional_reports=()):
    root = Path(project_root).resolve()
    def pinned(relative, digest, folder):
        path = (root / relative).resolve()
        if not path.is_relative_to(root / folder) or sha256(path) != digest:
            raise DataContractError("Diagnostic input path/checksum mismatch")
        return path
    path = pinned(capture_path, capture_sha256, "reports/session_observations")
    capture = json.loads(path.read_text(encoding="utf-8"))
    if (capture.get("kind") != "bounded_session_observations" or capture.get("status") == "RUNNING"
            or capture.get("source_rights_status") != "pending"):
        raise DataContractError("Use a stopped capture with pending source rights")
    settings, profile, context = _trial_context(root, capture["context"]["config_path"])
    symbol, day = capture["symbol"], capture["calendar_date"]
    if context != capture["context"] or symbol not in settings["data"]["symbols"]:
        raise DataContractError("Capture context/symbol mismatch")
    frames, failures = [], []
    paths = [path, root / context["config_path"], root / context["calendar_path"]]
    seen = set()
    for item in capture["observations"]:
        metadata = pinned(item["metadata_path"], item["metadata_sha256"], "data/raw")
        if metadata in seen:
            raise DataContractError("Duplicate diagnostic metadata")
        seen.add(metadata)
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if (meta.get("endpoint") != ENDPOINT or meta.get("symbol") != symbol
                or meta.get("expected_local_date") != day or meta.get("source_rights_status") != "pending"
                or meta.get("query", {}).get("symbols") != [symbol]
                or meta.get("query", {}).get("timeFrame") != "ONE_MINUTE"):
            raise DataContractError("Observation scope/source mismatch")
        paths.append(metadata)
        if meta.get("raw_path") is not None:
            raw = pinned(meta["raw_path"], meta["raw_sha256"], "data/raw")
            if raw.parent != metadata.parent:
                raise DataContractError("Observation raw must accompany its metadata")
            paths.append(raw)
        if meta.get("state") != "CURRENT_DATE_OBSERVED_CANDIDATE" or meta.get("http_status") != 200:
            failures.append({"metadata_path": item["metadata_path"], "state": meta.get("state"),
                             "error_type": meta.get("error_type"), "http_status": meta.get("http_status")})
            continue
        if meta.get("raw_path") is None or _stamp(meta["received_at_utc"]) < _stamp(meta["request_started_at_utc"]):
            raise DataContractError("Observation raw/receipt is invalid")
        frame = parse_vci_minutes(json.loads(raw.read_bytes()), symbol)
        frames.append({"id": item["metadata_path"], "frame": frame, "received_at_utc": meta["received_at_utc"]})
    for report_path, digest in additional_reports:
        frame, meta, report = _observation(root, report_path, digest, symbol)
        if meta["expected_local_date"] != day:
            raise DataContractError("Additional observation must use the same calendar date")
        raw = root / meta["raw_path"]
        if raw.with_name("metadata.json") in seen:
            raise DataContractError("Additional observation duplicates capture metadata")
        seen.add(raw.with_name("metadata.json"))
        paths.extend([report, raw, raw.with_name("metadata.json")])
        frames.append({"id": str(report.relative_to(root)), "frame": frame,
                       "received_at_utc": meta["received_at_utc"]})
    result = diagnose_frames(frames, profile, symbol=symbol, day=day, cutoff_utc=cutoff_utc)
    result.update(capture_status=capture["status"], failed_observations=failures,
                  artifacts=[{"path": str(p.relative_to(root)).replace("\\", "/"), "sha256": sha256(p)}
                             for p in dict.fromkeys(paths)])
    return result
