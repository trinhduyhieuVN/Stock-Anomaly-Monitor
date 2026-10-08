"""Market-data checks, evidence-driven conversion, sessions and gap boundaries."""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import pending_confirmations
from .io import BAR_COLUMNS, DataContractError, sha256

NUMERIC = ["open", "high", "low", "close", "volume"]


class SourceNotConfirmed(DataContractError):
    """Source evidence is incomplete; no approved normalization is possible."""


@dataclass
class ValidationResult:
    bars: pd.DataFrame
    rejected_rows: pd.DataFrame
    rejected_bars: pd.DataFrame
    report: dict


def load_profile(settings, diagnostic=False):
    pending = pending_confirmations(settings)
    research = not diagnostic and settings["data"].get("normalization_mode", "strict") == "research"
    if not diagnostic and pending and not research:
        raise SourceNotConfirmed("Pending source confirmations: " + ", ".join(pending))
    if diagnostic:
        profile = dict(settings["data"]["diagnostic_profile"])
        if not profile.get("evidence"):
            raise DataContractError("Diagnostic profile requires explicit evidence and limitations")
    elif research:
        profile = dict(settings["data"]["research_profile"])
    else:
        entries = settings["data"]["confirmations"]
        profile = {"bar_label": entries["source_bar_label"]["value"],
            "price": entries["source_price_unit"]["value"],
            "volume": entries["source_volume_unit"]["value"],
            "calendar_file": entries["session_calendar"]["value"]["calendar_file"],
            "evidence": {k: v["evidence"] for k, v in entries.items()}}
    if not diagnostic and profile["bar_label"] not in {"bar_start", "bar_end"}:
        raise DataContractError("Unknown source bar label")
    expected = {"price": {"VND": 1, "thousand_VND": 1000}, "volume": {"shares": 1, "lot_100_shares": 100}}
    for key in ("price", "volume"):
        if diagnostic:
            continue  # Diagnostic checks never apply guessed units.
        conversion = profile[key]
        unit, factor = conversion.get("unit"), conversion.get("multiplier")
        if unit not in expected[key] or isinstance(factor, bool) or factor != expected[key].get(unit):
            raise DataContractError(f"Invalid evidenced {key} conversion")
    calendar_path = (settings["_root"] / profile["calendar_file"]).resolve()
    if not calendar_path.is_relative_to(settings["_root"]):
        raise DataContractError("Calendar must stay in the project")
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    if calendar.get("timezone") != "Asia/Ho_Chi_Minh" or not calendar.get("dates"):
        raise DataContractError("Calendar requires explicit local dates and timezone")
    dates = calendar["dates"]
    if not isinstance(dates, list) or dates != sorted(set(dates)) or any(date.fromisoformat(d).isoformat() != d for d in dates):
        raise DataContractError("Invalid/duplicate calendar dates")
    if not diagnostic and not research and (calendar.get("status") != "confirmed" or calendar.get("complete") is not True):
        raise SourceNotConfirmed("An observed-only calendar cannot approve missing-day coverage")
    for sessions in [calendar["sessions"], *calendar.get("overrides", {}).values()]:
        last_end = -1
        ids = set()
        for session in sessions:
            start, end = clock_minutes(session["start"]), clock_minutes(session["end"])
            if start < last_end or end <= start or (end - start) % 5 or session["id"] in ids:
                raise DataContractError("Calendar sessions overlap or do not contain whole 5m bins")
            last_end = end
            ids.add(session["id"])
    profile.update(calendar=calendar, calendar_sha256=sha256(calendar_path), diagnostic=diagnostic,
                   confirmations_pending=pending, research=research)
    return profile


def clock_minutes(value):
    pieces = value.split(":")
    if len(pieces) != 2:
        raise DataContractError("Use HH:MM calendar times")
    hour, minute = map(int, pieces)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise DataContractError("Invalid calendar time")
    return hour * 60 + minute


def _flag(frame, mask, reason):
    frame.loc[mask.fillna(True) if hasattr(mask, "fillna") else mask, "reasons"] += reason + "|"


def _base_checks(frame):
    frame = frame.copy().reset_index(drop=True)
    frame["reasons"] = ""
    _flag(frame, frame.timestamp.isna(), "invalid_timestamp")
    for col in NUMERIC:
        boolean = frame[col].map(lambda v: isinstance(v, bool))
        frame[col] = pd.to_numeric(frame[col], errors="coerce").mask(boolean)
    finite = np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all(axis=1)
    _flag(frame, pd.Series(~finite, index=frame.index), "nonfinite_ohlcv")
    _flag(frame, frame[NUMERIC[:-1]].le(0).any(axis=1), "nonpositive_price")
    _flag(frame, frame.volume.lt(0), "negative_volume")
    _flag(frame, frame.volume.notna() & frame.volume.mod(1).ne(0), "fractional_volume")
    inconsistent = (frame.low.gt(frame.high) | frame.open.lt(frame.low) | frame.open.gt(frame.high)
                    | frame.close.lt(frame.low) | frame.close.gt(frame.high))
    _flag(frame, inconsistent, "invalid_ohlc")
    return frame


def _duplicate_checks(frame, extra_values=()):
    valid = frame.loc[frame.timestamp.notna()]
    keys = ["source", "symbol", "timestamp"]
    values = NUMERIC + list(extra_values)
    groups = valid.groupby(keys, dropna=False)[values].nunique(dropna=False)
    conflicts = groups.gt(1).any(axis=1)
    bad_keys = set(conflicts.index[conflicts].tolist())
    conflict = pd.Series([tuple(row) in bad_keys for row in frame[keys].itertuples(index=False, name=None)], index=frame.index)
    _flag(frame, conflict, "conflicting_duplicate")
    exact = frame.duplicated(keys, keep="first") & ~conflict & frame.timestamp.notna()
    _flag(frame, exact, "exact_duplicate")
    return exact


def _calendar_columns(frame, calendar, start, end, *, is_end=False):
    local = frame.timestamp.dt.tz_convert(calendar["timezone"])
    left = local - pd.Timedelta(minutes=5) if is_end else local
    frame["session_id"] = left.dt.strftime("%Y-%m-%d")
    minute = left.dt.hour * 60 + left.dt.minute
    frame["period"] = ""
    frame["bar_start"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    allowed = set(calendar["dates"])
    in_range = frame.session_id.between(start, end)
    _flag(frame, ~in_range & frame.timestamp.notna(), "outside_requested_range")
    _flag(frame, ~frame.session_id.isin(allowed) & in_range, "nontrading_date")
    dates = frame.session_id.dropna().unique()
    for day in dates:
        if day not in allowed:
            continue
        midnight = pd.Timestamp(day, tz=calendar["timezone"])
        for session in calendar.get("overrides", {}).get(day, calendar["sessions"]):
            begin, stop = clock_minutes(session["start"]), clock_minutes(session["end"])
            mask = frame.session_id.eq(day) & minute.ge(begin) & minute.lt(stop)
            # Floor relative to session start, never across a break/auction.
            offsets = ((minute.loc[mask] - begin) // 5) * 5 + begin
            frame.loc[mask, "bar_start"] = (midnight + pd.to_timedelta(offsets.to_numpy(dtype="int64"), unit="min")).tz_convert("UTC")
            frame.loc[mask, "period"] = session["id"]
    _flag(frame, frame.period.eq("") & in_range & frame.session_id.isin(allowed), "outside_continuous_session")
    return frame


def assign_segments(frame):
    frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if frame.empty:
        frame["segment_id"] = pd.Series(dtype="string")
        return frame
    previous = frame.shift()
    new = (frame.symbol.ne(previous.symbol) | frame.session_id.ne(previous.session_id)
           | frame.period.ne(previous.period) | frame.timestamp.sub(previous.timestamp).ne(pd.Timedelta(minutes=5)))
    number = new.groupby([frame.symbol, frame.session_id]).cumsum().astype(int)
    frame["segment_id"] = frame.session_id + ":" + frame.symbol + ":" + number.astype(str)
    return frame


def reason_counts(frame):
    return dict(sorted(Counter(reason for text in frame.reasons for reason in text.split("|") if reason).items()))


def _summary(bars, rejected_rows, rejected_bars, profile, start, end):
    by_symbol = {}
    date_sets = []
    for symbol, group in bars.groupby("symbol"):
        dates = set(group.session_id)
        date_sets.append(dates)
        by_symbol[symbol] = {"bars": len(group), "session_dates": len(dates), "segments": group.segment_id.nunique(),
                             "first_bar_end_utc": str(group.timestamp.min()), "last_bar_end_utc": str(group.timestamp.max())}
    common = sorted(set.intersection(*date_sets)) if date_sets else []
    return {"diagnostic": profile["diagnostic"], "source_confirmations_pending": profile["confirmations_pending"],
        "requested_start": start, "requested_end": end, "market_checks_passed_bars": len(bars),
        "accepted_bars": 0 if profile["diagnostic"] else len(bars),
        "unverified_bars": len(bars) if profile["diagnostic"] else 0,
        "rejected_source_rows": len(rejected_rows), "rejected_bar_groups": len(rejected_bars),
        "source_row_reasons": reason_counts(rejected_rows), "bar_reasons": reason_counts(rejected_bars),
        "by_symbol": by_symbol, "common_session_dates": common, "common_session_date_count": len(common),
        "calendar_complete": profile["calendar"].get("complete") is True,
        "source_units_verified": not profile["diagnostic"] and not profile.get("research", False),
        "source_profile_mode": "research" if profile.get("research", False) else "strict",
        "research_ready": False, "training_ready": False}


def normalize_minutes(minutes, profile, *, start, end, symbols, as_of=None):
    if profile["confirmations_pending"] and not profile.get("research", False):
        raise SourceNotConfirmed("Cannot shift source labels or convert units before evidence is confirmed")
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise DataContractError("as_of must have a timezone")
    frame = minutes.copy()
    if set(frame.symbol) != set(symbols) or not frame.source.eq("VCI").all():
        raise DataContractError("Snapshot symbols/source differ from configured scope")
    if profile["bar_label"] == "bar_end":
        frame["timestamp"] -= pd.Timedelta(minutes=1)
    for col in NUMERIC[:-1]:
        frame[col] *= profile["price"]["multiplier"]
    frame["volume"] *= profile["volume"]["multiplier"]
    frame["accumulated_volume"] *= profile["volume"]["multiplier"]
    frame = _base_checks(frame)
    _flag(frame, frame.timestamp.dt.second.ne(0) | frame.timestamp.dt.microsecond.ne(0) | frame.timestamp.dt.nanosecond.ne(0), "minute_timestamp_not_aligned")
    _flag(frame, frame.timestamp.add(pd.Timedelta(minutes=1)).gt(as_of), "unclosed_minute")
    _flag(frame, ~np.isfinite(frame.accumulated_volume.to_numpy(dtype=float)) | frame.accumulated_volume.lt(0) | frame.accumulated_volume.mod(1).ne(0), "invalid_accumulated_volume")
    exact = _duplicate_checks(frame, ["accumulated_volume"])
    working = frame.loc[~exact].sort_values(["symbol", "timestamp"]).copy()
    local_date = working.timestamp.dt.tz_convert(profile["calendar"]["timezone"]).dt.date
    delta = working.groupby([working.symbol, local_date]).accumulated_volume.diff()
    _flag(working, delta.notna() & delta.ne(working.volume), "volume_delta_mismatch")
    working = _calendar_columns(working, profile["calendar"], start, end)
    eligible = working.loc[working.bar_start.notna() & working.session_id.between(start, end)].copy()
    eligible["bad_minute"] = eligible.reasons.ne("")
    grouped = eligible.groupby(["symbol", "bar_start"], sort=True)
    bins = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), source_minutes=("timestamp", "size"),
        unique_minutes=("timestamp", "nunique"), bad_minute=("bad_minute", "any"),
        session_id=("session_id", "first"), period=("period", "first")).reset_index()
    bins["timestamp"] = bins.bar_start + pd.Timedelta(minutes=5)
    bins["source"] = "VCI"
    bins["interval"] = "5m"
    bins["reasons"] = ""
    _flag(bins, bins.unique_minutes.ne(5), "incomplete_minutes")
    _flag(bins, bins.bad_minute, "invalid_source_minute")
    _flag(bins, bins.timestamp.gt(as_of), "unclosed_bar")
    # Every source minute must be an exact minute in this particular bin.
    _flag(bins, bins.source_minutes.ne(bins.unique_minutes), "conflicting_duplicate")
    bars = assign_segments(bins.loc[bins.reasons.eq("")].copy())
    bars["quality_status"] = "unverified" if profile["diagnostic"] else "ok"
    bars["volume"] = bars.volume.astype("int64")
    rejected_rows = pd.concat([working.loc[working.reasons.ne("")], frame.loc[exact]], ignore_index=True)
    rejected_bars = bins.loc[bins.reasons.ne("")].copy()
    report = _summary(bars, rejected_rows, rejected_bars, profile, start, end)
    observed = set(zip(bins.symbol, bins.timestamp))
    missing = []
    for day in profile["calendar"]["dates"]:
        if not start <= day <= end:
            continue
        midnight = pd.Timestamp(day, tz=profile["calendar"]["timezone"])
        for session in profile["calendar"].get("overrides", {}).get(day, profile["calendar"]["sessions"]):
            for minute in range(clock_minutes(session["start"]) + 5, clock_minutes(session["end"]) + 1, 5):
                stamp = (midnight + pd.Timedelta(minutes=minute)).tz_convert("UTC")
                for symbol in symbols:
                    if (symbol, stamp) not in observed:
                        missing.append({"symbol": symbol, "bar_end_utc": str(stamp)})
    report.update(received_source_rows=len(minutes), in_requested_range_source_rows=int(working.session_id.between(start, end).sum()),
        observed_bar_groups=len(bins), missing_bar_groups=len(missing), missing_bars=missing,
        exact_duplicates_removed=int(exact.sum()), evaluated_as_of_utc=as_of.tz_convert("UTC").isoformat())
    report["coverage_passed"] = len(report["by_symbol"]) == len(symbols) and report["common_session_date_count"] >= 80
    report["training_ready"] = not profile["diagnostic"] and not profile.get("research", False) and report["coverage_passed"]
    report["research_ready"] = profile.get("research", False) and report["coverage_passed"]
    report["status"] = ("DIAGNOSTIC_ONLY" if profile["diagnostic"] else
        "RESEARCH_VALIDATED" if report["research_ready"] else
        "VALIDATED" if report["training_ready"] else "INSUFFICIENT_COVERAGE")
    return ValidationResult(bars[list(BAR_COLUMNS)], rejected_rows, rejected_bars, report)


def audit_raw_minutes(minutes, profile, *, start, end, symbols, as_of=None):
    """Inspect raw units and unchanged source labels; never emit model bar schema."""
    if set(minutes.symbol) != set(symbols) or not minutes.source.eq("VCI").all():
        raise DataContractError("Snapshot symbols/source differ from configured scope")
    frame = _base_checks(minutes)
    exact = _duplicate_checks(frame, ["accumulated_volume"])
    working = frame.loc[~exact].sort_values(["symbol", "timestamp"]).copy()
    local = working.timestamp.dt.tz_convert(profile["calendar"]["timezone"])
    working["reference_date"] = local.dt.strftime("%Y-%m-%d")
    delta = working.groupby([working.symbol, working.reference_date]).accumulated_volume.diff()
    _flag(working, delta.notna() & delta.ne(working.volume), "volume_delta_mismatch_raw_units")
    _flag(working, ~np.isfinite(working.accumulated_volume.to_numpy(dtype=float)) | working.accumulated_volume.lt(0), "invalid_accumulated_volume")
    in_range = working.reference_date.between(start, end)
    _flag(working, ~in_range & working.timestamp.notna(), "outside_requested_range")
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise DataContractError("as_of must have a timezone")
    _flag(working, working.timestamp.gt(as_of), "source_timestamp_in_future")
    minute = local.dt.hour * 60 + local.dt.minute
    working["reference_period"] = ""
    calendar = profile["calendar"]
    for day in working.reference_date.dropna().unique():
        for session in calendar.get("overrides", {}).get(day, calendar["sessions"]):
            mask = working.reference_date.eq(day) & minute.ge(clock_minutes(session["start"])) & minute.lt(clock_minutes(session["end"]))
            working.loc[mask, "reference_period"] = session["id"]
    _flag(working, working.timestamp.dt.second.ne(0) | working.timestamp.dt.microsecond.ne(0), "source_timestamp_not_minute_aligned")
    kept = working.loc[working.reasons.eq("")].copy()
    kept["interval"] = "1m"
    kept["quality_status"] = "unverified"
    rejected = pd.concat([working.loc[working.reasons.ne("")], frame.loc[exact]], ignore_index=True)
    date_sets, counts = [], {}
    for symbol in symbols:
        group = working.loc[working.symbol.eq(symbol) & in_range]
        dates = set(group.reference_date.dropna())
        date_sets.append(dates)
        counts[symbol] = {"received_rows": len(minutes.loc[minutes.symbol.eq(symbol)]),
            "in_requested_range_minutes": len(group), "observed_session_dates": len(dates),
            "unverified_output_minutes": int(kept.symbol.eq(symbol).sum()),
            "outside_reference_continuous_hours": int(group.reference_period.eq("").sum())}
    common = sorted(set.intersection(*date_sets)) if date_sets else []
    consecutive = working.loc[in_range & working.reference_period.ne("")].copy()
    gaps = consecutive.groupby(["symbol", "reference_date", "reference_period"]).timestamp.diff().dt.total_seconds().div(60)
    rejected_in_range = working.loc[in_range & working.reasons.ne("")]
    expected_reference_minutes = sum(sum(clock_minutes(s["end"]) - clock_minutes(s["start"]) for s in calendar.get("overrides", {}).get(day, calendar["sessions"]))
        for day in calendar["dates"] if start <= day <= end) * len(symbols)
    report = {"status": "DIAGNOSTIC_RAW_ONLY", "diagnostic": True,
        "received_source_rows": len(minutes), "in_requested_range_source_rows": int(in_range.sum()),
        "accepted_bars": 0, "unverified_output_minutes": len(kept), "rejected_source_rows": len(rejected),
        "source_row_reasons": reason_counts(rejected), "exact_duplicates_removed": int(exact.sum()),
        "in_requested_range_rejected_rows": len(rejected_in_range),
        "in_requested_range_source_row_reasons": reason_counts(rejected_in_range),
        "by_symbol": counts, "common_session_date_count": len(common), "common_session_dates": common,
        "within_reference_period_gap_count": int(gaps.gt(1).sum()),
        "missing_minutes_within_reference_gaps": int((gaps.loc[gaps.gt(1)] - 1).sum()),
        "reference_expected_minutes": expected_reference_minutes,
        "reference_observed_minutes": len(consecutive),
        "reference_missing_minutes": max(0, expected_reference_minutes - len(consecutive)),
        "calendar_complete": False, "timestamp_convention": "unchanged_source_label_epoch_utc",
        "source_bar_label": "unverified", "price_unit": "unverified_raw", "volume_unit": "unverified_raw",
        "unit_conversion_applied": False, "label_shift_seconds": 0, "aggregated_to_5m": False,
        "source_confirmations_pending": profile["confirmations_pending"],
        "requested_start": start, "requested_end": end,
        "evaluated_as_of_utc": as_of.tz_convert("UTC").isoformat(), "training_ready": False}
    return ValidationResult(kept, rejected, rejected.iloc[0:0], report)


def validate_standard_bars(frame, profile, *, start, end, symbols, as_of=None):
    """Reject dirty file rows individually; never trust declared quality/session IDs."""
    missing = set(BAR_COLUMNS) - set(frame.columns)
    if missing:
        raise DataContractError("Missing standard columns: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    stamps = []
    for value in frame.timestamp:
        try:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
                stamp = pd.NaT
        except (ValueError, TypeError):
            stamp = pd.NaT
        stamps.append(stamp)
    frame["timestamp"] = pd.to_datetime(stamps, utc=True)
    supplied_session = frame.session_id.copy().reset_index(drop=True)
    supplied_segment = frame.segment_id.copy().reset_index(drop=True)
    supplied_quality = frame.quality_status.copy().reset_index(drop=True)
    frame = _base_checks(frame)
    _flag(frame, ~frame.symbol.isin(symbols) | ~frame.source.eq("VCI") | ~frame.interval.eq("5m"), "wrong_scope")
    _flag(frame, ~supplied_quality.isin(["ok", "unverified", "rejected"]), "invalid_quality_status")
    _flag(frame, supplied_quality.eq("rejected"), "declared_rejected")
    if not profile["diagnostic"]:
        _flag(frame, supplied_quality.eq("unverified"), "declared_unverified")
    _flag(frame, ~supplied_segment.map(lambda v: isinstance(v, str) and bool(v.strip()) and v == v.strip()), "invalid_segment_id")
    exact = _duplicate_checks(frame, ["interval", "session_id", "segment_id", "quality_status"])
    frame = _calendar_columns(frame, profile["calendar"], start, end, is_end=True)
    _flag(frame, supplied_session.ne(frame.session_id), "wrong_session_id")
    _flag(frame, frame.timestamp.ne(frame.bar_start.add(pd.Timedelta(minutes=5))) & frame.bar_start.notna(), "bar_timestamp_not_aligned")
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise DataContractError("as_of must have a timezone")
    _flag(frame, frame.timestamp.gt(as_of), "unclosed_bar")
    ordered = frame.loc[~exact & frame.reasons.eq("")].sort_values(["source", "symbol", "timestamp"])
    # A supplied segment cannot span a symbol, date, break or missing bar.
    for _, group in ordered.groupby(supplied_segment.loc[ordered.index]):
        crossing = group.symbol.nunique() != 1 or group.session_id.nunique() != 1 or group.period.nunique() != 1 or group.timestamp.diff().dropna().ne(pd.Timedelta(minutes=5)).any()
        if crossing:
            _flag(frame, frame.index.isin(group.index), "segment_crosses_boundary")
    bars = assign_segments(frame.loc[frame.reasons.eq("")].copy())
    bars["quality_status"] = "unverified" if profile["diagnostic"] else "ok"
    bars["volume"] = bars.volume.astype("int64")
    rejected = frame.loc[frame.reasons.ne("")].copy()
    report = _summary(bars, rejected, rejected.iloc[0:0], profile, start, end)
    report.update(received_bars=len(frame), exact_duplicates_removed=int(exact.sum()),
                  evaluated_as_of_utc=as_of.tz_convert("UTC").isoformat())
    dirty = rejected.reasons.str.replace("exact_duplicate|", "", regex=False).ne("").any()
    report["status"] = "INVALID_INPUT" if dirty else ("DIAGNOSTIC_ONLY" if profile["diagnostic"] else "VALIDATED_FILE")
    # Source metadata is not supplied by an arbitrary standard file. Never attest provenance from its columns.
    report["training_ready"] = False
    return ValidationResult(bars[list(BAR_COLUMNS)], rejected, rejected.iloc[0:0], report)
