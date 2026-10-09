"""Frozen zero-return forecasts and causal maturity for synthetic logical replay."""
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
import math

from stock_monitor.io import DataContractError
from .contracts import LOCAL_ZONE, _fields, _hash, _number, _session_bar, utc_time, validate_event
from .synthetic import bars_as_of, content_hash

MODEL = {"algorithm": "zero_return_v1", "target": "log_target_close_over_frozen_origin_close", "horizons": [1, 3]}
MODEL_SHA256 = content_hash(MODEL)
FORECAST_FIELDS = {"schema_version", "forecast_id", "symbol", "session_id", "segment_id", "period", "source_mode",
                   "origin_bar_end", "horizon_bars", "target_bar_end", "created_at", "available_at", "as_of",
                   "prediction", "origin_close", "origin_revision", "origin_available_at", "input_sha256",
                   "model_sha256", "calendar_sha256", "period_end", "horizon_status", "clock_basis"}


def _identity(record):
    return content_hash({key: value for key, value in record.items() if key != "forecast_id"})


def validate_forecast(record):
    _fields(record, FORECAST_FIELDS)
    if record["schema_version"] != 2 or record["source_mode"] != "SYNTHETIC" or record["clock_basis"] != "logical_replay":
        raise DataContractError("This forecast baseline is synthetic logical replay only")
    if type(record["horizon_bars"]) is not int or record["horizon_bars"] not in {1, 3}:
        raise DataContractError("Unsupported horizon")
    origin, target, as_of, created, available, received, end = (utc_time(record[name], name) for name in
        ("origin_bar_end", "target_bar_end", "as_of", "created_at", "available_at", "origin_available_at", "period_end"))
    _session_bar(origin, record["session_id"])
    if (target != origin + timedelta(minutes=5 * record["horizon_bars"]) or
            not origin <= received <= as_of <= created <= available < target):
        raise DataContractError("Forecast backfill/late origin/time ordering")
    if record["symbol"] not in {"FPT", "HPG", "VCB"} or record["period"] not in {"am", "pm"} or not record["segment_id"]:
        raise DataContractError("Invalid forecast identity")
    for name in ("prediction", "origin_close"):
        _number(record[name], name)
    if record["prediction"] != 0 or record["origin_close"] <= 0:
        raise DataContractError("Expected frozen zero-return baseline with positive close")
    if type(record["origin_revision"]) is not int or record["origin_revision"] < 0:
        raise DataContractError("Invalid origin revision")
    for name in ("input_sha256", "model_sha256", "calendar_sha256", "forecast_id"):
        _hash(record[name], name)
    if record["model_sha256"] != MODEL_SHA256 or record["forecast_id"] != _identity(record):
        raise DataContractError("Frozen forecast model/input hash mismatch")
    expected_status = "censored_boundary" if target > end else "awaiting_outcome"
    if record["horizon_status"] != expected_status or origin > end:
        raise DataContractError("Invalid calendar/horizon status")
    return deepcopy(record)


def _period_end(origin, calendar):
    if calendar.get("timezone") != "Asia/Ho_Chi_Minh" or origin["session_id"] not in calendar["dates"]:
        raise DataContractError("Explicit synthetic calendar required")
    period = next((item for item in calendar["sessions"] if item["id"] == origin["period"]), None)
    if period is None:
        raise DataContractError("Unknown forecast period")
    day = datetime.fromisoformat(origin["session_id"]).date()
    start = datetime.combine(day, time.fromisoformat(period["start"]), LOCAL_ZONE)
    end = datetime.combine(day, time.fromisoformat(period["end"]), LOCAL_ZONE)
    stamp = origin["timestamp"].to_pydatetime()
    if not start < stamp.astimezone(LOCAL_ZONE) <= end:
        raise DataContractError("Origin outside closed-bar calendar period")
    return end.astimezone(timezone.utc)


def create_zero_forecast(observations, *, symbol, origin_bar_end, horizon_bars, as_of, created_at, calendar):
    if type(horizon_bars) is not int or horizon_bars not in {1, 3}:
        raise DataContractError("Unsupported horizon")
    view = bars_as_of(observations, as_of)
    matches = view.loc[view.symbol.eq(symbol) & view.timestamp.eq(utc_time(origin_bar_end))]
    if len(matches) != 1 or matches.iloc[0].source != "SYNTHETIC":
        raise DataContractError("Origin not visible in synthetic view")
    row = matches.iloc[0].to_dict()
    end = _period_end(row, calendar)
    origin = row["timestamp"].to_pydatetime()
    target = origin + timedelta(minutes=5 * horizon_bars)
    inputs = {key: row[key] for key in ("symbol", "session_id", "period", "source", "available_at")}
    inputs.update(timestamp=origin.isoformat(), close=float(row["close"]), bar_revision=int(row["bar_revision"]))
    record = {"schema_version": 2, "symbol": symbol, "session_id": row["session_id"], "segment_id": row["segment_id"],
              "period": row["period"], "source_mode": "SYNTHETIC", "origin_bar_end": origin.isoformat(),
              "horizon_bars": horizon_bars, "target_bar_end": target.isoformat(), "created_at": created_at,
              "available_at": created_at, "as_of": as_of, "prediction": 0., "origin_close": float(row["close"]),
              "origin_revision": int(row["bar_revision"]), "origin_available_at": utc_time(row["available_at"]).isoformat(),
              "input_sha256": content_hash(inputs), "model_sha256": MODEL_SHA256,
              "calendar_sha256": content_hash(calendar), "period_end": end.isoformat(),
              "horizon_status": "censored_boundary" if target > end else "awaiting_outcome", "clock_basis": "logical_replay"}
    record["forecast_id"] = _identity(record)
    return validate_forecast(record)


def assess_forecast(record, observations, *, as_of, final_session=False):
    forecast = validate_forecast(record)
    cutoff = utc_time(as_of)
    if type(final_session) is not bool or cutoff < utc_time(forecast["available_at"]):
        raise DataContractError("Reassessment precedes forecast availability")
    status, evidence, outcomes, origin_revision_observed = "pending", None, [], None
    if forecast["horizon_status"] == "censored_boundary":
        status = "censored_boundary"
    elif cutoff >= utc_time(forecast["target_bar_end"]):
        view = bars_as_of(observations, as_of)
        start = utc_time(forecast["origin_bar_end"])
        origin_now = view.loc[view.symbol.eq(forecast["symbol"]) & view.timestamp.eq(start)]
        if len(origin_now) == 1:
            origin_revision_observed = int(origin_now.iloc[0].bar_revision) != forecast["origin_revision"]
        for offset in range(1, forecast["horizon_bars"] + 1):
            target = start + timedelta(minutes=5 * offset)
            matches = view.loc[view.symbol.eq(forecast["symbol"]) & view.session_id.eq(forecast["session_id"]) &
                               view.period.eq(forecast["period"]) & view.timestamp.eq(target)]
            if len(matches) != 1:
                break
            row = matches.iloc[0]
            if row.source != "SYNTHETIC":
                raise DataContractError("Outcome source mismatch")
            outcomes.append({"timestamp": target.isoformat(), "close": float(row.close),
                             "revision": int(row.bar_revision), "available_at": utc_time(row.available_at).isoformat()})
        if len(outcomes) < forecast["horizon_bars"]:
            status = "censored_missing_bar" if final_session and cutoff >= utc_time(forecast["period_end"]) else "pending_missing_bar"
        else:
            status = "mature"
            realized = math.log(outcomes[-1]["close"] / forecast["origin_close"])
            evidence = {"horizon_bars": forecast["horizon_bars"], "forecast_created_at": forecast["created_at"],
                        "target_bar_end": forecast["target_bar_end"],
                        "outcome_available_at": max((row["available_at"] for row in outcomes), key=utc_time),
                        "prediction": forecast["prediction"], "realized_return": realized,
                        "residual": realized - forecast["prediction"], "forecast_sha256": content_hash(forecast)}
    outcome_hash = content_hash(outcomes) if status == "mature" else None
    key = content_hash({"forecast_sha256": content_hash(forecast), "status": status, "outcome_sha256": outcome_hash})
    return {"assessment_key": key, "forecast_id": forecast["forecast_id"], "status": status, "as_of": as_of,
            "outcome_sha256": outcome_hash, "post_evidence": evidence, "source_mode": "SYNTHETIC",
            "origin_revision_observed": origin_revision_observed}


def append_post_event(initial, forecast, assessment, *, version, available_at):
    event, forecast = validate_event(initial), validate_forecast(forecast)
    if event["stage"] != "initial" or type(version) is not int or version <= event["version"]:
        raise DataContractError("Post-event must append a newer version to initial")
    if (event["symbol"], event["session_id"], utc_time(event["event_end"]), event["source_mode"]) != (
            forecast["symbol"], forecast["session_id"], utc_time(forecast["origin_bar_end"]), forecast["source_mode"]):
        raise DataContractError("Forecast does not refer to this event endpoint")
    if (assessment["status"] != "mature" or assessment["forecast_id"] != forecast["forecast_id"] or
            assessment["post_evidence"]["forecast_sha256"] != content_hash(forecast)):
        raise DataContractError("Only matching mature evidence can append post-event")
    if utc_time(available_at) < max(utc_time(event["available_at"]), utc_time(assessment["as_of"])):
        raise DataContractError("Cannot backdate post-event update")
    event.update(version=version, stage="post_event", as_of=assessment["as_of"], detected_at=available_at,
                 available_at=available_at, post_evidence=deepcopy(assessment["post_evidence"]))
    event["reason_codes"] = event["reason_codes"] + ["mature_zero_return_residual_not_a_fraud_label"]
    return validate_event(event)
