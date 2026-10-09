"""Independent synthetic UI resources and causal logical delivery replay."""
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import pandas as pd

from stock_monitor.io import DataContractError
from .context import causal_context
from .contracts import utc_time, validate_news
from .forecast import MODEL_SHA256, create_zero_forecast, assess_forecast, append_post_event
from .forecast_training import rank_zero_residual
from .journal import Journal
from .segment_replay import return_view
from .segments import score_suffixes
from .synthetic import bars_as_of, content_hash, validate_observations


def _safe(root, name):
    if not isinstance(name, str) or Path(name).is_absolute() or PureWindowsPath(name).is_absolute() or "\\" in name:
        raise DataContractError("UI resource must be a relative project path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DataContractError("UI resource escapes project")
    return path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ui_bundle(root, config_name="configs/upgrade_ui.json"):
    root = Path(root)
    config = json.loads(_safe(root, config_name).read_text(encoding="utf-8"))
    if set(config) != {"schema_version", "manifest", "sha256"} or config["schema_version"] != 2:
        raise DataContractError("Invalid UI configuration")
    path = _safe(root, config["manifest"])
    if _digest(path) != config["sha256"]:
        raise DataContractError("Pinned synthetic UI manifest mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["kind"] != "upgrade_ui_synthetic_only" or manifest["source_mode"] != "SYNTHETIC" or
            manifest["contains_real_data"] is not False or manifest["distributable"] is not True):
        raise DataContractError("Only independent synthetic UI resources allowed")
    members = {row["name"]: row["sha256"] for row in manifest["files"]}
    if len(members) != len(manifest["files"]) or set(members) != {"observations.csv", "news.json", "segment_calibration.json", "residual_calibration.json"}:
        raise DataContractError("Unexpected or duplicate UI manifest member")
    for name, expected in members.items():
        if _digest(_safe(path.parent, name)) != expected:
            raise DataContractError("Pinned UI data/model artifact mismatch")
    frame = pd.read_csv(path.parent / "observations.csv")
    validate_observations(frame)
    news = json.loads((path.parent / "news.json").read_text(encoding="utf-8"))
    news = [validate_news(row) for row in news]
    segment = json.loads((path.parent / "segment_calibration.json").read_text(encoding="utf-8"))
    residual = json.loads((path.parent / "residual_calibration.json").read_text(encoding="utf-8"))
    if residual["model_sha256"] != MODEL_SHA256 or content_hash(segment) != members["segment_calibration.json"] or content_hash(residual) != members["residual_calibration.json"]:
        raise DataContractError("Frozen UI calibration binding mismatch")
    sessions = [day for day in manifest["calendar"]["dates"] if day > utc_time(segment["validation_event_end"]).date().isoformat()]
    return {"rows": frame.to_dict("records"), "news": news, "segment": segment, "residual": residual,
            "calendar": manifest["calendar"], "sessions": sessions, "key": config["sha256"], "manifest": manifest}


def replay_session(bundle, journal, *, symbol, session_id, as_of):
    """Replay only deliveries <=T; physically commit forecasts before later bars."""
    if symbol not in {"FPT", "HPG", "VCB"} or session_id not in bundle["sessions"]:
        raise DataContractError("Unknown symbol/session or session before calibration")
    cutoff = utc_time(as_of)
    scoped = [row for row in bundle["rows"] if row["symbol"] == symbol and row["session_id"] == session_id]
    deliveries = sorted([row for row in scoped if utc_time(row["available_at"]) <= cutoff],
                        key=lambda row: (utc_time(row["available_at"]), utc_time(row["timestamp"]), int(row["bar_revision"])))
    visible, forecasts, initial_events, details, versions, assessed_keys, completed = [], {}, {}, {}, {}, set(), set()
    segment_hash = content_hash(bundle["segment"])
    for delivery in deliveries:
        visible.append(delivery)
        clock = utc_time(delivery["available_at"]).isoformat()
        end = utc_time(delivery["timestamp"])
        if int(delivery["bar_revision"]) == 0:
            view = [row for row in return_view(visible, as_of=clock, split="evaluation") if utc_time(row["timestamp"]) <= end]
            current = view[-1]
            history = [row for row in view if row["segment_id"] == current["segment_id"]][-12:]
            detail = score_suffixes(history, bundle["segment"], expected_sha256=segment_hash,
                                   model_run="synthetic_ui:" + bundle["key"][:16], emitted_at=clock, latest_only=True)[0]
            event = detail["event"]
            journal.append("events", event)
            initial_events[end] = event
            details[event["event_id"]] = detail
            versions[event["event_id"]] = 1
            for horizon in (1, 3):
                if utc_time(clock) >= end + timedelta(minutes=5 * horizon):
                    continue  # late-origin backfill is forbidden, not imputed.
                forecast = create_zero_forecast(visible, symbol=symbol, origin_bar_end=delivery["timestamp"],
                    horizon_bars=horizon, as_of=clock, created_at=clock, calendar=bundle["calendar"])
                journal.append("forecasts", forecast)  # transaction committed before later outcome input.
                forecasts[forecast["forecast_id"]] = forecast
                born = assess_forecast(forecast, visible, as_of=clock)
                assert born["post_evidence"] is None
                journal.append("assessments", born)
                assessed_keys.add(born["assessment_key"])
                if born["status"] == "censored_boundary":
                    completed.add(forecast["forecast_id"])
        for forecast_id, forecast in forecasts.items():
            revised_path = int(delivery["bar_revision"]) > 0 and utc_time(forecast["origin_bar_end"]) <= end <= utc_time(forecast["target_bar_end"])
            if forecast_id in completed and not revised_path or utc_time(clock) < utc_time(forecast["target_bar_end"]):
                continue
            outcome = assess_forecast(forecast, visible, as_of=clock,
                                      final_session=utc_time(clock) >= utc_time(forecast["period_end"]))
            if outcome["assessment_key"] in assessed_keys:
                continue
            journal.append("assessments", outcome)
            assessed_keys.add(outcome["assessment_key"])
            if outcome["status"] in {"mature", "censored_missing_bar", "censored_boundary"}:
                completed.add(forecast_id)
            if outcome["status"] == "mature":
                initial = initial_events[utc_time(forecast["origin_bar_end"])]
                event_id = initial["event_id"]
                versions[event_id] += 1
                journal.append("events", append_post_event(initial, forecast, outcome, version=versions[event_id], available_at=clock))
    # News is imported with its declared logical known_at, but queries remain <=T.
    journal.append_many([("news", row) for row in bundle["news"] if row["symbol"] == symbol and utc_time(row["available_at"]) <= cutoff])
    if not visible:
        return {"bars": pd.DataFrame(), "events": [], "versions": [], "details": {}, "forecasts": [], "assessments": [], "news": [], "reviews": [], "context": pd.DataFrame(), "as_of": as_of}
    bars = bars_as_of(visible, as_of)
    # U03 context API uses all already known past sessions, never current/future context.
    past = [row for row in bundle["rows"] if row["symbol"] == symbol and utc_time(row["available_at"]) <= cutoff]
    context = causal_context(bars_as_of(past, as_of), bundle["calendar"])
    context = context.loc[context.session_id.eq(session_id)]
    event_versions = [row for row in journal.query("events", as_of=as_of, symbol=symbol) if row["session_id"] == session_id]
    events = [row for row in journal.query("events", as_of=as_of, symbol=symbol, latest=True) if row["session_id"] == session_id]
    session_forecasts = [row for row in journal.query("forecasts", as_of=as_of, symbol=symbol) if row["session_id"] == session_id]
    forecast_ids = {row["forecast_id"] for row in session_forecasts}
    assessments = [row for row in journal.query("assessments", as_of=as_of, symbol=symbol) if row["forecast_id"] in forecast_ids]
    event_ids = {row["event_id"] for row in events}
    reviews = [row for row in journal.query("reviews", as_of=as_of, symbol=symbol) if row["event_id"] in event_ids]
    return {"bars": bars, "events": events, "versions": event_versions, "details": details, "forecasts": session_forecasts,
            "assessments": assessments, "news": journal.query("news", as_of=as_of, symbol=symbol), "reviews": reviews,
            "context": context, "as_of": as_of}


def filtered_events(snapshot, *, kinds=None, stages=None, review_status="all"):
    if review_status not in {"all", "reviewed", "unreviewed"}:
        raise DataContractError("Unknown review filter")
    reviewed = {(row["event_id"], row["event_version"]) for row in snapshot["reviews"]}
    source = [row for row in snapshot["versions"] if row["stage"] == "initial" and row["version"] == 1] if stages is not None and set(stages) == {"initial"} else snapshot["events"]
    return [row for row in source if (kinds is None or row["kind"] in kinds) and
            (stages is None or row["stage"] in stages) and
            (review_status == "all" or ((row["event_id"], row["version"]) in reviewed) == (review_status == "reviewed"))]


def event_csv(events):
    if not events:
        return "event_id,version,available_at,kind,stage\n".encode("utf-8-sig")
    rows = [{key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value
             for key, value in row.items()} for row in events]
    # Spreadsheet formula injection guard for future user-supplied text fields.
    rows = [{key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")) else value
             for key, value in row.items()} for row in rows]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def forecast_view(snapshot, event, bundle):
    output = []
    for record in snapshot["forecasts"]:
        if utc_time(record["origin_bar_end"]) != utc_time(event["event_end"]):
            continue
        available = [row for row in snapshot["assessments"] if row["forecast_id"] == record["forecast_id"]]
        assessment = max(available, key=lambda row: utc_time(row["as_of"])) if available else None
        rank = rank_zero_residual(assessment, record, bundle["residual"], expected_sha256=content_hash(bundle["residual"])) if assessment else {"rank": None, "reference_count": 0}
        evidence = assessment["post_evidence"] if assessment else None
        output.append({"horizon_bars": record["horizon_bars"], "created_at": record["created_at"], "target_bar_end": record["target_bar_end"],
                       "prediction": record["prediction"], "status": assessment["status"] if assessment else "pending",
                       "realized_return": evidence["realized_return"] if evidence else None, "residual": evidence["residual"] if evidence else None,
                       "rank": rank["rank"], "reference_count": rank["reference_count"], "forecast_id": record["forecast_id"]})
    return output
