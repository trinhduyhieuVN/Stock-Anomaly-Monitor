"""Retrospective review projection, never an observed historical receipt."""
from datetime import datetime, timezone
import hashlib

import pandas as pd

from stock_monitor.upgrade.contracts import validate_event
from .history import AVAILABILITY_BASIS


def review_snapshot(bundle, history, journal, *, replay):
    """Materialize only the approved bar-end prefix in a model-scoped journal."""
    model = history["model"]
    model_file = "best_checkpoint.pt" if model == "lstm" else "models.joblib"
    model_hash = next(value for name, value in bundle["pins"].items() if name.endswith("/" + model_file))
    data_name = (bundle["dataset_directory"] / "bars.parquet").relative_to(bundle["root"]).as_posix()
    events, details = [], {}
    for row in history["scores"].to_dict("records"):
        stamp = row["timestamp"].isoformat()
        identity = "|".join((bundle["key"], model, row["symbol"], row["session_id"], stamp))
        event_id = hashlib.sha256(identity.encode()).hexdigest()
        scored = pd.notna(row["score"])
        event = {"schema_version": 2, "event_unit": "bar_5m", "event_id": event_id,
                 "symbol": row["symbol"], "session_id": row["session_id"], "segment_id": str(row["segment_id"]),
                 "version": 1, "event_start": stamp, "event_end": stamp, "detected_at": stamp,
                 "available_at": stamp, "as_of": stamp, "stage": "initial",
                 "kind": ("point" if bool(row["alert"]) else "normal") if scored else "insufficient_data",
                 "model_run": history["provenance"]["model_run"] + ":" + model,
                 "model_sha256": model_hash, "data_sha256": bundle["pins"][data_name],
                 "data_revision": data_name, "source_mode": "HISTORICAL_REPLAY",
                 "reason_codes": ["saved_v1_alert" if scored and bool(row["alert"]) else "saved_v1_below_threshold" if scored else "frozen_v1_window_unavailable"],
                 "score": float(row["score"]) if scored else None,
                 "threshold": float(row["threshold"]) if scored else None, "decision_method": "baseline"}
        events.append(validate_event(event, as_of=history["as_of"]))
        details[event_id] = row
    journal.append_many([("events", event) for event in events])
    review_as_of = history["as_of"] if replay else datetime.now(timezone.utc).isoformat()
    ids = {row["event_id"] for row in events}
    reviews = [row for row in journal.query("reviews", as_of=review_as_of, symbol=history["symbol"]) if row["event_id"] in ids]
    return {"bars": history["bars"], "events": events, "versions": events, "reviews": reviews,
            "details": details, "as_of": review_as_of, "bar_as_of": history["as_of"],
            "availability_basis": AVAILABILITY_BASIS, "model": model, "received_at_utc": None,
            "source_mode": "HISTORICAL_REPLAY"}
