"""Optional official CAPA1.2 adapter; isolated imports, retrospective only."""
from datetime import datetime, timedelta
import hashlib
from importlib.metadata import version
import json
import math

import pandas as pd


def payload_hash(value):
    # Same canonical JSON, without importing v1/PyYAML into the isolated env.
    return hashlib.sha256((json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                      separators=(",", ":")) + "\n").encode("utf-8")).hexdigest()


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("CAPA timestamps must explicitly be UTC")
    return result


def map_interval(rows, interval):
    """Native left-closed ilocs [a,b) -> inclusive bar-end [rows[a],rows[b-1]]."""
    if not isinstance(interval, pd.Interval) or interval.closed != "left":
        raise ValueError("Expected official left-closed ilocs interval")
    start, end = interval.left, interval.right
    if isinstance(start, bool) or isinstance(end, bool) or int(start) != start or int(end) != end:
        raise ValueError("Expected integral interval indices")
    start, end = int(start), int(end)
    if not 0 <= start < end <= len(rows):
        raise ValueError("CAPA interval outside chunk")
    first, last = rows[start], rows[end - 1]
    return {"symbol": first["symbol"], "session_id": first["session_id"], "segment_id": first["segment_id"],
            "event_start": first["timestamp"], "event_end": last["timestamp"], "length": end - start,
            "kind": "point" if end - start == 1 else "collective", "ilocs_start": start, "ilocs_end_exclusive": end,
            "method": "CAPA", "mode": "RETROSPECTIVE_SYNTHETIC", "score": None,
            "score_status": "native_predict_returns_intervals_not_numeric_scores"}


def run_capa_batch(rows, normalizer, *, expected_normalizer_sha256, penalty):
    """Never train/relabel raw or cross invalid returns/gaps/session/seed boundaries."""
    if payload_hash(normalizer) != expected_normalizer_sha256:
        raise ValueError("Frozen normalizer hash mismatch")
    if normalizer["source_mode"] != "SYNTHETIC":
        raise ValueError("This optional adapter is synthetic only")
    if isinstance(penalty, bool) or not isinstance(penalty, (float, int)) or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Expected positive constant penalty")
    if version("sktime") != "1.2.0":
        raise ValueError("Expected pinned official sktime1.2.0")
    from sktime.detection.capa import CAPA
    rows = sorted(rows, key=lambda row: (row["symbol"], stamp(row["timestamp"])))
    seen, chunks, current, previous = set(), [], [], None
    for row in rows:
        if set(row) & {"truth", "label", "kind", "scenario", "alert"}:
            raise ValueError("No labels/truth in CAPA inputs")
        key = (row["symbol"], stamp(row["timestamp"]))
        if key in seen:
            raise ValueError("Duplicate seed/bar: call independent replicas separately")
        seen.add(key)
        if row["source_mode"] != "SYNTHETIC" or row["split"] not in {"validation", "evaluation"}:
            raise ValueError("Unsupported CAPA source/split")
        if stamp(row["timestamp"]) <= stamp(normalizer["train_event_end"]):
            raise ValueError("CAPA scoring overlaps normalizer training")
        if stamp(row["information_available_at"]) > stamp(row["decision_as_of"]):
            raise ValueError("Unavailable CAPA evidence")
        continuous = previous is not None and all(row[key] == previous[key] for key in
            ("symbol", "session_id", "segment_id", "period", "split")) and (
                stamp(row["timestamp"]) - stamp(previous["timestamp"]) == timedelta(minutes=5))
        fitted = normalizer["symbols"].get(row["symbol"], {"ready": False})
        raw = row["log_return"]
        valid = fitted["ready"] and raw is not None and not isinstance(raw, bool) and math.isfinite(raw)
        if not continuous or not valid:
            if current:
                chunks.append(current)
            current = []
        if valid:
            current.append(row)
        previous = row
    if current:
        chunks.append(current)
    events, eligible = [], 0
    for chunk in chunks:
        if len(chunk) < 3:
            continue
        fitted = normalizer["symbols"][chunk[0]["symbol"]]
        z = [(row["log_return"] - fitted["median"]) / fitted["scale"] for row in chunk]
        if not all(math.isfinite(value) for value in z):
            raise ValueError("Nonfinite CAPA standardized residual")
        detector = CAPA(min_segment_length=3, max_segment_length=12, segment_penalty=float(penalty), point_penalty=float(penalty))
        native = detector.fit_predict(pd.DataFrame({"z": z}))
        eligible += len(chunk)
        for interval in native["ilocs"]:
            record = map_interval(chunk, interval)
            record.update(as_of=chunk[-1]["decision_as_of"], normalizer_sha256=expected_normalizer_sha256,
                          data_sha256=payload_hash(chunk), penalty=float(penalty), package="sktime==1.2.0")
            events.append(record)
    return {"events": events, "eligible_bars": eligible, "excluded_bars": len(rows) - eligible,
            "covered_bars": sum(event["length"] for event in events), "mode": "RETROSPECTIVE_SYNTHETIC",
            "online_detection_latency": None, "implementation": "sktime.detection.capa.CAPA", "v1_model_fits": 0}
