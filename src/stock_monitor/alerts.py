"""Alert decoration and grouping without crossing market-data segments."""

import pandas as pd

from .io import DataContractError


def apply_thresholds(scores, thresholds):
    result = scores.copy()
    mapping = {symbol: values["threshold"] for symbol, values in thresholds.items()}
    result["threshold"] = result.symbol.map(mapping)
    if result["threshold"].isna().any():
        raise DataContractError("Every score symbol requires a calibrated threshold")
    result["alert"] = result.score.gt(result.threshold)
    result["reason"] = result["alert"].map({True: "lstm_score_above_validation_q98", False: ""})
    return result


def group_consecutive_alerts(scores):
    required = {"symbol", "segment_id", "timestamp", "window_start", "score", "threshold", "alert", "reason"}
    if not required.issubset(scores.columns):
        raise DataContractError("Alert grouping input is missing required columns")
    ordered = scores.sort_values(["symbol", "timestamp", "window_start"]).reset_index(drop=True)
    records = []
    active = None
    previous = None
    for row in ordered.itertuples(index=False):
        consecutive = (active is not None and previous is not None and previous.alert and
                       row.symbol == previous.symbol and row.segment_id == previous.segment_id and
                       row.timestamp - previous.timestamp == pd.Timedelta(minutes=5))
        if row.alert:
            if not consecutive:
                if active is not None:
                    records.append(active)
                active = {"symbol": row.symbol, "segment_id": row.segment_id,
                          "start_timestamp": row.timestamp, "end_timestamp": row.timestamp,
                          "window_start": row.window_start, "bars": 1, "max_score": float(row.score),
                          "threshold": float(row.threshold), "reason": row.reason}
            else:
                active["end_timestamp"] = row.timestamp
                active["bars"] += 1
                active["max_score"] = max(active["max_score"], float(row.score))
        elif active is not None:
            records.append(active)
            active = None
        previous = row
    if active is not None:
        records.append(active)
    columns = ["symbol", "segment_id", "start_timestamp", "end_timestamp", "window_start",
               "bars", "max_score", "threshold", "reason"]
    return pd.DataFrame(records, columns=columns)
