"""Separate offline ablations; reuse frozen diagnostic scores, never retune U04."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from stock_monitor.baselines import score_baselines
from stock_monitor.io import DataContractError
from stock_monitor.scoring import score_reconstructions
from stock_monitor.training import reconstruct
from stock_monitor.windows import make_windows, transform_features
from .context import causal_context
from .segments import _scan, _rank
from .synthetic import content_hash


MODELS = ("robust_zscore", "isolation_forest", "lstm")


def extract_frozen_windows(bars, calendar, resources, *, endpoint=None):
    """Availability-resolved input only; no fitting, context or label feedback."""
    frame = causal_context(bars, calendar)
    days = calendar["dates"]
    frame["split"] = np.where(frame.session_id.isin(days[:15]), "train",
                             np.where(frame.session_id.isin(days[15:20]), "validation", "test"))
    if endpoint:
        symbol, stamp = endpoint
        frame = frame.loc[frame.symbol.eq(symbol) & frame.session_id.eq(stamp.tz_convert("Asia/Ho_Chi_Minh").date().isoformat())
                          & frame.timestamp.le(stamp)].reset_index(drop=True)
    scaled = transform_features(frame, resources["scalers"])
    windows = make_windows(scaled, window_length=6)
    rows = frame.iloc[windows.index.end_row.to_numpy(dtype=int)].reset_index(drop=True)
    allowed = rows.information_available_at.le(rows.decision_as_of).to_numpy()
    if endpoint:
        allowed = allowed & rows.timestamp.eq(endpoint[1]).to_numpy()
    index = windows.index.loc[allowed].reset_index(drop=True)
    index["window_id"] = np.arange(len(index))
    return windows.values[allowed], index, rows.loc[allowed].reset_index(drop=True)


def infer_frozen_windows(values, index, rows, resources, feature_order):
    if values.shape[1:] != (6, 4) or len(rows) != len(index):
        raise DataContractError("Frozen six-by-four window contract differs")
    if not len(rows):
        result = rows.copy()
        for model in MODELS:
            result[model] = pd.Series(dtype=float)
        return result
    scores = score_baselines(resources["baseline"], values, index)
    lstm = score_reconstructions(values, reconstruct(resources["lstm"], values), index,
                                feature_order, expected_feature_order=feature_order)
    result = rows.copy()
    for model in MODELS[:2]:
        result[model] = scores[model].to_numpy()
    result["lstm"] = lstm.score.to_numpy()
    return result


def variant_references(replicates, calibration):
    """Two predetermined validation-only variants, no labels/truth accepted."""
    maxima, rolling = {}, {}
    for rows in replicates:
        if not rows or any(row["split"] != "validation" or set(row) & {"truth", "label", "scenario"} for row in rows):
            raise DataContractError("Validation residuals without truth required")
        for endpoint in _scan(rows, calibration["normalizer"]):
            if not endpoint["ready"]:
                continue
            symbol, mask = endpoint["row"]["symbol"], endpoint["coverage_mask"]
            if mask:
                maxima.setdefault(symbol, {}).setdefault(mask, []).append(max(
                    (row["raw_score"] for row in endpoint["candidates"]), default=0.))
            six = next((row for row in endpoint["candidates"] if row["length"] == 6), None)
            if six:
                rolling.setdefault(symbol, []).append(six["raw_score"])
    return {"unguarded_maxima": {symbol: {mask: sorted(values) for mask, values in groups.items()}
                                for symbol, groups in maxima.items()},
            "rolling_six": {symbol: sorted(values) for symbol, values in rolling.items()},
            "calibration_sha256": content_hash(calibration), "truth_used": False}


def variant_decision(detail, references, calibration, variant):
    """Score an already-causal saved endpoint; no future/oracle deduplication."""
    if references["calibration_sha256"] != content_hash(calibration):
        raise DataContractError("Variant references do not bind frozen base")
    if variant not in {"v2_base", "no_context", "no_segment", "no_single_point_guard", "no_post_event", "simple_rolling"}:
        raise DataContractError("Unknown predeclared ablation")
    event = deepcopy(detail["event"])
    event["event_id"] = content_hash({"original_id": event["event_id"], "variant": variant})
    if variant in {"v2_base", "no_context", "no_post_event"}:
        return event
    q, minimum, symbol = calibration["quantile"], calibration["minimum_samples"], event["symbol"]
    point = next((row for row in detail["candidates"] if "length" not in row), None)
    candidates = [row for row in detail["candidates"] if "length" in row]
    if point:
        event["event_start"] = event["event_end"]
    point_rank = detail["point_rank"]
    if variant == "no_segment":
        event["kind"] = "insufficient_data" if point_rank is None else "point" if point_rank > q else "normal"
        event["score"] = point_rank
        return event
    if variant == "simple_rolling":
        best = next((row for row in candidates if row["length"] == 6), None)
        refs = references["rolling_six"].get(symbol, [])
    else:
        best = min(candidates, key=lambda row: (-row["raw_score"], row["length"])) if candidates else None
        refs = references["unguarded_maxima"].get(symbol, {}).get(detail["coverage_mask"], [])
    rank, _ = _rank(best["raw_score"], refs, minimum, q) if best else (None, None)
    if rank is None or (variant != "simple_rolling" and point_rank is None):
        event["kind"], event["score"] = "insufficient_data", None
    elif rank > q and (variant == "simple_rolling" or rank >= point_rank):
        event.update(kind="collective", event_start=best["event_start"], score=rank)
    elif variant != "simple_rolling" and point_rank > q:
        event.update(kind="point", score=point_rank)
    else:
        event.update(kind="normal", score=rank)
    return event
