"""Lagged bar context and frozen per-model empirical ranks (not probabilities)."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from stock_monitor.features import compute_features
from stock_monitor.io import DataContractError
from stock_monitor.realtime import verify_frozen_scoring_manifest
from .contracts import utc_time
from .synthetic import content_hash

CONTEXT_FIELDS = ["lagged_volatility", "lagged_relative_volume", "past_trend"]


def causal_context(bars, calendar, *, lookback=6, history_sessions=20, minimum_history_sessions=10):
    """Input is one revision per bar with explicit observation availability.

    Delayed history cannot be used for an earlier decision even if a retrospective
    view already includes it. Replay must materialize the actual view at each T.
    """
    if type(lookback) is not int or lookback < 2 or "available_at" not in bars:
        raise DataContractError("Context requires explicit availability and integer lookback >=2")
    frame = compute_features(bars, calendar, history_sessions=history_sessions,
                             min_history_sessions=minimum_history_sessions, warmup_sessions=0).frame
    if frame.source.nunique() != 1 or frame.source.iloc[0] not in {"SYNTHETIC", "VCI"}:
        raise DataContractError("Context must use one declared source")
    frame["available_at"] = pd.to_datetime(frame.available_at.map(lambda x: utc_time(str(x))), utc=True)
    if frame.available_at.lt(frame.timestamp).any():
        raise DataContractError("Observation availability precedes bar close")
    if "decision_as_of" in frame:
        frame["decision_as_of"] = pd.to_datetime(frame.decision_as_of.map(lambda x: utc_time(str(x))), utc=True)
    else:
        frame["decision_as_of"] = frame.available_at
    group = frame.groupby(["symbol", "session_id", "segment_id"], sort=False)
    frame["lagged_volatility"] = group.log_return.transform(lambda x: x.shift(1).rolling(lookback, min_periods=lookback).std(ddof=0))
    frame["past_trend"] = group.log_return.transform(lambda x: x.shift(1).rolling(lookback, min_periods=lookback).mean())
    frame["lagged_relative_volume"] = group.volume_surprise.shift(1)
    frame["context_end"] = group.timestamp.shift(1)
    frame["context_period"] = frame.slot.str.split(":").str[0]
    frame["context_hour"] = frame.slot.str.split(":").str[1]
    # Conservative max over all earlier observations covers lag and prior-day slot references.
    frame["information_available_at"] = frame.groupby("symbol", sort=False).available_at.cummax()
    finite = np.isfinite(frame[CONTEXT_FIELDS].to_numpy(dtype=float)).all(axis=1)
    mature = frame.information_available_at.le(frame.decision_as_of) & frame.available_at.le(frame.decision_as_of)
    frame["context_eligible"] = finite & mature & frame.feature_eligible
    frame["context_reason"] = np.where(~mature, "history_not_available_at_decision",
                                       np.where(~finite | ~frame.feature_eligible, "insufficient_history", "ready"))
    # Price-trend proxy only: no aggressor side, order flow or liquidity assertion.
    frame["price_trend_contradiction"] = (frame.log_return * frame.past_trend < 0).where(frame.context_eligible)
    return frame


def _eligible(frame):
    required = {"symbol", "timestamp", "source", "split", "context_eligible", "context_period",
                "context_hour", "decision_as_of", "context_end", "context_reason",
                "information_available_at", *CONTEXT_FIELDS}
    if required - set(frame.columns) or frame.empty:
        raise DataContractError("Missing/nonempty context contract")
    rows = frame.loc[frame.context_eligible].copy()
    if not np.isfinite(rows[CONTEXT_FIELDS].to_numpy(dtype=float)).all():
        raise DataContractError("Eligible context has nonfinite statistics")
    if not rows.context_end.lt(rows.timestamp).all():
        raise DataContractError("Context must strictly precede the current bar")
    if not rows.information_available_at.le(rows.decision_as_of).all():
        raise DataContractError("Eligible context uses unavailable evidence")
    if frame.source.nunique() != 1:
        raise DataContractError("Cannot mix sources in a contextual calibrator")
    return rows


def fit_context_bins(train):
    if not train.split.eq("train").all():
        raise DataContractError("Fit context bins only on train")
    rows = _eligible(train)
    if len(rows) < 2:
        raise DataContractError("Insufficient train context")
    cutoffs = {key: float(rows[key].median()) for key in CONTEXT_FIELDS[:2]}
    return {"schema_version": 2, "source": train.source.iloc[0], "cutoffs": cutoffs,
            "fit_rows": len(rows), "fit_event_end": train.timestamp.max().isoformat(),
            "fit_available_at": train.decision_as_of.max().isoformat(),
            "zero_variation_fields": [key for key in cutoffs if rows[key].nunique() == 1],
            "bin_rule": "low <= train median < high; no division by zero",
            "context_rule": "symbol/period/hour/lagged volatility/lagged relative-volume"}


def context_keys(frame, bins):
    _eligible(frame)
    if frame.source.iloc[0] != bins["source"]:
        raise DataContractError("Context source differs from fit source")
    key = frame.symbol.astype(str) + "|" + frame.context_period + "|" + frame.context_hour
    for name in CONTEXT_FIELDS[:2]:
        cutoff = bins["cutoffs"][name]
        if not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff):
            raise DataContractError("Invalid frozen context cutoff")
        key = key + "|" + np.where(frame[name].le(cutoff), "low", "high")
    return key.where(frame.context_eligible, "unavailable")


def calibrate_context_ranks(validation, bins, models, *, frozen_binding, minimum_samples=100, quantile=.98):
    if type(minimum_samples) is not int or minimum_samples < 1 or not 0 < quantile < 1:
        raise DataContractError("Invalid reference minimum/quantile")
    if (not validation.split.eq("validation").all() or not models or len(set(models)) != len(models) or
            any(model not in {"robust_zscore", "isolation_forest", "lstm"} for model in models)):
        raise DataContractError("References must come from validation, with unique model columns")
    if set(models) - set(validation.columns):
        raise DataContractError("Missing raw model scores")
    if (validation.timestamp.min() <= pd.Timestamp(bins["fit_event_end"]) or
            validation.decision_as_of.min() <= pd.Timestamp(bins["fit_available_at"])):
        raise DataContractError("Validation must follow train in event/availability time")
    _eligible(validation)
    keys = context_keys(validation, bins)
    references, excluded = {}, {}
    for model in models:
        if validation[model].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise DataContractError("Raw model scores cannot be boolean")
        values = pd.to_numeric(validation[model], errors="coerce")
        good = validation.context_eligible & np.isfinite(values)
        excluded[model] = int((~good).sum())
        selected = validation.loc[good].copy()
        selected["_key"], selected["_value"] = keys.loc[good], values.loc[good]
        groups = {"global": sorted(selected._value.astype(float).tolist())}
        groups.update({"symbol:" + str(symbol): sorted(rows._value.astype(float).tolist())
                       for symbol, rows in selected.groupby("symbol")})
        groups.update({"context:" + str(key): sorted(rows._value.astype(float).tolist())
                       for key, rows in selected.groupby("_key")})
        references[model] = groups
    return {"schema_version": 2, "source": bins["source"], "bins": deepcopy(bins), "bins_sha256": content_hash(bins),
            "models": list(models), "references": references, "excluded_validation_rows": excluded,
            "minimum_samples": minimum_samples, "quantile": quantile,
            "calibration_event_end": validation.timestamp.max().isoformat(),
            "calibration_available_at": validation.decision_as_of.max().isoformat(),
            "frozen_binding": deepcopy(frozen_binding), "rank_rule": "(less + 0.5*equal)/n; empirical midrank, not probability",
            "fallback_order": ["context", "symbol", "global", "unavailable"]}


def rank_points(points, calibration, *, expected_sha256):
    """Frozen references are never updated by evaluation points or their labels."""
    if content_hash(calibration) != expected_sha256 or content_hash(calibration["bins"]) != calibration["bins_sha256"]:
        raise DataContractError("Frozen contextual artifact checksum mismatch")
    if points.source.iloc[0] != calibration["source"]:
        raise DataContractError("Cannot apply synthetic references to real observations")
    if (points.timestamp.min() <= pd.Timestamp(calibration["calibration_event_end"]) or
            points.decision_as_of.min() < pd.Timestamp(calibration["calibration_available_at"])):
        raise DataContractError("Point precedes frozen reference availability")
    result = points.copy(deep=True)
    result["context_key"] = context_keys(points, calibration["bins"])
    for model in calibration["models"]:
        if model not in result:
            raise DataContractError("Missing bound raw model score")
        records = []
        for row in result.itertuples(index=False):
            score = getattr(row, model)
            ready_score = isinstance(score, (int, float, np.number)) and not isinstance(score, (bool, np.bool_)) and np.isfinite(score)
            reason = row.context_reason if not row.context_eligible else "raw_score_unavailable" if not ready_score else "sparse_reference"
            rank, alert, threshold, used, n = None, None, None, "unavailable", 0
            if row.context_eligible and ready_score:
                for level, key in [("context", "context:" + row.context_key), ("symbol", "symbol:" + row.symbol), ("global", "global")]:
                    values = np.asarray(calibration["references"][model].get(key, []), dtype=float)
                    if not np.isfinite(values).all():
                        raise DataContractError("Nonfinite frozen reference")
                    n = len(values)
                    if n >= calibration["minimum_samples"]:
                        rank = float((np.count_nonzero(values < score) + .5 * np.count_nonzero(values == score)) / n)
                        threshold = float(np.quantile(values, calibration["quantile"]))
                        alert, used, reason = rank > calibration["quantile"], level, "ready"
                        break
            records.append((rank, n, used, reason, threshold, alert))
        for index, suffix in enumerate(("rank", "reference_count", "fallback_level", "context_reason", "raw_reference_threshold", "contextual_alert")):
            result[f"{model}_{suffix}"] = [record[index] for record in records]
    return result


def rank_frozen_points(points, calibration, *, expected_sha256, project_root, manifest_path="demo/manifest.local.json"):
    verified = verify_frozen_scoring_manifest(project_root, manifest_path)
    if verified != calibration["frozen_binding"]:
        raise DataContractError("Frozen model/scaler/threshold binding differs from contextual artifact")
    return rank_points(points, calibration, expected_sha256=expected_sha256)
