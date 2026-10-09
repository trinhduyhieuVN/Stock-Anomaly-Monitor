"""Purged causal training samples and one fixed Ridge trial, separate from v1."""
from collections import Counter
from datetime import timedelta
import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stock_monitor.io import DataContractError
from .contracts import utc_time
from .forecast import MODEL_SHA256, validate_forecast
from .segment_replay import return_view
from .synthetic import content_hash


def build_forecast_samples(observations, *, session_splits, replicate_id):
    """Targets stay inside split/session/period/gap; no holdout fit samples."""
    if not session_splits or set(session_splits.values()) - {"train", "validation"}:
        raise DataContractError("Fit sample builder only accepts explicit train/validation sessions")
    scoped = [row for row in observations if row["session_id"] in session_splits]
    if not scoped or any(int(row["bar_revision"]) != 0 for row in scoped):
        raise DataContractError("Training/calibration warmup requires initial-only immutable bars")
    view = return_view(scoped, as_of=max((row["available_at"] for row in scoped), key=utc_time), split="train")
    prices = {(row["symbol"], utc_time(row["timestamp"])): row for row in scoped}
    groups = {}
    for row in view:
        groups.setdefault((row["symbol"], row["session_id"], row["segment_id"], row["period"]), []).append(row)
    samples, purged = [], Counter()
    for group in groups.values():
        for index, origin in enumerate(group):
            history = group[max(0, index - 5):index + 1]
            if len(history) != 6 or any(row["log_return"] is None for row in history):
                purged["insufficient_causal_features"] += 1
                continue
            as_of = utc_time(origin["available_at"])
            if any(utc_time(row["information_available_at"]) > as_of for row in history):
                purged["late_feature"] += 1
                continue
            path = group[index + 1:index + 4]
            if len(path) != 3 or any(utc_time(row["timestamp"]) != utc_time(origin["timestamp"]) + timedelta(minutes=5 * offset)
                                     for offset, row in enumerate(path, 1)):
                purged["cross_gap_period_session_or_split"] += 1
                continue
            split = session_splits[origin["session_id"]]
            if any(session_splits.get(row["session_id"]) != split for row in history + path):
                purged["cross_split"] += 1
                continue
            if as_of >= utc_time(path[0]["timestamp"]):
                purged["late_origin_backfill"] += 1
                continue
            close = float(prices[(origin["symbol"], utc_time(origin["timestamp"]))]["close"])
            targets = [math.log(float(prices[(row["symbol"], utc_time(row["timestamp"]))]["close"]) / close)
                       for row in (path[0], path[2])]
            samples.append({"sample_id": f"{replicate_id}:{origin['symbol']}:{origin['timestamp']}",
                            "symbol": origin["symbol"], "session_id": origin["session_id"], "segment_id": origin["segment_id"],
                            "origin_bar_end": origin["timestamp"], "split": split,
                            "feature_end": origin["timestamp"], "input_available_at": origin["available_at"],
                            "features": [row["log_return"] for row in history], "targets_h1_h3": targets,
                            "target_bar_ends": [path[0]["timestamp"], path[2]["timestamp"]],
                            "outcome_available_h1_h3": [path[0]["available_at"], max((row["available_at"] for row in path), key=utc_time)],
                            "input_sha256": content_hash(history), "source_mode": "SYNTHETIC"})
    return {"samples": samples, "purged": dict(purged)}


def fit_ridge_trial(train, validation):
    if not train or not validation or any(row["split"] != "train" for row in train) or any(row["split"] != "validation" for row in validation):
        raise DataContractError("Ridge requires separate nonempty train/validation")
    train_end = max(utc_time(value) for row in train for value in row["target_bar_ends"])
    train_available_end = max(utc_time(value) for row in train for value in row["outcome_available_h1_h3"])
    if (train_end >= min(utc_time(row["origin_bar_end"]) for row in validation) or
            train_available_end >= min(utc_time(row["input_available_at"]) for row in validation)):
        raise DataContractError("Ridge labels overlap validation input event/availability boundary")
    x_train, y_train = np.asarray([row["features"] for row in train]), np.asarray([row["targets_h1_h3"] for row in train])
    x_val, y_val = np.asarray([row["features"] for row in validation]), np.asarray([row["targets_h1_h3"] for row in validation])
    if x_train.shape[1] != 6 or y_train.shape[1] != 2 or x_val.shape[1] != 6 or y_val.shape[1] != 2:
        raise DataContractError("Ridge trial feature/horizon shape mismatch")
    if not all(np.isfinite(array).all() for array in (x_train, y_train, x_val, y_val)):
        raise DataContractError("Nonfinite Ridge data")
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
    predictions = model.predict(scaler.transform(x_val))
    zero_mae = np.abs(y_val).mean(axis=0)
    ridge_mae = np.abs(predictions - y_val).mean(axis=0)
    retain = bool(ridge_mae.mean() <= .99 * zero_mae.mean() and (ridge_mae <= zero_mae).all())
    return {"scaler": scaler, "model": model, "validation_predictions": predictions,
            "diagnostic": {"ridge_alpha": 1., "ridge_configurations": 1, "train_samples": len(train),
                           "validation_samples": len(validation), "train_sha256": content_hash(train),
                           "validation_sha256": content_hash(validation), "train_target_end": train_end.isoformat(),
                           "train_outcome_available_end": train_available_end.isoformat(),
                           "zero_MAE_h1_h3": zero_mae.tolist(), "ridge_MAE_h1_h3": ridge_mae.tolist(),
                           "ridge_retention_rule_met": retain, "selection_uses_holdout": False}}


def calibrate_zero_residuals(validation, *, minimum_samples=100, quantile=.98):
    if not validation or any(row["split"] != "validation" for row in validation):
        raise DataContractError("Residual references require matured validation samples")
    if type(minimum_samples) is not int or minimum_samples < 1 or isinstance(quantile, bool) or not 0 < quantile < 1:
        raise DataContractError("Invalid residual calibration controls")
    references = {}
    for row in validation:
        for index, horizon in enumerate((1, 3)):
            if not math.isfinite(row["targets_h1_h3"][index]):
                raise DataContractError("Nonfinite matured calibration residual")
            references.setdefault(f"{row['symbol']}:h{horizon}", []).append(abs(row["targets_h1_h3"][index]))
    return {"algorithm": "zero_return_absolute_residual_midrank", "model_sha256": MODEL_SHA256,
            "minimum_samples": minimum_samples, "quantile": quantile,
            "references": {key: sorted(values) for key, values in references.items()},
            "validation_sha256": content_hash(validation),
            "validation_outcome_available_end": max((value for row in validation for value in row["outcome_available_h1_h3"]), key=utc_time)}


def rank_zero_residual(assessment, forecast, calibration, *, expected_sha256):
    forecast = validate_forecast(forecast)
    if content_hash(calibration) != expected_sha256 or forecast["model_sha256"] != calibration["model_sha256"]:
        raise DataContractError("Frozen residual calibration/model mismatch")
    if assessment["forecast_id"] != forecast["forecast_id"]:
        raise DataContractError("Residual refers to a different forecast")
    if assessment["status"] != "mature":
        return {"rank": None, "reference_count": 0, "reason": assessment["status"]}
    if assessment["post_evidence"]["forecast_sha256"] != content_hash(forecast):
        raise DataContractError("Mature evidence forecast hash mismatch")
    if utc_time(forecast["as_of"]) <= utc_time(calibration["validation_outcome_available_end"]):
        raise DataContractError("Residual evaluation overlaps validation outcome availability")
    references = calibration["references"].get(f"{forecast['symbol']}:h{forecast['horizon_bars']}", [])
    if len(references) < calibration["minimum_samples"]:
        return {"rank": None, "reference_count": len(references), "reason": "insufficient_reference"}
    value = abs(assessment["post_evidence"]["residual"])
    array = np.asarray(references)
    rank = float(((array < value).sum() + .5 * (array == value).sum()) / len(array))
    return {"rank": rank, "reference_count": len(references), "reason": "absolute_residual_midrank_not_probability"}
