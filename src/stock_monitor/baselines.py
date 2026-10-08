"""Train-only robust z-score and Isolation Forest baselines."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .io import DataContractError


@dataclass
class RobustZScore:
    median: np.ndarray
    mad: np.ndarray
    scale: np.ndarray
    zero_mad_features: tuple[str, ...]

    def score_samples(self, values):
        values = _finite_matrix(values, "Robust z-score input")
        if values.shape[1] != len(self.median):
            raise DataContractError("Robust z-score feature count differs from the fitted model")
        return np.max(np.abs(values - self.median) / self.scale, axis=1)


@dataclass
class BaselineBundle:
    feature_order: tuple[str, ...]
    robust_models: dict[str, RobustZScore]
    isolation_forests: dict[str, IsolationForest]
    fit_report: dict


def _finite_matrix(values, label):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise DataContractError(f"{label} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise DataContractError(f"{label} must contain only finite values")
    return matrix


def fit_robust_zscore(values, feature_order):
    values = _finite_matrix(values, "Robust z-score train data")
    feature_order = tuple(feature_order)
    if values.shape[1] != len(feature_order):
        raise DataContractError("Feature order length differs from robust z-score train data")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    zero_mask = mad == 0
    scale = 1.4826 * mad
    scale[zero_mask] = 1.0
    return RobustZScore(median, mad, scale,
                        tuple(name for name, is_zero in zip(feature_order, zero_mask) if is_zero))


def _validate_windows(windows, index, feature_order):
    values = np.asarray(windows)
    required = {"window_id", "symbol", "split", "timestamp"}
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] != len(feature_order):
        raise DataContractError("Windows must have shape [samples, window, feature_order]")
    if len(values) != len(index) or not required.issubset(index.columns):
        raise DataContractError("Window values and index contract differ")
    if not np.array_equal(index.window_id.to_numpy(), np.arange(len(index))):
        raise DataContractError("Window index must be ordered by contiguous window_id")
    return _finite_matrix(values[:, -1, :], "Last-bar features")


def fit_baselines(windows, index, feature_order, *, random_state=42):
    feature_order = tuple(feature_order)
    last_bar = _validate_windows(windows, index, feature_order)
    robust_models, isolation_forests, fit_report = {}, {}, {}
    for symbol in sorted(index.symbol.unique()):
        train_mask = index.symbol.eq(symbol) & index.split.eq("train")
        train_values = last_bar[train_mask.to_numpy()]
        if len(train_values) == 0:
            raise DataContractError(f"No train windows for baseline: {symbol}")
        robust = fit_robust_zscore(train_values, feature_order)
        forest = IsolationForest(random_state=random_state, contamination="auto").fit(train_values)
        robust_models[symbol] = robust
        isolation_forests[symbol] = forest
        fit_report[symbol] = {
            "fit_split": "train",
            "fit_windows": int(len(train_values)),
            "first_timestamp": index.loc[train_mask, "timestamp"].min().isoformat(),
            "last_timestamp": index.loc[train_mask, "timestamp"].max().isoformat(),
            "zero_mad_features": list(robust.zero_mad_features),
            "zero_mad_policy": "Use scale 1.0 for constant train features; retain finite raw deviation",
            "isolation_forest_score_direction": "-score_samples; larger means more noteworthy",
            "random_state": random_state,
        }
    return BaselineBundle(feature_order, robust_models, isolation_forests, fit_report)


def score_baselines(bundle, windows, index):
    last_bar = _validate_windows(windows, index, bundle.feature_order)
    result = index.copy()
    result["robust_zscore"] = np.nan
    result["isolation_forest"] = np.nan
    for symbol, rows in result.groupby("symbol", sort=True):
        if symbol not in bundle.robust_models or symbol not in bundle.isolation_forests:
            raise DataContractError(f"Missing baseline model: {symbol}")
        positions = rows.index.to_numpy()
        values = last_bar[positions]
        result.loc[positions, "robust_zscore"] = bundle.robust_models[symbol].score_samples(values)
        result.loc[positions, "isolation_forest"] = -bundle.isolation_forests[symbol].score_samples(values)
    if not np.isfinite(result[["robust_zscore", "isolation_forest"]].to_numpy()).all():
        raise DataContractError("Baseline scores must be finite")
    return result


def calibrate_thresholds(scores, *, quantile=0.98):
    if not 0 < quantile < 1:
        raise DataContractError("Threshold quantile must be between zero and one")
    required = {"symbol", "split", "robust_zscore", "isolation_forest"}
    if not required.issubset(scores.columns):
        raise DataContractError("Baseline score columns are missing")
    thresholds = {}
    for symbol in sorted(scores.symbol.unique()):
        validation = scores.loc[scores.symbol.eq(symbol) & scores.split.eq("validation")]
        if validation.empty:
            raise DataContractError(f"No validation scores for threshold: {symbol}")
        thresholds[symbol] = {
            model: {"threshold": float(validation[model].quantile(quantile)),
                    "validation_samples": int(len(validation)), "quantile": quantile}
            for model in ("robust_zscore", "isolation_forest")
        }
    return thresholds
