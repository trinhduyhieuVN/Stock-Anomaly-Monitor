"""Deterministic last-bar reconstruction scores and validation thresholds."""

import numpy as np
import pandas as pd

from .io import DataContractError


def score_reconstructions(windows, reconstructed, index, feature_order, *, expected_feature_order=None):
    feature_order = tuple(feature_order)
    if expected_feature_order is not None and feature_order != tuple(expected_feature_order):
        raise DataContractError("Scoring feature order differs from the checkpoint contract")
    if windows.shape != reconstructed.shape or windows.ndim != 3:
        raise DataContractError("Original and reconstructed windows must have the same three-dimensional shape")
    if windows.shape[0] != len(index) or windows.shape[2] != len(feature_order):
        raise DataContractError("Window, index and feature-order shapes differ")
    if not np.isfinite(windows).all() or not np.isfinite(reconstructed).all():
        raise DataContractError("Scoring inputs must be finite")

    feature_errors = np.square(reconstructed[:, -1, :] - windows[:, -1, :]).astype(np.float64)
    total_errors = feature_errors.sum(axis=1)
    shares = np.divide(feature_errors, total_errors[:, None], out=np.zeros_like(feature_errors),
                       where=total_errors[:, None] != 0)
    result = index.reset_index(drop=True).copy()
    result["score"] = feature_errors.mean(axis=1)
    result["zero_total_error"] = total_errors == 0
    for position, name in enumerate(feature_order):
        result[f"error_{name}"] = feature_errors[:, position]
        result[f"share_{name}"] = shares[:, position]
    return result


def calibrate_thresholds(scores, *, quantile=0.98, minimum_samples=100):
    if not 0 < quantile < 1:
        raise DataContractError("Threshold quantile must be between zero and one")
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples < 1:
        raise DataContractError("Minimum calibration samples must be a positive integer")
    required = {"symbol", "split", "score"}
    if not required.issubset(scores.columns) or not np.isfinite(scores["score"]).all():
        raise DataContractError("Calibration scores must include finite symbol, split and score columns")
    thresholds = {}
    for symbol in sorted(scores["symbol"].unique()):
        values = scores.loc[scores.symbol.eq(symbol) & scores.split.eq("validation"), "score"].to_numpy(dtype=float)
        if len(values) == 0:
            raise DataContractError(f"No validation scores for threshold calibration: {symbol}")
        thresholds[symbol] = {"threshold": float(np.quantile(values, quantile)), "quantile": quantile,
                              "calibration_split": "validation", "calibration_samples": len(values),
                              "low_sample_warning": len(values) < minimum_samples,
                              "minimum_recommended_samples": minimum_samples}
    return thresholds
