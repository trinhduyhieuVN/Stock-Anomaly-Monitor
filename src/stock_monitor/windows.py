"""Common-date splits, train-only scaling and contiguous six-bar windows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .features import FEATURE_ORDER
from .io import DataContractError

SPLITS = ("train", "validation", "test")
SCALED_COLUMNS = tuple("scaled_" + name for name in FEATURE_ORDER)


def split_by_common_dates(frame, symbols, fractions=(0.6, 0.2, 0.2)):
    if len(fractions) != 3 or any(not np.isfinite(v) or v <= 0 for v in fractions) or not np.isclose(sum(fractions), 1):
        raise DataContractError("Split fractions must be three positive values summing to one")
    date_sets = [set(frame.loc[frame.symbol.eq(symbol) & frame.feature_eligible, "session_id"]) for symbol in symbols]
    dates = sorted(set.intersection(*date_sets)) if date_sets else []
    n_train, n_validation = int(len(dates) * fractions[0]), int(len(dates) * fractions[1])
    if min(n_train, n_validation, len(dates) - n_train - n_validation) < 1:
        raise DataContractError("Too few common eligible dates for train/validation/test")
    mapping = {"train": dates[:n_train], "validation": dates[n_train:n_train + n_validation],
               "test": dates[n_train + n_validation:]}
    result = frame.copy()
    result["split"] = ""
    for name, days in mapping.items():
        result.loc[result.session_id.isin(days), "split"] = name
    report = {"dates": mapping, "fractions": list(fractions), "rounding": "floor train and validation; remainder test",
        "common_eligible_date_count": len(dates), "excluded_noncommon_or_warmup_dates":
            sorted(set(frame.session_id) - set(dates)), "by_symbol": {}}
    for symbol in symbols:
        report["by_symbol"][symbol] = {name: {
            "eligible_bars": int((result.symbol.eq(symbol) & result.feature_eligible & result.split.eq(name)).sum()),
            "dates": len(days)} for name, days in mapping.items()}
    return result, report


def fit_scalers(frame, symbols):
    scalers, report = {}, {}
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise DataContractError("Scaler input must contain unique bars, not replicated window rows")
    for symbol in symbols:
        rows = frame.loc[frame.symbol.eq(symbol) & frame.feature_eligible & frame.split.eq("train")]
        values = rows[list(FEATURE_ORDER)].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise DataContractError(f"No finite train rows for scaler: {symbol}")
        scaler = RobustScaler().fit(values)
        scalers[symbol] = scaler
        raw_iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
        report[symbol] = {"fit_rows": len(rows), "fit_dates": sorted(rows.session_id.unique()),
            "first_timestamp": rows.timestamp.min().isoformat(), "last_timestamp": rows.timestamp.max().isoformat(),
            "center": scaler.center_.tolist(), "scale": scaler.scale_.tolist(), "raw_iqr": raw_iqr.tolist(),
            "zero_iqr_features": [name for name, value in zip(FEATURE_ORDER, raw_iqr) if value == 0],
            "zero_iqr_policy": "RobustScaler uses unit scale; no arbitrary clipping",
            "fit_split": "train", "fit_on": "unique eligible bars before window creation"}
    return scalers, report


def transform_features(frame, scalers):
    result = frame.copy()
    for col in SCALED_COLUMNS:
        result[col] = np.nan
    for symbol, rows in result.loc[result.feature_eligible].groupby("symbol"):
        if symbol not in scalers:
            raise DataContractError(f"Missing scaler: {symbol}")
        scaler = scalers[symbol]
        if scaler.n_features_in_ != len(FEATURE_ORDER):
            raise DataContractError("Scaler feature count differs from the feature contract")
        values = scaler.transform(rows[list(FEATURE_ORDER)].to_numpy(dtype=float))
        if not np.isfinite(values).all():
            raise DataContractError("Scaled features must be finite")
        result.loc[rows.index, list(SCALED_COLUMNS)] = values
    return result


@dataclass
class WindowResult:
    values: np.ndarray
    row_indices: np.ndarray
    index: pd.DataFrame
    report: dict


def make_windows(frame, *, window_length=6):
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 1:
        raise DataContractError("Window length must be a positive integer")
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise DataContractError("Windows require unique input bars")
    # The persisted feature-row order is the stable index used by the NPZ.
    data = frame.reset_index(drop=True)
    eligible = data.loc[data.feature_eligible & data.split.isin(SPLITS)]
    windows, pointers, records = [], [], []
    for (symbol, day, segment, split), rows in eligible.groupby(
            ["symbol", "session_id", "segment_id", "split"], sort=True):
        rows = rows.sort_values("timestamp")
        for stop in range(window_length, len(rows) + 1):
            chunk = rows.iloc[stop - window_length:stop]
            if not chunk.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all():
                continue
            values = chunk[list(SCALED_COLUMNS)].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise DataContractError("Nonfinite window values")
            windows.append(values)
            pointers.append(chunk.index.to_numpy(dtype=np.int64))
            records.append({"window_id": len(records), "symbol": symbol, "session_id": day,
                "segment_id": segment, "split": split, "timestamp": chunk.timestamp.iloc[-1],
                "window_start": chunk.timestamp.iloc[0], "start_row": int(chunk.index[0]), "end_row": int(chunk.index[-1])})
    columns = ["window_id", "symbol", "session_id", "segment_id", "split", "timestamp", "window_start", "start_row", "end_row"]
    index = pd.DataFrame(records, columns=columns)
    if windows:
        values, row_indices = np.stack(windows), np.stack(pointers)
    else:
        values = np.empty((0, window_length, len(FEATURE_ORDER)), dtype=np.float32)
        row_indices = np.empty((0, window_length), dtype=np.int64)
    counts = {symbol: {split: int((index.symbol.eq(symbol) & index.split.eq(split)).sum())
                       for split in SPLITS} for symbol in sorted(data.symbol.unique())}
    used = set(row_indices.ravel().tolist())
    return WindowResult(values, row_indices, index, {"window_length": window_length,
        "feature_order": list(FEATURE_ORDER), "shape": list(values.shape), "by_symbol": counts,
        "eligible_split_rows": len(eligible), "eligible_rows_unused_by_windows": len(set(eligible.index) - used),
        "no_crossing": ["symbol", "session_id", "segment_id", "split", "5m gap"]})
