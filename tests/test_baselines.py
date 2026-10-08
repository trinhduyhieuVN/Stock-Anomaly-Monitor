from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest

from stock_monitor.baselines import calibrate_thresholds, fit_baselines, fit_robust_zscore, score_baselines
from stock_monitor.io import DataContractError


def baseline_data():
    records, windows = [], []
    window_id = 0
    for symbol_offset, symbol in enumerate(["FPT", "HPG"]):
        for split, count, shift in [("train", 20, 0.0), ("validation", 10, 0.2), ("test", 8, 0.4)]:
            for sample in range(count):
                last = np.array([sample % 5, 1.0, symbol_offset + sample / 10, shift], dtype=np.float32)
                windows.append(np.stack([last - 0.1, last]))
                records.append({"window_id": window_id, "symbol": symbol, "split": split,
                                "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=5 * window_id)})
                window_id += 1
    return np.stack(windows), pd.DataFrame(records), ("a", "constant", "c", "d")


def test_robust_zscore_records_zero_mad_and_stays_finite():
    model = fit_robust_zscore([[0, 1], [1, 1], [2, 1]], ["moving", "constant"])
    assert model.zero_mad_features == ("constant",)
    assert model.scale[1] == 1.0
    assert np.isfinite(model.score_samples([[3, 4]])).all()


def test_fit_score_and_calibration_use_contract_splits():
    windows, index, feature_order = baseline_data()
    bundle = fit_baselines(windows, index, feature_order)
    scores = score_baselines(bundle, windows, index)
    thresholds = calibrate_thresholds(scores)
    assert scores[["window_id", "symbol", "split", "timestamp"]].equals(index)
    for symbol in ["FPT", "HPG"]:
        assert bundle.fit_report[symbol]["fit_windows"] == 20
        assert bundle.fit_report[symbol]["fit_split"] == "train"
        expected = scores.loc[scores.symbol.eq(symbol) & scores.split.eq("validation"), "robust_zscore"].quantile(.98)
        assert thresholds[symbol]["robust_zscore"] == {"threshold": expected, "validation_samples": 10, "quantile": .98}
        train = index.symbol.eq(symbol) & index.split.eq("train")
        direct = -bundle.isolation_forests[symbol].score_samples(windows[train.to_numpy(), -1, :])
        np.testing.assert_allclose(scores.loc[train, "isolation_forest"], direct)


def test_reload_preserves_baseline_scores():
    windows, index, feature_order = baseline_data()
    bundle = fit_baselines(windows, index, feature_order)
    persisted = BytesIO()
    joblib.dump(bundle, persisted)
    persisted.seek(0)
    expected = score_baselines(bundle, windows, index)
    actual = score_baselines(joblib.load(persisted), windows, index)
    pd.testing.assert_frame_equal(actual, expected)


def test_validation_is_not_allowed_to_replace_train_fit():
    windows, index, feature_order = baseline_data()
    train = index.split.eq("train")
    index.loc[train, "split"] = "validation"
    with pytest.raises(DataContractError, match="No train windows"):
        fit_baselines(windows, index, feature_order)
