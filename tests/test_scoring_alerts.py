from io import BytesIO

import numpy as np
import pandas as pd
import pytest
import torch

from stock_monitor.alerts import apply_thresholds, group_consecutive_alerts
from stock_monitor.io import DataContractError
from stock_monitor.model import LSTMAutoencoder
from stock_monitor.scoring import calibrate_thresholds, score_reconstructions
from stock_monitor.training import reconstruct


FEATURES = ("log_return", "range_pct", "body_pct", "log_volume_ratio")


def index(rows=3):
    timestamps = pd.date_range("2026-01-05 09:25", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame({"window_id": range(rows), "symbol": ["AAA"] * rows,
                         "segment_id": ["morning"] * rows, "split": ["validation"] * rows,
                         "timestamp": timestamps, "window_start": timestamps - pd.Timedelta(minutes=25)})


def test_last_bar_mean_mse_contributions_and_zero_policy():
    windows = np.zeros((2, 6, 4), dtype=np.float32)
    reconstructed = windows.copy()
    reconstructed[0, -1] = [1, 2, 0, 1]
    scores = score_reconstructions(windows, reconstructed, index(2), FEATURES,
                                   expected_feature_order=FEATURES)
    assert scores.loc[0, "score"] == pytest.approx(1.5)
    assert scores.loc[0, "share_range_pct"] == pytest.approx(4 / 6)
    assert scores.loc[0, [f"share_{name}" for name in FEATURES]].sum() == pytest.approx(1)
    assert scores.loc[1, "zero_total_error"]
    assert scores.loc[1, [f"share_{name}" for name in FEATURES]].sum() == 0


def test_feature_order_mismatch_blocks_and_reload_scores_match():
    windows = np.random.default_rng(42).normal(size=(3, 6, 4)).astype(np.float32)
    model = LSTMAutoencoder()
    output = reconstruct(model, windows)
    expected = score_reconstructions(windows, output, index(), FEATURES, expected_feature_order=FEATURES)
    checkpoint = BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    loaded = LSTMAutoencoder()
    loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    actual = score_reconstructions(windows, reconstruct(loaded, windows), index(), FEATURES,
                                   expected_feature_order=FEATURES)
    np.testing.assert_array_equal(actual["score"], expected["score"])
    with pytest.raises(DataContractError, match="feature order"):
        score_reconstructions(windows, output, index(), tuple(reversed(FEATURES)),
                              expected_feature_order=FEATURES)


def test_validation_thresholds_are_per_symbol_and_warn_on_low_samples():
    scores = pd.DataFrame({"symbol": ["AAA"] * 4 + ["BBB"] * 3,
                           "split": ["validation"] * 3 + ["test"] + ["validation"] * 3,
                           "score": [1., 2., 3., 999., 10., 20., 30.]})
    thresholds = calibrate_thresholds(scores, quantile=.98, minimum_samples=4)
    assert thresholds["AAA"]["threshold"] == pytest.approx(np.quantile([1, 2, 3], .98))
    assert thresholds["BBB"]["threshold"] == pytest.approx(np.quantile([10, 20, 30], .98))
    assert thresholds["AAA"]["low_sample_warning"]


def test_alert_grouping_stops_at_normal_rows_segments_and_gaps():
    timestamps = pd.to_datetime(["2026-01-05 09:25Z", "2026-01-05 09:30Z", "2026-01-05 09:35Z",
                                 "2026-01-05 10:00Z", "2026-01-05 10:05Z"])
    scores = pd.DataFrame({"symbol": ["AAA"] * 5, "segment_id": ["morning"] * 3 + ["afternoon"] * 2,
                           "timestamp": timestamps, "window_start": timestamps - pd.Timedelta(minutes=25),
                           "score": [2., 3., 0., 4., 5.], "threshold": [1.] * 5})
    decorated = apply_thresholds(scores, {"AAA": {"threshold": 1.}})
    alerts = group_consecutive_alerts(decorated)
    assert alerts["bars"].tolist() == [2, 2]
    assert alerts["reason"].eq("lstm_score_above_validation_q98").all()
    assert alerts.iloc[0].start_timestamp == timestamps[0]
    assert alerts.iloc[1].start_timestamp == timestamps[3]
