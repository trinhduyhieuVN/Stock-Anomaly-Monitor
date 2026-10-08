import pandas as pd
import pytest

from stock_monitor.dashboard import dashboard_slice
from stock_monitor.io import DataContractError


def test_dashboard_slice_selects_one_symbol_day_and_model():
    frame = pd.DataFrame([
        {"symbol": "FPT", "session_id": "2026-09-01", "timestamp": pd.Timestamp("2026-09-01T02:20Z"),
         "robust_zscore": 1.0, "robust_zscore_threshold": 2.0, "robust_zscore_alert": False},
        {"symbol": "FPT", "session_id": "2026-09-01", "timestamp": pd.Timestamp("2026-09-01T02:25Z"),
         "robust_zscore": 3.0, "robust_zscore_threshold": 2.0, "robust_zscore_alert": True},
        {"symbol": "HPG", "session_id": "2026-09-01", "timestamp": pd.Timestamp("2026-09-01T02:20Z"),
         "robust_zscore": 4.0, "robust_zscore_threshold": 2.0, "robust_zscore_alert": True},
    ])
    selected = dashboard_slice(frame, "FPT", "2026-09-01", "robust_zscore")
    assert selected.symbol.eq("FPT").all()
    assert selected.alert.tolist() == [False, True]
    assert selected.score.tolist() == [1.0, 3.0]


def test_dashboard_slice_rejects_unknown_model_or_empty_selection():
    frame = pd.DataFrame({"symbol": ["FPT"], "session_id": ["2026-09-01"],
                          "timestamp": [pd.Timestamp("2026-09-01T02:20Z")]})
    with pytest.raises(DataContractError, match="không được hỗ trợ"):
        dashboard_slice(frame, "FPT", "2026-09-01", "unknown")
    with pytest.raises(DataContractError, match="Không có dữ liệu"):
        dashboard_slice(frame, "VCB", "2026-09-01", "robust_zscore")
