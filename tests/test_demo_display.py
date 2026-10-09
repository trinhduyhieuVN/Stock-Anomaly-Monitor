"""Presentation regressions reported by the owner: candles, initial view, nulls."""
import base64
from io import BytesIO
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stock_monitor.upgrade.ui_service import load_ui_bundle

ROOT = Path(__file__).resolve().parents[1]


def start(monkeypatch):
    monkeypatch.setenv("STOCK_MONITOR_UPGRADE_JOURNAL", ".local/demo_display/" + uuid.uuid4().hex + ".sqlite")
    monkeypatch.delenv("STOCK_MONITOR_UPGRADE_CONFIG", raising=False)
    return AppTest.from_file(str(ROOT / "app_upgrade.py")).run(timeout=40)


def candles(at):
    assert not at.exception
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert spec["layout"]["xaxis"]["rangeslider"]["visible"] is False
    return [trace for trace in spec["data"] if trace["type"] == "candlestick"]


def values(array):
    return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"]) if isinstance(array, dict) else array


@pytest.mark.parametrize("code", ["FPT", "HPG", "VCB"])
def test_default_full_session_has_exact_ohlc_and_useful_event(monkeypatch, code):
    at = start(monkeypatch)
    if code != "FPT":
        at.sidebar.selectbox(key="v2_symbol").select(code).run(timeout=40)
    assert at.sidebar.radio[0].value == "Rà soát hiện tại"
    traces = candles(at)
    assert sum(len(trace["x"]) for trace in traces) == 48
    bundle = load_ui_bundle(ROOT)
    rows = sorted([row for row in bundle["rows"] if row["symbol"] == code and row["session_id"] == "2025-02-03"], key=lambda r: r["timestamp"])
    for key in ["open", "high", "low", "close"]:
        actual = np.concatenate([values(trace[key]) for trace in traces])
        np.testing.assert_allclose(actual, [row[key] for row in rows], rtol=0, atol=0)
    assert any(kind in at.selectbox(key="v2_event").value for kind in [" · point ·", " · collective ·"])
    assert not any("None" in item.value for item in at.markdown)


def test_first_bar_explains_unavailability_without_filling_future_or_zero(monkeypatch):
    at = start(monkeypatch)
    at.sidebar.radio[0].set_value("Replay tại T").run(timeout=40)
    assert sum(len(t["x"]) for t in candles(at)) == 1
    assert any("nến đầu tiên" in item.value for item in at.info)
    assert any("Point rank: chưa đủ dữ liệu" in item.value for item in at.markdown)
    assert any("Chưa có ứng viên" in item.value for item in at.info)
    assert any("Đang chờ nến đích" in item.value for item in at.info)
    assert not any("None" in item.value for item in at.markdown)
    frame = pd.read_csv(BytesIO(at.session_state["v2_export_csv"]))
    assert len(frame) == 1 and frame.score.isna().all() and set(frame.kind) == {"insufficient_data"}
    forecast = next(df.value for df in at.dataframe if "horizon_bars" in df.value)
    assert forecast.residual.isna().all() and forecast.realized_return.isna().all()
    # Streamlit renders Arrow nulls using its own placeholder, not Styler's
    # na_rep. Check the actual emitted frontend proto AND preserve raw nulls.
    assert all(df.proto.placeholder == "—" for df in at.dataframe)
    next(b for b in at.button if b.label == "Xem toàn phiên").click().run(timeout=40)
    assert at.sidebar.radio[0].value == "Rà soát hiện tại"
    assert sum(len(t["x"]) for t in candles(at)) == 48


def test_candidate_schemas_are_not_merged_into_sparse_union(monkeypatch):
    at = start(monkeypatch)
    picker = at.selectbox(key="v2_event")
    picker.select(next(label for label in picker.options if label.startswith("10:35:00 ·"))).run(timeout=40)
    candidates = [df.value for df in at.dataframe if "admissible" in df.value]
    assert len(candidates) == 2
    assert set(candidates[0].columns) == {"raw_score", "admissible"}
    assert "length" in candidates[1] and candidates[1].length.notna().all()
    assert any("Chưa có rà soát" in item.value for item in at.caption)
