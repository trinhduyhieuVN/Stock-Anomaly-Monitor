"""Owner's six demo tasks through the actual Streamlit entrypoint."""
from io import BytesIO
from pathlib import Path
import uuid

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def demo(monkeypatch, day="2025-02-03", code="FPT"):
    monkeypatch.setenv("STOCK_MONITOR_UPGRADE_JOURNAL", ".local/demo_scenarios/" + uuid.uuid4().hex + ".sqlite")
    monkeypatch.delenv("STOCK_MONITOR_UPGRADE_CONFIG", raising=False)
    at = AppTest.from_file(str(ROOT / "app_upgrade.py"))
    at.session_state["v2_mode"] = "Replay tại T"
    at.run(timeout=40)
    if code != "FPT": at.sidebar.selectbox(key="v2_symbol").select(code).run(timeout=40)
    if day != "2025-02-03": at.sidebar.selectbox(key="v2_session").select(day).run(timeout=40)
    assert not at.exception
    return at


def exported(at):
    assert not at.exception
    return pd.read_csv(BytesIO(at.session_state["v2_export_csv"]))


def pick_clock(at, clock):
    select = at.selectbox(key="v2_event")
    select.select(next(label for label in select.options if label.startswith(clock + " ·"))).run(timeout=40)
    assert not at.exception


def forecasts(at):
    return next(item.value for item in at.dataframe if "horizon_bars" in item.value.columns)


@pytest.mark.parametrize("code", ["FPT", "HPG", "VCB"])
def test_isolated_point_initial_and_prefix_csv(monkeypatch, code):
    at = demo(monkeypatch, code=code)
    at.sidebar.select_slider[0].set_value(18)
    at.sidebar.multiselect(key="v2_kinds").set_value(["point"])
    at.sidebar.multiselect(key="v2_stages").set_value(["initial"])
    at.run(timeout=40)
    frame = exported(at)
    assert len(frame) == 1 and frame.kind.iloc[0] == "point" and frame.version.iloc[0] == 1
    assert pd.Timestamp(frame.available_at.iloc[0]) <= pd.Timestamp("2025-02-03T03:35:02Z")
    assert set(forecasts(at).status) == {"pending"} and "post_evidence" not in frame


def test_mild_collective_has_estimated_visible_bounds(monkeypatch):
    at = demo(monkeypatch, "2025-02-06")
    at.sidebar.select_slider[0].set_value(23)
    at.sidebar.multiselect(key="v2_kinds").set_value(["collective"])
    at.sidebar.multiselect(key="v2_stages").set_value(["initial"])
    at.run(timeout=40)
    frame = exported(at)
    assert len(frame) > 0 and set(frame.kind) == {"collective"}
    assert pd.to_datetime(frame.event_end).max() <= pd.Timestamp("2025-02-06T04:00:02Z")
    assert any("ranh giới ước lượng tại T" in item.value for item in at.caption)


def test_known_later_news_not_shown_early(monkeypatch):
    at = demo(monkeypatch, "2025-02-10")
    at.sidebar.select_slider[0].set_value(18).run(timeout=40)
    assert any("Chưa có tin" in item.value for item in at.markdown)
    assert not any("news_id" in item.value.columns for item in at.dataframe)
    at.sidebar.select_slider[0].set_value(20).run(timeout=40)
    news = next(item.value for item in at.dataframe if "news_id" in item.value.columns)
    assert len(news) == 1 and news.symbol.iloc[0] == "FPT"
    assert pd.Timestamp(news.available_at.iloc[0]) <= pd.Timestamp("2025-02-10T03:45:02Z")
    at.sidebar.select_slider[0].set_value(18).run(timeout=40)
    assert not any("news_id" in item.value.columns for item in at.dataframe)


def test_delayed_outcome_matures_without_mutating_initial(monkeypatch):
    at = demo(monkeypatch, "2025-02-13")
    at.sidebar.select_slider[0].set_value(18).run(timeout=40)
    pick_clock(at, "10:35:00")
    assert set(forecasts(at).status) == {"pending"}
    initial = exported(at).loc[lambda f: pd.to_datetime(f.event_end).eq(pd.Timestamp("2025-02-13T03:35:00Z"))].iloc[0]
    at.sidebar.select_slider[0].set_value(24).run(timeout=40)
    pick_clock(at, "10:35:00")
    assert set(forecasts(at).status) == {"mature"}
    mature = exported(at).loc[lambda f: f.event_id.eq(initial.event_id)].iloc[0]
    assert mature.version > initial.version and mature.kind == initial.kind and mature.score == initial.score
    at.sidebar.select_slider[0].set_value(18).run(timeout=40)
    pick_clock(at, "10:35:00")
    again = exported(at).loc[lambda f: f.event_id.eq(initial.event_id)].iloc[0]
    assert again.version == initial.version and set(forecasts(at).status) == {"pending"}


def test_end_session_censored_not_zero_residual_then_reset(monkeypatch):
    at = demo(monkeypatch, "2025-02-14")
    at.sidebar.select_slider[0].set_value(47).run(timeout=40)
    pick_clock(at, "14:30:00")
    frame = forecasts(at)
    assert set(frame.status) == {"censored_boundary"}
    assert frame.residual.isna().all() and frame.realized_return.isna().all()
    at.sidebar.button[1].click().run(timeout=40)
    assert at.session_state["v2_cursor"] == 0 and len(exported(at)) == 1
    assert exported(at).version.iloc[0] == 1


def test_review_validation_save_filter_export_and_no_backdate(monkeypatch):
    at = demo(monkeypatch)
    at.sidebar.radio[0].set_value("Rà soát hiện tại").run(timeout=40)
    pick_clock(at, "10:35:00")
    next(b for b in at.button if b.label == "Lưu review mới").click().run(timeout=40)
    assert not at.exception and any("Review chưa lưu" in item.value for item in at.error)
    at.text_input(key="v2_reviewer").set_value("Demo control")
    at.text_area(key="v2_reason").set_value("Synthetic QA only, not a research or expert label")
    next(b for b in at.button if b.label == "Lưu review mới").click().run(timeout=40)
    assert not at.exception and any("Đã lưu" in item.value for item in at.success)
    at.sidebar.selectbox(key="v2_review_filter").select("reviewed").run(timeout=40)
    assert len(exported(at)) == 1
    at.sidebar.radio[0].set_value("Replay tại T").run(timeout=40)
    assert len(exported(at)) == 0
    at.sidebar.selectbox(key="v2_review_filter").select("all").run(timeout=40)
    assert len(exported(at)) == 1
