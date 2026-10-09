from pathlib import Path
import uuid

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app(monkeypatch, mode="Rà soát hiện tại"):
    monkeypatch.setenv("STOCK_MONITOR_UPGRADE_JOURNAL", ".local/upgrade_ui_tests/" + uuid.uuid4().hex + ".sqlite")
    monkeypatch.delenv("STOCK_MONITOR_UPGRADE_CONFIG", raising=False)
    at = AppTest.from_file(str(ROOT / "app_upgrade.py"))
    at.session_state["v2_mode"] = mode
    return at.run(timeout=30)


def test_app_replay_step_reset_filters_and_prefix_export(monkeypatch):
    at = app(monkeypatch, mode="Replay tại T")
    assert not at.exception and at.session_state["v2_cursor"] == 0
    assert any("SYNTHETIC" in item.value and "không phải LIVE" in item.value for item in at.caption)
    at.sidebar.button[0].click().run(timeout=30)
    assert not at.exception and at.session_state["v2_cursor"] == 1
    at.sidebar.button[1].click().run(timeout=30)
    assert at.session_state["v2_cursor"] == 0
    at.sidebar.select_slider[0].set_value(18).run(timeout=30)
    at.sidebar.multiselect[0].set_value(["point"]).run(timeout=30)
    at.sidebar.multiselect[1].set_value(["initial"]).run(timeout=30)
    assert not at.exception
    csv = at.session_state["v2_export_csv"]
    frame = pd.read_csv(__import__('io').BytesIO(csv))
    assert len(frame) == 1 and frame.kind.iloc[0] == "point" and frame.version.iloc[0] == 1
    assert "post_evidence" not in frame.columns
    at.sidebar.select_slider[0].set_value(21).run(timeout=30)
    at.sidebar.multiselect[1].set_value(["post_event"]).run(timeout=30)
    assert not at.exception
    frame = pd.read_csv(__import__('io').BytesIO(at.session_state["v2_export_csv"]))
    assert frame.version.iloc[0] == 3 and frame.stage.iloc[0] == "post_event"
    at.sidebar.selectbox[0].select("HPG").run(timeout=30)
    assert not at.exception and at.session_state["v2_cursor"] == 0


def test_manual_review_persists_current_only_and_filters(monkeypatch):
    at = app(monkeypatch)
    at.sidebar.radio[0].set_value("Rà soát hiện tại").run(timeout=30)
    assert not at.exception
    at.selectbox(key="v2_event").select_index(18).run(timeout=30)
    at.text_input(key="v2_reviewer").set_value("AppTest control")
    at.selectbox(key="v2_reviewer_kind").select("ai_assisted")
    at.text_area(key="v2_reason").set_value("Synthetic persistence test only, not expert/research truth")
    next(button for button in at.button if button.label == "Lưu review mới").click().run(timeout=30)
    assert not at.exception and any("Đã lưu" in item.value for item in at.success)
    at.sidebar.selectbox(key="v2_review_filter").select("reviewed").run(timeout=30)
    assert not at.exception
    frame = pd.read_csv(__import__('io').BytesIO(at.session_state["v2_export_csv"]))
    assert len(frame) == 1
    at.sidebar.radio[0].set_value("Replay tại T").run(timeout=30)
    assert not at.exception and len(pd.read_csv(__import__('io').BytesIO(at.session_state["v2_export_csv"]))) == 0
    assert any("Bộ lọc chưa" in item.value for item in at.info)


def test_missing_artifact_has_recovery_not_live_or_fake_model(monkeypatch):
    monkeypatch.setenv("STOCK_MONITOR_UPGRADE_CONFIG", "configs/nonexistent_upgrade_ui.json")
    at = AppTest.from_file(str(ROOT / "app_upgrade.py")).run(timeout=30)
    assert not at.exception and any("Không tải" in item.value for item in at.error)
    assert any("không upload" in item.value for item in at.info)


def test_empty_kind_filter_teaches_recovery(monkeypatch):
    at = app(monkeypatch)
    at.sidebar.multiselect[0].set_value([]).run(timeout=30)
    assert not at.exception
    assert any("Chọn thêm" in item.value for item in at.info)
