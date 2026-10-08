"""UI service contract tests using explicit synthetic bars."""

import json
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from stock_monitor.io import DataContractError, resolve_project_reference, sha256
from stock_monitor.services import (checked_metadata, daily_overview, filtered_csv,
                                    filtered_scores, load_operator_snapshot, observation_window,
                                    operator_status_view)


def sample_frame():
    stamps = pd.date_range("2026-09-01T02:20Z", periods=8, freq="5min")
    return pd.DataFrame({"symbol": ["FPT"] * 8, "timestamp": stamps, "window_start": stamps - pd.Timedelta(minutes=25),
                         "session_id": ["2026-09-01"] * 8, "segment_id": ["morning"] * 8,
                         "open": [100.] * 8, "high": [102.] * 8, "low": [99.] * 8, "close": [101.] * 8,
                         "volume": [1000.] * 8, "quality_status": ["ok"] * 8, "split": ["test"] * 8,
                         "lstm": [np.nan] * 5 + [1., 3., 4.], "lstm_threshold": [2.] * 8,
                         "lstm_alert": [False] * 6 + [True, True], "reason": [""] * 6 + ["q98", "q98"],
                         "error_log_return": [0.] * 8, "share_log_return": [0.] * 8})


def test_export_filters_preserve_timestamp_quality_model_and_run():
    selected = filtered_scores(sample_frame(), "FPT", "2026-09-01", "lstm")
    exported = pd.read_csv(BytesIO(filtered_csv(selected, "lstm", "synthetic_test_run", ["log_return"])))
    assert exported.symbol.eq("FPT").all() and exported.model.eq("lstm").all()
    assert exported.run_id.eq("synthetic_test_run").all()
    assert exported.quality_status.eq("ok").all()
    assert exported.alert.sum() == 2
    assert exported.score.notna().sum() == 3
    assert pd.to_datetime(exported.timestamp, utc=True).equals(selected.timestamp)


def test_overview_counts_only_scored_points_and_groups_consecutive_alerts():
    summary = daily_overview(sample_frame(), "2026-09-01", "lstm")
    assert summary.iloc[0]["Số nến"] == 8
    assert summary.iloc[0]["Đã chấm"] == 3
    assert summary.iloc[0]["Điểm cảnh báo"] == 2
    assert summary.iloc[0]["Đoạn cảnh báo"] == 1


def test_observation_window_shows_six_past_bars_and_rejects_gaps():
    features = sample_frame()
    selected = filtered_scores(features, "FPT", "2026-09-01", "lstm").iloc[6]
    window = observation_window(features, selected)
    assert len(window) == 6
    assert window.timestamp.max() == selected.timestamp
    assert window.timestamp.min() == selected.window_start
    with pytest.raises(DataContractError, match="sáu nến"):
        observation_window(features.drop(index=3), selected)


def test_missing_lstm_and_unscored_window_have_clear_errors():
    with pytest.raises(DataContractError, match="Chưa có LSTM"):
        filtered_scores(sample_frame().drop(columns="lstm"), "FPT", "2026-09-01", "lstm")
    selected = filtered_scores(sample_frame(), "FPT", "2026-09-01", "lstm").iloc[0]
    with pytest.raises(DataContractError, match="chưa đủ cửa sổ"):
        observation_window(sample_frame(), selected)


def test_checked_metadata_rejects_tampering_and_missing_required_files(tmp_path):
    directory = tmp_path / "synthetic_bundle"
    directory.mkdir()
    artifact = directory / "scores.parquet"
    artifact.write_bytes(b"explicit-checksum-test-fixture")
    metadata = {"kind": "test_bundle", "run_id": directory.name,
                "artifact_sha256": {"scores.parquet": sha256(artifact)}}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert checked_metadata(directory, "test_bundle", ["scores.parquet"])["kind"] == "test_bundle"
    with pytest.raises(DataContractError, match="thiếu file"):
        checked_metadata(directory, "test_bundle", ["best_checkpoint.pt"])
    artifact.write_bytes(b"tampered")
    with pytest.raises(DataContractError, match="checksum"):
        checked_metadata(directory, "test_bundle", ["scores.parquet"])


def test_relocated_project_reference_prefers_clean_copy(tmp_path):
    original = tmp_path / "original" / "artifacts" / "features" / "features_run"
    relocated_root = tmp_path / "clean_copy"
    relocated = relocated_root / "artifacts" / "features" / "features_run"
    original.mkdir(parents=True)
    relocated.mkdir(parents=True)
    assert resolve_project_reference(original, relocated_root) == relocated.resolve()


def test_operator_status_view_keeps_offline_timing_separate_from_live():
    view = operator_status_view({
        "state": "BLOCKED_UNKNOWN_CALENDAR", "live_evidence": False,
        "source_rights_status": "pending", "latest_closed_bar_utc": None,
        "gap_count": 2, "stale": True, "received_at_utc": None, "score_duration_ms": None,
        "offline_audit": {"received_at_utc": "2026-10-01T07:31:00Z", "score_duration_ms": 12.5},
    })
    assert view["mode_label"] == "OFFLINE — KHÔNG PHẢI LIVE"
    assert view["latest_closed_bar"] == "Chưa đo trong phiên thật"
    assert view["receive_time"] == view["score_duration"] == "Chưa đo trong phiên thật"
    assert view["source_rights_status"] == "pending" and view["gap_count"] == 2 and view["stale"] is True
    assert "12.500 ms" in view["offline_audit"] and "historical prefix" in view["offline_audit"]


def test_operator_snapshot_loader_and_unknown_state_fail_closed(tmp_path):
    snapshot = tmp_path / "status.json"
    snapshot.write_text(json.dumps({"schema_version": 1, "kind": "operator_status_snapshot",
        "state": "UNRECOGNIZED", "live_evidence": False, "source_rights_status": "pending"}), encoding="utf-8")
    view = operator_status_view(load_operator_snapshot(tmp_path, snapshot))
    assert view["severity"] == "warning" and "Bị chặn" in view["state_label"]
    with pytest.raises(DataContractError, match="inside the project"):
        load_operator_snapshot(tmp_path, tmp_path.parent / "outside.json")
