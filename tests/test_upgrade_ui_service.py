from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from stock_monitor.io import DataContractError
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.journal import Journal
from stock_monitor.upgrade.ui_service import load_ui_bundle, replay_session, filtered_events, event_csv, forecast_view

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_ui_bundle(ROOT)


def snapshot(bundle, tmp_path, clock, session="2025-02-03"):
    return replay_session(bundle, Journal(tmp_path / "j.sqlite"), symbol="FPT", session_id=session, as_of=clock)


def test_pending_to_mature_initial_preserved_reset_and_csv(bundle, tmp_path):
    first = snapshot(bundle, tmp_path, "2025-02-03T03:35:02Z")
    event = next(row for row in first["events"] if utc_time(row["event_end"]) == utc_time("2025-02-03T03:35:00Z"))
    assert event["kind"] == "point" and event["stage"] == "initial"
    assert all(row["status"] == "pending" for row in forecast_view(first, event, bundle))
    after = snapshot(bundle, tmp_path, "2025-02-03T03:50:02Z")
    versions = [row for row in after["versions"] if row["event_id"] == event["event_id"]]
    assert versions[0] == event and len(versions) == 3
    assert all(row["kind"] == event["kind"] and row["score"] == event["score"] for row in versions)
    assert all(row["status"] == "mature" for row in forecast_view(after, event, bundle))
    reset = snapshot(bundle, tmp_path, "2025-02-03T03:35:02Z")
    assert reset["events"] == first["events"] and reset["versions"] == first["versions"]
    initial_only = filtered_events(after, kinds=["point"], stages=["initial"])
    assert event in initial_only and all(row["version"] == 1 and "post_evidence" not in row for row in initial_only)
    csv = event_csv(filtered_events(reset, kinds=["point"]))
    assert b"post_evidence" not in csv
    assert pd.read_csv(__import__('io').BytesIO(csv)).shape[0] == 1


def test_future_data_and_news_cannot_change_prefix(bundle, tmp_path):
    clock = "2025-02-10T03:35:02Z"
    before = snapshot(bundle, tmp_path / "a", clock, "2025-02-10")
    altered = deepcopy(bundle)
    for row in altered["rows"]:
        if utc_time(row["available_at"]) > utc_time(clock):
            row["close"] *= 1.001
            row["high"] = max(row["high"], row["close"] * 1.001)
    after = snapshot(altered, tmp_path / "b", clock, "2025-02-10")
    assert before["events"] == after["events"] and before["news"] == []
    later = snapshot(bundle, tmp_path / "a", "2025-02-10T03:45:02Z", "2025-02-10")
    assert len(later["news"]) == 1


def test_collective_reference_diagnostics_and_gap_boundaries(bundle, tmp_path):
    seen = snapshot(bundle, tmp_path / "shift", "2025-02-06T04:00:02Z", "2025-02-06")
    segments = filtered_events(seen, kinds=["collective"], stages=["initial"])
    assert segments and all(utc_time(row["event_end"]) <= utc_time(seen["as_of"]) for row in segments)
    assert seen["details"][segments[-1]["event_id"]]["joint_reference_count"] >= 100
    gap = snapshot(bundle, tmp_path / "gap", "2025-02-11T04:00:02Z", "2025-02-11")
    assert all(row["kind"] != "collective" or row["segment_id"] for row in gap["events"])
    assert len(gap["bars"].segment_id.unique()) == 2


def test_current_review_never_leaks_into_historical_T(bundle, tmp_path):
    journal = Journal(tmp_path / "j.sqlite")
    clock = "2025-02-03T03:35:02Z"
    replay = replay_session(bundle, journal, symbol="FPT", session_id="2025-02-03", as_of=clock)
    event = replay["events"][-1]
    now = datetime.now(timezone.utc).isoformat()
    review = {"schema_version": 2, "review_id": "service-control", "event_id": event["event_id"], "event_version": event["version"],
              "reviewer": "test-control", "reviewer_kind": "ai_assisted", "label": "uncertain", "reason": "Not expert or research evidence",
              "reviewed_at": now, "available_at": now, "reference_url": ""}
    journal.append("reviews", review)
    replay_again = replay_session(bundle, journal, symbol="FPT", session_id="2025-02-03", as_of=clock)
    assert replay_again["reviews"] == [] and filtered_events(replay_again, review_status="reviewed") == []
    assert journal.query("reviews", as_of=now) == [review]


def test_pinned_resource_corruption_and_path_escape(tmp_path):
    shutil.copytree(ROOT / "demo/upgrade_ui", tmp_path / "demo/upgrade_ui")
    (tmp_path / "configs").mkdir()
    shutil.copy2(ROOT / "configs/upgrade_ui.json", tmp_path / "configs/upgrade_ui.json")
    assert load_ui_bundle(tmp_path)["manifest"]["contains_real_data"] is False
    calibration = tmp_path / "demo/upgrade_ui/segment_calibration.json"
    calibration.write_bytes(calibration.read_bytes() + b" ")
    with pytest.raises(DataContractError, match="artifact mismatch"):
        load_ui_bundle(tmp_path)
    config = json.loads((tmp_path / "configs/upgrade_ui.json").read_text())
    config["manifest"] = "../escape.json"
    (tmp_path / "configs/upgrade_ui.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(DataContractError, match="escapes"):
        load_ui_bundle(tmp_path)


def test_empty_and_invalid_session_filters(bundle, tmp_path):
    empty = snapshot(bundle, tmp_path, "2025-02-03T02:00:00Z")
    assert empty["events"] == [] and empty["bars"].empty
    assert event_csv([]).decode("utf-8-sig").startswith("event_id,")
    with pytest.raises(DataContractError):
        snapshot(bundle, tmp_path, "2025-01-06T02:05:02Z", "2025-01-06")
