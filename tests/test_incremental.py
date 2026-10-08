import json
from pathlib import Path

import pandas as pd
import pytest

import stock_monitor.incremental as incremental
from stock_monitor.incremental import load_incremental_current, merge_incremental_minutes, persist_incremental_poll
from stock_monitor.io import DataContractError


COLUMNS = ["symbol", "timestamp", "retrieved_at_utc", "source", "open", "high", "low", "close", "volume", "accumulated_volume"]


def minute(timestamp, retrieved, *, close=100.0, volume=10, accumulated=10):
    return {
        "symbol": "FPT", "timestamp": timestamp, "retrieved_at_utc": retrieved,
        "source": "VCI", "open": 100.0, "high": max(101.0, close),
        "low": min(99.0, close), "close": close, "volume": volume,
        "accumulated_volume": accumulated,
    }


def frame(*rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_incremental_merge_appends_audits_revision_and_is_idempotent():
    existing = frame(
        minute("2026-10-05T02:15:00Z", "2026-10-05T02:16:10Z"),
        minute("2026-10-05T02:16:00Z", "2026-10-05T02:17:10Z", accumulated=20),
    )
    incoming = frame(
        minute("2026-10-05T02:15:00Z", "2026-10-05T02:16:10Z"),
        minute("2026-10-05T02:16:00Z", "2026-10-05T02:18:10Z", close=101.5, accumulated=20),
        minute("2026-10-05T02:17:00Z", "2026-10-05T02:18:10Z", close=102.0, accumulated=30),
    )

    first = merge_incremental_minutes(existing, incoming)
    assert first.report == {
        "existing_rows": 2, "incoming_rows": 3,
        "existing_exact_duplicates_removed": 0, "incoming_exact_duplicates_removed": 0,
        "appended_rows": 1, "refreshed_rows": 0, "revisions_applied": 1,
        "stale_observations_ignored": 0, "same_observations_ignored": 1,
        "output_rows": 3, "state_changed": True,
    }
    assert first.minutes[["symbol", "timestamp"]].duplicated().sum() == 0
    assert first.minutes.loc[first.minutes.timestamp.eq(pd.Timestamp("2026-10-05T02:16:00Z")), "close"].item() == 101.5
    assert first.revisions.changed_fields.tolist() == ["high|close"]
    assert first.revisions.previous_high.item() == 101.0
    assert first.revisions.incoming_high.item() == 101.5
    assert first.revisions.previous_close.item() == 100.0
    assert first.revisions.incoming_close.item() == 101.5

    repeated = merge_incremental_minutes(first.minutes, incoming)
    pd.testing.assert_frame_equal(repeated.minutes, first.minutes)
    assert repeated.revisions.empty
    assert repeated.report["same_observations_ignored"] == 3
    assert repeated.report["state_changed"] is False


def test_incremental_merge_never_rolls_back_to_stale_observation():
    existing = frame(minute("2026-10-05T02:15:00Z", "2026-10-05T02:20:00Z", close=102.0))
    stale = frame(minute("2026-10-05T02:15:00Z", "2026-10-05T02:19:00Z", close=98.0))
    result = merge_incremental_minutes(existing, stale)
    assert result.minutes.close.item() == 102.0
    assert result.report["stale_observations_ignored"] == 1
    assert result.report["state_changed"] is False
    assert result.revisions.empty


def test_newer_unchanged_observation_refreshes_retrieval_time_once():
    existing = frame(minute("2026-10-05T02:15:00Z", "2026-10-05T02:16:00Z"))
    incoming = frame(minute("2026-10-05T02:15:00Z", "2026-10-05T02:17:00Z"))
    result = merge_incremental_minutes(existing, incoming)
    assert result.report["refreshed_rows"] == 1
    assert result.minutes.retrieved_at_utc.item() == pd.Timestamp("2026-10-05T02:17:00Z")
    assert result.revisions.empty


def test_exact_duplicates_inside_snapshot_collapse_to_latest_observation():
    existing = frame()
    incoming = frame(
        minute("2026-10-05T02:15:00Z", "2026-10-05T02:16:00Z"),
        minute("2026-10-05T02:15:00Z", "2026-10-05T02:17:00Z"),
    )
    result = merge_incremental_minutes(existing, incoming)
    assert result.report["incoming_exact_duplicates_removed"] == 1
    assert result.minutes.retrieved_at_utc.item() == pd.Timestamp("2026-10-05T02:17:00Z")


def test_conflicts_without_strictly_newer_retrieval_are_rejected():
    existing = frame(minute("2026-10-05T02:15:00Z", "2026-10-05T02:16:00Z"))
    incoming = frame(minute("2026-10-05T02:15:00Z", "2026-10-05T02:16:00Z", close=101.5))
    with pytest.raises(DataContractError, match="Equal retrieval timestamps"):
        merge_incremental_minutes(existing, incoming)

    conflicting_snapshot = pd.concat([existing, incoming], ignore_index=True)
    with pytest.raises(DataContractError, match="conflicting duplicate"):
        merge_incremental_minutes(frame(), conflicting_snapshot)


@pytest.mark.parametrize("column,value,match", [
    ("timestamp", "2026-10-05 02:15:00", "explicit UTC"),
    ("retrieved_at_utc", "2026-10-05 02:16:00", "explicit UTC"),
    ("source", "OTHER", "configured VCI scope"),
    ("symbol", "fpt", "configured VCI scope"),
])
def test_incremental_scope_and_time_contracts_are_fail_closed(column, value, match):
    existing = frame()
    row = minute("2026-10-05T02:15:00Z", "2026-10-05T02:16:00Z")
    row[column] = value
    with pytest.raises(DataContractError, match=match):
        merge_incremental_minutes(existing, frame(row))



def raw_poll(points):
    keys = {"t": [], "o": [], "h": [], "l": [], "c": [], "v": [], "accumulatedVolume": []}
    for timestamp, close, accumulated in points:
        keys["t"].append(int(pd.Timestamp(timestamp).timestamp()))
        keys["o"].append(100.0)
        keys["h"].append(max(101.0, close))
        keys["l"].append(99.0)
        keys["c"].append(close)
        keys["v"].append(10)
        keys["accumulatedVolume"].append(accumulated)
    return json.dumps([{"symbol": "FPT", **keys}], separators=(",", ":")).encode("utf-8")


def test_poll_store_preserves_raw_hash_versions_revision_and_repeat(tmp_path):
    first_body = raw_poll([("2026-10-05T02:15:00Z", 100.0, 10)])
    first = persist_incremental_poll(tmp_path, "poll-001", first_body, "FPT", "2026-10-05T02:16:10Z")
    raw_path = tmp_path / "polls/poll-001/response.json"
    assert raw_path.read_bytes() == first_body
    assert first.pointer["source_rights_status"] == "pending"
    assert first.pointer["live_evidence"] is False
    pointer_before_repeat = (tmp_path / "current.json").read_bytes()

    repeated = persist_incremental_poll(tmp_path, "poll-001", first_body, "FPT", "2026-10-05T02:16:10Z")
    assert repeated.report["already_applied"] is True
    assert (tmp_path / "current.json").read_bytes() == pointer_before_repeat
    with pytest.raises(DataContractError, match="Immutable incremental artifact differs"):
        persist_incremental_poll(tmp_path, "poll-001", raw_poll([("2026-10-05T02:15:00Z", 90.0, 10)]),
                                 "FPT", "2026-10-05T02:16:10Z")

    second_body = raw_poll([
        ("2026-10-05T02:15:00Z", 101.5, 10),
        ("2026-10-05T02:16:00Z", 102.0, 20),
    ])
    second = persist_incremental_poll(tmp_path, "poll-002", second_body, "FPT", "2026-10-05T02:17:10Z")
    assert second.report["revisions_applied"] == 1
    assert second.report["appended_rows"] == 1
    assert second.revisions.changed_fields.tolist() == ["high|close"]
    current, pointer = load_incremental_current(tmp_path)
    assert len(current) == 2 and pointer["applied_poll_ids"] == ["poll-001", "poll-002"]
    manifest = json.loads((tmp_path / pointer["version_manifest"]).read_text(encoding="utf-8"))
    assert manifest["revisions"][0]["changed_fields"] == "high|close"
    assert manifest["poll_manifest_sha256"] == second.report["poll_manifest_sha256"]


def test_pointer_swap_failure_keeps_old_view_and_retry_finishes(tmp_path, monkeypatch):
    first_body = raw_poll([("2026-10-05T02:15:00Z", 100.0, 10)])
    persist_incremental_poll(tmp_path, "poll-001", first_body, "FPT", "2026-10-05T02:16:10Z")
    old_pointer = (tmp_path / "current.json").read_bytes()
    second_body = raw_poll([
        ("2026-10-05T02:15:00Z", 101.5, 10),
        ("2026-10-05T02:16:00Z", 102.0, 20),
    ])
    original = incremental._atomic_json

    def fail_pointer(path, value):
        assert Path(path).name == "current.json"
        raise OSError("simulated crash before pointer replace")

    monkeypatch.setattr(incremental, "_atomic_json", fail_pointer)
    with pytest.raises(OSError, match="simulated crash"):
        persist_incremental_poll(tmp_path, "poll-002", second_body, "FPT", "2026-10-05T02:17:10Z")
    assert (tmp_path / "current.json").read_bytes() == old_pointer
    current, pointer = load_incremental_current(tmp_path)
    assert len(current) == 1 and pointer["latest_poll_id"] == "poll-001"

    monkeypatch.setattr(incremental, "_atomic_json", original)
    recovered = persist_incremental_poll(tmp_path, "poll-002", second_body, "FPT", "2026-10-05T02:17:10Z")
    assert recovered.pointer["latest_poll_id"] == "poll-002"
    assert len(recovered.minutes) == 2


def test_current_view_hash_tamper_fails_closed(tmp_path):
    body = raw_poll([("2026-10-05T02:15:00Z", 100.0, 10)])
    stored = persist_incremental_poll(tmp_path, "poll-001", body, "FPT", "2026-10-05T02:16:10Z")
    view = tmp_path / stored.pointer["view_file"]
    view.write_bytes(view.read_bytes() + b"tampered")
    with pytest.raises(DataContractError, match="current view checksum mismatch"):
        load_incremental_current(tmp_path)


def test_poll_store_rejects_unsafe_identifier_and_naive_retrieval(tmp_path):
    body = raw_poll([("2026-10-05T02:15:00Z", 100.0, 10)])
    with pytest.raises(DataContractError, match="poll_id"):
        persist_incremental_poll(tmp_path, "../escape", body, "FPT", "2026-10-05T02:16:10Z")
    with pytest.raises(DataContractError, match="explicit UTC"):
        persist_incremental_poll(tmp_path, "poll-001", body, "FPT", "2026-10-05 02:16:10")
