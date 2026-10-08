import hashlib
import json

import pandas as pd
import pytest

import stock_monitor.realtime as realtime
from stock_monitor.io import DataContractError
from stock_monitor.realtime import (
    ClosedBarReadiness,
    closed_bar_readiness,
    score_ready_closed_bars,
    verify_frozen_scoring_manifest,
)


ROLES = ["feature_scalers", "baseline_models", "baseline_thresholds", "lstm_checkpoint", "scoring_thresholds"]


def frozen_manifest(root):
    files = []
    for role in ROLES:
        path = root / "frozen" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((role + "\n").encode())
        files.append({"role": role, "path": path.relative_to(root).as_posix(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "schema_version": 1,
        "data_policy": {"classification": "local_only_real_data", "source_rights_status": "pending", "distributable": False},
        "files": files,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def profile():
    return {
        "confirmations_pending": ["source_bar_label", "data_usage"],
        "research": True,
        "diagnostic": False,
        "bar_label": "bar_start",
        "price": {"unit": "VND", "multiplier": 1},
        "volume": {"unit": "shares", "multiplier": 1},
        "calendar": {
            "status": "observed_only", "complete": False,
            "timezone": "Asia/Ho_Chi_Minh", "dates": ["2026-10-01"],
            "sessions": [{"id": "morning", "start": "09:15", "end": "11:30"},
                         {"id": "afternoon", "start": "13:00", "end": "14:30"}],
            "overrides": {},
        },
    }


def minutes_at(*ranges):
    rows = []
    accumulated = 0
    for start, count in ranges:
        for offset in range(count):
            accumulated += 10
            rows.append({
                "symbol": "FPT", "timestamp": pd.Timestamp(start) + pd.Timedelta(minutes=offset),
                "source": "VCI", "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0, "volume": 10, "accumulated_volume": accumulated,
                "retrieved_at_utc": pd.Timestamp("2026-10-01T02:31:00Z"),
            })
    return pd.DataFrame(rows)


def readiness(data, root, observed, *, delay=60, stale=600, day_profile=None):
    return closed_bar_readiness(
        data, profile() if day_profile is None else day_profile,
        symbols=["FPT"], observed_at_utc=observed,
        provider_delay_seconds=delay, stale_after_seconds=stale,
        provider_delay_evidence="offline causal-boundary test; not live",
        project_root=root, manifest_path=frozen_manifest(root), delay_measured_live=False,
    )


def test_bar_is_not_eligible_until_close_plus_explicit_delay(tmp_path):
    data = minutes_at(("2026-10-01T02:15:00Z", 5))
    early = readiness(data, tmp_path, "2026-10-01T02:20:59Z")
    assert early.bars.empty
    assert early.status["eligibility_cutoff_utc"] == "2026-10-01T02:19:59+00:00"
    assert early.status["state"] == "WAITING_FIRST_CLOSED_BAR"
    assert early.status["scoring_allowed"] is False

    closed = readiness(data, tmp_path, "2026-10-01T02:21:00Z")
    assert closed.bars.timestamp.tolist() == [pd.Timestamp("2026-10-01T02:20:00Z")]
    assert closed.status["state"] == "READY"
    assert closed.status["scoring_allowed"] is True
    assert closed.status["live_update_allowed"] is False
    assert closed.status["latest_age_seconds"] == 60


def test_gap_count_is_causal_and_latest_closed_bar_can_be_ready(tmp_path):
    data = minutes_at(("2026-10-01T02:15:00Z", 5), ("2026-10-01T02:25:00Z", 5),
                      ("2026-10-01T02:30:00Z", 5))
    result = readiness(data, tmp_path, "2026-10-01T02:31:00Z")
    assert result.bars.timestamp.tolist() == [pd.Timestamp("2026-10-01T02:20:00Z"), pd.Timestamp("2026-10-01T02:30:00Z")]
    assert result.status["gap_count"] == 1
    assert result.status["gaps"] == [{"symbol": "FPT", "bar_end_utc": "2026-10-01T02:25:00+00:00"}]
    assert result.status["state"] == "READY_WITH_GAPS"
    assert result.status["scoring_allowed"] is True
    assert result.status["validation_rejected_bars"] == 1


def test_missing_expected_latest_bar_is_stale_inside_session(tmp_path):
    data = minutes_at(("2026-10-01T02:15:00Z", 5))
    result = readiness(data, tmp_path, "2026-10-01T02:26:00Z")
    assert result.status["expected_latest_bar_end_utc"] == "2026-10-01T02:25:00+00:00"
    assert result.status["latest_bar_end_utc"] == "2026-10-01T02:20:00+00:00"
    assert result.status["state"] == "STALE"
    assert result.status["stale_symbols"] == ["FPT"]
    assert result.status["scoring_allowed"] is False


def test_unknown_calendar_date_blocks_before_normalization(tmp_path, monkeypatch):
    data = minutes_at(("2026-10-02T02:15:00Z", 5))
    monkeypatch.setattr(realtime, "normalize_minutes", lambda *args, **kwargs: pytest.fail("normalization must not run"))
    result = readiness(data, tmp_path, "2026-10-02T02:21:00Z")
    assert result.status["state"] == "BLOCKED_UNKNOWN_CALENDAR"
    assert result.status["calendar_date_known"] is False
    assert result.status["live_update_allowed"] is False


def test_frozen_resources_are_verified_before_normalization(tmp_path, monkeypatch):
    manifest = frozen_manifest(tmp_path)
    verified = verify_frozen_scoring_manifest(tmp_path, manifest)
    assert set(verified["resources"]) == set(ROLES)
    (tmp_path / verified["resources"]["lstm_checkpoint"]["path"]).write_bytes(b"changed")
    monkeypatch.setattr(realtime, "normalize_minutes", lambda *args, **kwargs: pytest.fail("normalization must not run"))
    with pytest.raises(DataContractError, match="checksum mismatch: lstm_checkpoint"):
        closed_bar_readiness(
            minutes_at(("2026-10-01T02:15:00Z", 5)), profile(), symbols=["FPT"],
            observed_at_utc="2026-10-01T02:21:00Z", provider_delay_seconds=60,
            stale_after_seconds=600, provider_delay_evidence="offline test", project_root=tmp_path,
            manifest_path=manifest,
        )


@pytest.mark.parametrize("kwargs,match", [
    ({"provider_delay_seconds": None}, "provider_delay_seconds"),
    ({"provider_delay_seconds": True}, "provider_delay_seconds"),
    ({"stale_after_seconds": 30}, "stale_after_seconds"),
    ({"provider_delay_evidence": ""}, "provider_delay_evidence"),
])
def test_delay_and_stale_inputs_are_never_inferred(tmp_path, kwargs, match):
    values = {"provider_delay_seconds": 60, "stale_after_seconds": 600,
              "provider_delay_evidence": "offline test"}
    values.update(kwargs)
    with pytest.raises(DataContractError, match=match):
        closed_bar_readiness(
            minutes_at(("2026-10-01T02:15:00Z", 5)), profile(), symbols=["FPT"],
            observed_at_utc="2026-10-01T02:21:00Z", project_root=tmp_path,
            manifest_path=frozen_manifest(tmp_path), **values,
        )


def approved_bar(root):
    manifest = frozen_manifest(root)
    bar = pd.DataFrame([{
        "symbol": "FPT", "timestamp": pd.Timestamp("2026-10-01T02:20:00Z"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 50,
        "interval": "5m", "source": "VCI", "session_id": "2026-10-01",
        "segment_id": "FPT:2026-10-01:morning:0", "quality_status": "ok",
    }])
    frozen = verify_frozen_scoring_manifest(root, manifest)
    status = {
        "scoring_allowed": True,
        "eligibility_cutoff_utc": "2026-10-01T02:20:00+00:00",
        "latest_by_symbol": {"FPT": "2026-10-01T02:20:00+00:00"},
    }
    return ClosedBarReadiness(bar, status, frozen), manifest


def test_incremental_scoring_rejects_any_future_history_before_loading_models(tmp_path, monkeypatch):
    ready, manifest = approved_bar(tmp_path)
    future = ready.bars.copy()
    future["timestamp"] = pd.Timestamp("2026-10-01T02:25:00Z")
    monkeypatch.setattr(realtime, "_checked_scoring_directories",
                        lambda *args: pytest.fail("models must not load after a future-data violation"))
    with pytest.raises(DataContractError, match="beyond the readiness cutoff"):
        score_ready_closed_bars(
            ready, future, profile(), project_root=tmp_path,
            received_at_utc="2026-10-01T02:21:00Z", manifest_path=manifest,
        )


def test_incremental_scoring_rechecks_manifest_after_readiness(tmp_path, monkeypatch):
    ready, manifest = approved_bar(tmp_path)
    checkpoint = tmp_path / ready.frozen_resources["resources"]["lstm_checkpoint"]["path"]
    checkpoint.write_bytes(b"changed-after-readiness")
    monkeypatch.setattr(realtime, "_checked_scoring_directories",
                        lambda *args: pytest.fail("models must not load after a checksum violation"))
    with pytest.raises(DataContractError, match="checksum mismatch: lstm_checkpoint"):
        score_ready_closed_bars(
            ready, ready.bars.iloc[0:0].copy(), profile(), project_root=tmp_path,
            received_at_utc="2026-10-01T02:21:00Z", manifest_path=manifest,
        )
