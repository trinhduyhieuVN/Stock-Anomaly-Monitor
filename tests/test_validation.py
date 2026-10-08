"""Market boundaries using synthetic fixtures, never a production dataset."""

import json

import pandas as pd
import pytest

from stock_monitor.io import BAR_COLUMNS
from stock_monitor.providers import ProviderError, fetch_vci_raw, parse_vci_minutes
from stock_monitor.validation import SourceNotConfirmed, audit_raw_minutes, normalize_minutes, validate_standard_bars


@pytest.fixture
def profile():
    return {"bar_label": "bar_start", "price": {"unit": "VND", "multiplier": 1},
        "volume": {"unit": "shares", "multiplier": 1}, "diagnostic": False,
        "confirmations_pending": [], "calendar": {"timezone": "Asia/Ho_Chi_Minh", "complete": True,
            "dates": ["2026-04-01", "2026-04-02"], "sessions": [
                {"id": "morning", "start": "09:15", "end": "09:35"},
                {"id": "afternoon", "start": "13:00", "end": "13:10"}]}}


@pytest.fixture
def minutes():
    stamps = pd.date_range("2026-04-01T02:15:00Z", periods=20, freq="min").append(
        pd.date_range("2026-04-01T06:00:00Z", periods=10, freq="min"))
    return pd.DataFrame({"timestamp": stamps, "symbol": "FPT", "source": "VCI", "raw_row": range(30),
        "open": 100000, "high": 100200, "low": 99900, "close": 100100, "volume": 100,
        "accumulated_volume": [100 * (i + 1) for i in range(30)]})


def normalize(minutes, profile, as_of="2026-04-03T00:00:00Z"):
    return normalize_minutes(minutes, profile, start="2026-04-01", end="2026-04-02", symbols=["FPT"], as_of=as_of)


def test_bar_end_units_and_lunch_boundary(minutes, profile):
    result = normalize(minutes, profile)
    assert list(result.bars.columns) == list(BAR_COLUMNS)
    assert len(result.bars) == 6
    assert result.bars.iloc[0].timestamp == pd.Timestamp("2026-04-01T02:20:00Z")
    assert result.bars.volume.eq(500).all()
    assert result.bars.open.eq(100000).all()
    assert result.bars.iloc[3].segment_id != result.bars.iloc[4].segment_id
    assert result.bars.quality_status.eq("ok").all()
    assert result.report["missing_bar_groups"] == 6  # Entire second synthetic session date absent.
    assert not result.report["coverage_passed"]


def test_missing_minute_rejects_group_and_breaks_segment(minutes, profile):
    minutes = minutes.loc[minutes.timestamp.ne(pd.Timestamp("2026-04-01T02:22:00Z"))].copy()
    result = normalize(minutes, profile)
    assert result.report["bar_reasons"]["incomplete_minutes"] == 1
    assert not result.bars.timestamp.eq(pd.Timestamp("2026-04-01T02:25:00Z")).any()
    assert result.bars.iloc[0].segment_id != result.bars.iloc[1].segment_id
    assert not result.bars.volume.eq(0).any()


def test_conflicting_duplicates_reject_entire_key_and_bar(minutes, profile):
    duplicate = minutes.iloc[[1]].copy()
    duplicate["close"] = 100150
    result = normalize(pd.concat([minutes, duplicate], ignore_index=True), profile)
    assert result.report["source_row_reasons"]["conflicting_duplicate"] == 2
    assert not result.bars.timestamp.eq(pd.Timestamp("2026-04-01T02:20:00Z")).any()
    assert result.report["exact_duplicates_removed"] == 0


def test_identical_duplicates_are_removed_without_extra_volume(minutes, profile):
    result = normalize(pd.concat([minutes, minutes.iloc[[1]]], ignore_index=True), profile)
    assert result.report["exact_duplicates_removed"] == 1
    assert len(result.bars) == 6
    assert result.bars.volume.eq(500).all()


@pytest.mark.parametrize("column,value,reason", [
    ("volume", -1, "negative_volume"), ("volume", 1.5, "fractional_volume"),
    ("high", 90000, "invalid_ohlc"), ("low", 0, "nonpositive_price"),
    ("close", float("nan"), "nonfinite_ohlcv"),
    ("timestamp", pd.NaT, "invalid_timestamp"),
    ("accumulated_volume", -1, "invalid_accumulated_volume"),
])
def test_dirty_minute_is_counted_and_not_used(minutes, profile, column, value, reason):
    if column != "timestamp":
        minutes[column] = minutes[column].astype(float)
    minutes.loc[1, column] = value
    result = normalize(minutes, profile)
    assert result.report["source_row_reasons"][reason] >= 1
    assert not result.bars.timestamp.eq(pd.Timestamp("2026-04-01T02:20:00Z")).any()


def test_unclosed_bars_are_rejected(minutes, profile):
    result = normalize(minutes, profile, as_of="2026-04-01T02:18:30Z")
    assert result.bars.empty
    assert result.report["bar_reasons"]["unclosed_bar"] == 6


def test_auction_is_excluded_without_fabricating_continuous_bars(minutes, profile):
    auction = minutes.iloc[[-1]].copy()
    auction["timestamp"] = pd.Timestamp("2026-04-01T07:45:00Z")
    auction["accumulated_volume"] = 3100
    result = normalize(pd.concat([minutes, auction], ignore_index=True), profile)
    assert len(result.bars) == 6
    assert result.report["source_row_reasons"]["outside_continuous_session"] == 1


def test_source_end_labels_are_shifted_before_aggregation(minutes, profile):
    minutes["timestamp"] += pd.Timedelta(minutes=1)
    profile["bar_label"] = "bar_end"
    result = normalize(minutes, profile)
    assert result.bars.iloc[0].timestamp == pd.Timestamp("2026-04-01T02:20:00Z")


def test_explicit_conversion_and_diagnostic_quality(minutes, profile):
    minutes[["open", "high", "low", "close"]] = minutes[["open", "high", "low", "close"]] / 1000
    profile["price"] = {"unit": "thousand_VND", "multiplier": 1000}
    profile["diagnostic"] = True
    result = normalize(minutes, profile)
    assert result.bars.open.eq(100000).all()
    assert result.bars.quality_status.eq("unverified").all()
    assert result.report["accepted_bars"] == 0
    assert result.report["unverified_bars"] == 6
    assert not result.report["training_ready"]


@pytest.mark.parametrize("value", ["2026-04-01 09:20:00", "invalid", "2026-04-01T09:20:00+07:00"])
def test_standard_timestamp_never_guesses_timezone(minutes, profile, value):
    bars = normalize(minutes, profile).bars
    bars["timestamp"] = bars.timestamp.astype(str)
    bars.loc[0, "timestamp"] = value
    result = validate_standard_bars(bars, profile, start="2026-04-01", end="2026-04-02", symbols=["FPT"])
    assert result.report["source_row_reasons"]["invalid_timestamp"] == 1
    assert result.report["status"] == "INVALID_INPUT"


def test_standard_segment_cannot_cross_lunch(minutes, profile):
    bars = normalize(minutes, profile).bars
    bars["segment_id"] = "2026-04-01:FPT:1"
    result = validate_standard_bars(bars, profile, start="2026-04-01", end="2026-04-02", symbols=["FPT"])
    assert result.bars.empty
    assert result.report["source_row_reasons"]["segment_crosses_boundary"] == 6


def test_vci_array_shape_and_invalid_timestamp():
    payload = [{"symbol": "FPT", "t": ["invalid"], "o": [1], "h": [2], "l": [1], "c": [2], "v": [1], "accumulatedVolume": [1]}]
    assert parse_vci_minutes(payload, "FPT").timestamp.isna().all()
    payload[0]["v"] = []
    with pytest.raises(ProviderError, match="unequal lengths"):
        parse_vci_minutes(payload, "FPT")


def valid_body():
    return json.dumps([{"symbol": "FPT", "t": ["1775009700"], "o": [100000], "h": [100100],
        "l": [99900], "c": [100000], "v": [100], "accumulatedVolume": [100]}]).encode()


def test_http_backoff_cap_and_response_preservation(tmp_path):
    responses = iter([(503, {}, b"unavailable"), (200, {}, valid_body())])
    waits = []
    raw, metadata = fetch_vci_raw(tmp_path, "FPT", "2026-04-01", "2026-04-01", 100,
        transport=lambda payload, timeout: next(responses), sleep=waits.append)
    assert waits == [1]
    assert raw.name == "response_02.json"
    assert raw.with_name("response_01.json").read_bytes() == b"unavailable"
    assert json.loads(metadata.read_text(encoding="utf-8"))["status"] == "received"


@pytest.mark.parametrize("status,headers", [(401, {}), (403, {}), (429, {}), (429, {"Retry-After": "120"}), (400, {})])
def test_access_or_quota_failure_stops_immediately(tmp_path, status, headers):
    attempts, waits = [], []
    def transport(payload, timeout):
        attempts.append(payload)
        return status, headers, b"blocked"
    with pytest.raises(ProviderError):
        fetch_vci_raw(tmp_path, "FPT", "2026-04-01", "2026-04-01", 100, transport=transport, sleep=waits.append)
    assert len(attempts) == 1
    assert waits == []
    meta = next(tmp_path.glob("vci/*/metadata.json"))
    assert json.loads(meta.read_text(encoding="utf-8"))["status"] == "failed"


def test_retry_budget_does_not_continue_forever(tmp_path):
    attempts, waits = [], []
    def transport(payload, timeout):
        attempts.append(1)
        return 503, {}, b"unavailable"
    with pytest.raises(ProviderError):
        fetch_vci_raw(tmp_path, "FPT", "2026-04-01", "2026-04-01", 100, transport=transport, sleep=waits.append)
    assert len(attempts) == 3
    assert waits == [1, 2]


def test_raw_diagnostic_preserves_prices_volumes_and_source_labels(minutes, profile):
    profile["diagnostic"] = True
    profile["confirmations_pending"] = ["source_bar_label", "source_price_unit", "source_volume_unit"]
    # Deliberately wrong conversion instructions cannot affect raw auditing.
    profile["price"] = {"unit": "thousand_VND", "multiplier": 1000}
    profile["volume"] = {"unit": "lot_100_shares", "multiplier": 100}
    profile["bar_label"] = "bar_end"
    result = audit_raw_minutes(minutes, profile, start="2026-04-01", end="2026-04-02", symbols=["FPT"], as_of="2026-04-03T00:00:00Z")
    assert result.bars.timestamp.tolist() == minutes.timestamp.tolist()
    assert result.bars.open.tolist() == minutes.open.tolist()
    assert result.bars.volume.tolist() == minutes.volume.tolist()
    assert result.bars.interval.eq("1m").all()
    assert result.bars.quality_status.eq("unverified").all()
    assert not result.report["unit_conversion_applied"]
    assert result.report["label_shift_seconds"] == 0
    with pytest.raises(SourceNotConfirmed):
        normalize(minutes, profile)


def test_http_quota_wait_is_respected_within_budget(tmp_path):
    responses = iter([(429, {"Retry-After": "3"}, b"limited"), (200, {}, valid_body())])
    waits = []
    fetch_vci_raw(tmp_path, "FPT", "2026-04-01", "2026-04-01", 100,
        transport=lambda payload, timeout: next(responses), sleep=waits.append)
    assert waits == [3]


def test_http_200_access_message_is_not_retried(tmp_path):
    attempts = []
    def transport(payload, timeout):
        attempts.append(1)
        return 200, {}, b'{"error": "access denied"}'
    with pytest.raises(ProviderError, match="access-denied"):
        fetch_vci_raw(tmp_path, "FPT", "2026-04-01", "2026-04-01", 100, transport=transport)
    assert attempts == [1]
