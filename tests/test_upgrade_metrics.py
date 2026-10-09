from datetime import datetime, timedelta, timezone

import pytest

from stock_monitor.io import DataContractError
from stock_monitor.upgrade.metrics import counts, match_segments, point_metrics


def interval(identifier, start, end, truth=False, symbol="FPT", day="2025-02-03"):
    origin = datetime(2025, 2, 3, 2, 5, tzinfo=timezone.utc)
    return {"truth_id" if truth else "event_id": identifier, "symbol": symbol, "session_id": day,
            "event_start": (origin + timedelta(minutes=5 * start)).isoformat(),
            "event_end": (origin + timedelta(minutes=5 * end)).isoformat()}


def test_binary_counts_and_zero_denominators():
    result = point_metrics([True, True, False, False], [True, False, True, False])
    assert (result["tp"], result["fp"], result["fn"], result["f1"]) == (1, 1, 1, .5)
    assert counts(0, 0, 0)["f1"] is None
    assert counts(0, 0, 2)["precision"] is None
    assert counts(0, 2, 2)["f1"] == 0
    with pytest.raises(DataContractError):
        point_metrics([1], [True])


def test_overlapping_duplicate_emissions_not_multiple_tp():
    p = [interval("a", 0, 5), interval("b", 1, 6), interval("c", 0, 5)]
    result = match_segments(p, [interval("t", 0, 5, True)])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["precision"] == 1 / 3 and result["recall"] == 1
    assert result["matches"][0]["event_id"] == "a"
    assert result == match_segments(list(reversed(p)), [interval("t", 0, 5, True)])


def test_cardinality_before_iou_and_boundary_errors():
    p = [interval("p1", 0, 9), interval("p2", 0, 2)]
    t = [interval("t1", 0, 5, True), interval("t2", 6, 11, True)]
    result = match_segments(p, t, minimum_iou=.3)
    assert result["tp"] == 2 and result["fp"] == result["fn"] == 0
    pairs = {row["event_id"]: row for row in result["matches"]}
    assert pairs["p1"]["truth_id"] == "t2" and pairs["p1"]["start_error_bars"] == -6
    assert pairs["p2"]["truth_id"] == "t1"
    assert match_segments(p, t, minimum_iou=.5)["tp"] == 1


def test_symbol_session_boundaries_and_empty_predictions():
    t = [interval("t", 0, 5, True)]
    p = [interval("p", 0, 5, symbol="HPG"), interval("q", 0, 5, day="2025-02-04")]
    assert match_segments(p, t)["tp"] == 0
    assert match_segments([], t)["misses"] == ["t"]
    with pytest.raises(DataContractError):
        match_segments([interval("p", 4, 1)], t)
    with pytest.raises(DataContractError):
        match_segments([interval("p", 0, 1)] * 2, t)
