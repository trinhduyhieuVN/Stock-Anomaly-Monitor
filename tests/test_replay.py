"""Replay boundary tests on explicitly synthetic data."""

import pandas as pd
import pytest

from stock_monitor.io import DataContractError
from stock_monitor.replay import advance_cursor, replay_prefix, replay_segments


def synthetic_day():
    stamps = pd.date_range("2026-09-01T02:20Z", periods=9, freq="5min")
    return pd.DataFrame({"symbol": ["FPT"] * 9, "timestamp": stamps, "session_id": ["2026-09-01"] * 9,
                         "segment_id": ["morning"] * 9, "split": ["test"] * 9,
                         "window_start": stamps - pd.Timedelta(minutes=25), "score": [0.] * 6 + [3., 4., 0.],
                         "threshold": [2.] * 9, "alert": [False] * 6 + [True, True, False], "reason": ["q98"] * 9})


def test_prefix_is_unchanged_by_future_rows_and_future_mutations():
    frame = synthetic_day()
    cutoff = frame.timestamp.iloc[6]
    observed = replay_prefix(frame, "2026-09-01", cutoff, ["2026-09-01"])
    changed = frame.copy()
    changed.loc[changed.timestamp.gt(cutoff), "score"] = 999999.
    pd.testing.assert_frame_equal(observed, replay_prefix(changed, "2026-09-01", cutoff, ["2026-09-01"]))
    pd.testing.assert_frame_equal(observed, replay_prefix(frame.iloc[:7], "2026-09-01", cutoff, ["2026-09-01"]))
    assert len(observed) == 7 and observed.timestamp.max() == cutoff


def test_open_alert_never_discloses_future_end_or_maximum():
    frame = synthetic_day()
    prefix = replay_prefix(frame, "2026-09-01", frame.timestamp.iloc[6], ["2026-09-01"])
    segment = replay_segments(prefix).iloc[0]
    assert segment.ongoing and segment.bars == 1 and segment.max_score == 3.
    assert segment.end_timestamp == frame.timestamp.iloc[6]
    closed = replay_segments(frame).iloc[0]
    assert not closed.ongoing and closed.bars == 2 and closed.max_score == 4.
    frame.loc[7:, "segment_id"] = "afternoon"
    split = replay_segments(frame)
    assert len(split) == 2 and not split.ongoing.any()


def test_replay_rejects_training_dates_naive_time_and_wrong_day():
    frame = synthetic_day()
    with pytest.raises(DataContractError, match="phiên test"):
        replay_prefix(frame, "2026-09-01", frame.timestamp.iloc[0], [])
    for stamp in ["2026-09-01T02:20", "2026-09-02T02:20Z"]:
        with pytest.raises(DataContractError, match="timezone"):
            replay_prefix(frame, "2026-09-01", stamp, ["2026-09-01"])
    frame.loc[3, "split"] = "train"
    with pytest.raises(DataContractError, match="split test"):
        replay_prefix(frame, "2026-09-01", frame.timestamp.iloc[0], ["2026-09-01"])


def test_cursor_steps_resets_and_stops_at_final_bar():
    assert advance_cursor(0, 9) == 1
    assert advance_cursor(8, 9) == 8
    assert advance_cursor(8, 9, reset=True) == 0
    with pytest.raises(DataContractError):
        advance_cursor(9, 9)
