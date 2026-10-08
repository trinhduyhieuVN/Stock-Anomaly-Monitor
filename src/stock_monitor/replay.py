"""Historical test replay using immutable, causally audited artifacts."""

import pandas as pd

from .alerts import group_consecutive_alerts
from .io import DataContractError


def replay_dates(feature_metadata):
    return tuple(feature_metadata["validation"]["split"]["dates"]["test"])


def replay_prefix(frame, session_id, cutoff, test_dates):
    if session_id not in test_dates:
        raise DataContractError("Replay chỉ được dùng phiên test đóng băng")
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or cutoff.tz_convert("Asia/Ho_Chi_Minh").date().isoformat() != session_id:
        raise DataContractError("Thời điểm replay phải có timezone và nằm trong ngày đã chọn")
    day = frame.loc[frame.session_id.eq(session_id)]
    if day.empty or not day.split.dropna().eq("test").all():
        raise DataContractError("Phiên replay không khớp split test")
    return day.loc[day.timestamp.le(cutoff)].sort_values(["symbol", "timestamp"]).copy()


def replay_segments(selected):
    segments = group_consecutive_alerts(selected)
    segments["ongoing"] = False
    for symbol, rows in selected.groupby("symbol"):
        latest = rows.sort_values("timestamp").iloc[-1]
        if latest.alert:
            mask = segments.symbol.eq(symbol) & segments.end_timestamp.eq(latest.timestamp)
            segments.loc[mask, "ongoing"] = True
    return segments


def advance_cursor(position, total, *, reset=False):
    if total < 1 or not 0 <= position < total:
        raise DataContractError("Vị trí replay không hợp lệ")
    return 0 if reset else min(position + 1, total - 1)
