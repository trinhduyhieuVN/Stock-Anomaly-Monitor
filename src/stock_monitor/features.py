"""Causal intraday features; past volume slots never move after a data gap."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import BAR_COLUMNS, DataContractError
from .validation import clock_minutes

FEATURE_ORDER = ("log_return", "range_ratio", "body_ratio", "volume_surprise")


@dataclass
class FeatureResult:
    frame: pd.DataFrame
    report: dict


def compute_features(bars, calendar, *, history_sessions=20, min_history_sessions=10, warmup_sessions=20):
    """Return every input bar, its features and explicit exclusion reasons.

    The rolling horizon is the previous calendar sessions, including missing
    observations as NaN; it is not the last 20 available volume observations.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) for v in
           (history_sessions, min_history_sessions, warmup_sessions)):
        raise DataContractError("Feature history and warm-up must be integer session counts")
    if not 1 <= min_history_sessions <= history_sessions or warmup_sessions < 0:
        raise DataContractError("Invalid history/minimum/warm-up session counts")
    if set(BAR_COLUMNS) - set(bars.columns) or bars.empty:
        raise DataContractError("Features require a nonempty standard bar dataset")
    frame = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise DataContractError("Features require unique symbol/timestamp rows")
    if str(frame.timestamp.dt.tz) != "UTC" or frame.timestamp.isna().any():
        raise DataContractError("Feature timestamps must explicitly be UTC")
    if not frame.interval.eq("5m").all() or not frame.quality_status.eq("ok").all():
        raise DataContractError("Features require checked 5m bars with quality_status ok")
    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, :4] <= 0).any() or (numeric[:, 4] < 0).any():
        raise DataContractError("Features require finite positive prices and nonnegative volume")
    dates = calendar["dates"]
    if dates != sorted(set(dates)) or not dates:
        raise DataContractError("Feature calendar dates must be unique and sorted")
    position = {day: i for i, day in enumerate(dates)}
    left = (frame.timestamp - pd.Timedelta(minutes=5)).dt.tz_convert(calendar["timezone"])
    if not frame.session_id.eq(left.dt.strftime("%Y-%m-%d")).all() or not frame.session_id.isin(dates).all():
        raise DataContractError("Feature dates disagree with bar timestamps/calendar")
    minute = left.dt.hour * 60 + left.dt.minute
    frame["slot"] = ""
    for day in frame.session_id.unique():
        for period in calendar.get("overrides", {}).get(day, calendar["sessions"]):
            begin, end = clock_minutes(period["start"]), clock_minutes(period["end"])
            mask = frame.session_id.eq(day) & minute.ge(begin) & minute.lt(end)
            aligned = (minute - begin).mod(5).eq(0) & left.dt.second.eq(0) & left.dt.microsecond.eq(0) & left.dt.nanosecond.eq(0)
            if (mask & ~aligned).any():
                raise DataContractError("Feature bar does not align with a 5m session slot")
            # Clock slot stays fixed when segment_id changes after a missing bar.
            frame.loc[mask, "slot"] = period["id"] + ":" + left.loc[mask].dt.strftime("%H:%M")
    if frame.slot.eq("").any():
        raise DataContractError("Feature bar is outside continuous-session hours")

    previous = frame.shift()
    continuous = (frame.symbol.eq(previous.symbol) & frame.segment_id.eq(previous.segment_id)
                  & frame.session_id.eq(previous.session_id)
                  & frame.timestamp.sub(previous.timestamp).eq(pd.Timedelta(minutes=5)))
    frame["log_return"] = np.log(frame.close / previous.close).where(continuous)
    frame["range_ratio"] = (frame.high - frame.low) / frame.open
    frame["body_ratio"] = (frame.close - frame.open) / frame.open
    frame["volume_reference"] = np.nan
    frame["volume_history_count"] = 0
    for _, symbol_rows in frame.groupby("symbol", sort=False):
        if symbol_rows.duplicated(["session_id", "slot"]).any():
            raise DataContractError("More than one bar at the same daily volume slot")
        for _, rows in symbol_rows.groupby("slot", sort=False):
            series = rows.set_index("session_id").volume.reindex(dates)
            past_median = series.rolling(history_sessions, min_periods=min_history_sessions).median().shift(1)
            past_count = series.rolling(history_sessions, min_periods=1).count().shift(1).fillna(0)
            frame.loc[rows.index, "volume_reference"] = rows.session_id.map(past_median).to_numpy(dtype=float)
            frame.loc[rows.index, "volume_history_count"] = rows.session_id.map(past_count).to_numpy(dtype=int)
    frame["volume_surprise"] = np.log1p(frame.volume) - np.log1p(frame.volume_reference)
    frame["feature_reasons"] = ""
    flags = {
        "no_previous_bar_in_segment": ~continuous,
        "insufficient_volume_history": frame.volume_history_count.lt(min_history_sessions),
        "warmup_session": frame.session_id.map(position).lt(warmup_sessions),
    }
    for reason, mask in flags.items():
        frame.loc[mask, "feature_reasons"] += reason + "|"
    finite = np.isfinite(frame[list(FEATURE_ORDER)].to_numpy(dtype=float)).all(axis=1)
    bad = frame.feature_reasons.eq("") & ~finite
    frame.loc[bad, "feature_reasons"] += "nonfinite_feature|"
    frame["feature_eligible"] = frame.feature_reasons.eq("")
    by_symbol = {}
    for symbol, rows in frame.groupby("symbol"):
        by_symbol[symbol] = {"bars": len(rows), "eligible_bars": int(rows.feature_eligible.sum()),
            "excluded_bars": int((~rows.feature_eligible).sum()),
            "eligible_dates": int(rows.loc[rows.feature_eligible, "session_id"].nunique())}
    return FeatureResult(frame, {"feature_order": list(FEATURE_ORDER), "bars": len(frame),
        "eligible_bars": int(frame.feature_eligible.sum()), "excluded_bars": int((~frame.feature_eligible).sum()),
        "reason_counts": {key: int(mask.sum()) for key, mask in flags.items()},
        "nonfinite_feature_count": int(bad.sum()), "by_symbol": by_symbol,
        "history_sessions": history_sessions, "minimum_history_sessions": min_history_sessions,
        "warmup_sessions": warmup_sessions, "warmup_dates": dates[:warmup_sessions],
        "volume_slot_rule": "period id + fixed local bar-start HH:MM; previous calendar sessions, excluding current date"})
