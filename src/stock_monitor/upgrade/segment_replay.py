"""Causal synthetic bar-to-residual adapter and endpoint replay (never LIVE)."""
from datetime import timedelta
import math

from stock_monitor.io import DataContractError
from .contracts import utc_time
from .segments import score_suffixes
from .synthetic import bars_as_of, content_hash


def return_view(observations, *, as_of, split):
    """Resolve revisions at T and cover both close availability times in a return."""
    if split not in {"train", "validation", "evaluation"}:
        raise DataContractError("Invalid residual split")
    frame = bars_as_of(observations, as_of)
    if frame.empty:
        return []
    if set(frame.source) != {"SYNTHETIC"}:
        raise DataContractError("This replay adapter is synthetic only")
    results, previous = [], None
    for row in frame.to_dict("records"):
        stamp = row["timestamp"].isoformat()
        continuous = previous is not None and all(row[key] == previous[key] for key in
            ("symbol", "session_id", "segment_id", "period")) and (
                utc_time(stamp) - utc_time(previous["timestamp"].isoformat()) == timedelta(minutes=5))
        available = utc_time(str(row["available_at"]))
        information = max(available, utc_time(str(previous["available_at"]))) if continuous else available
        revision = f"close:{int(row['bar_revision'])}"
        if continuous:
            revision += f";previous_close:{int(previous['bar_revision'])}"
        results.append({"symbol": row["symbol"], "session_id": row["session_id"],
                        "segment_id": row["segment_id"], "period": row["period"], "timestamp": stamp,
                        "available_at": available.isoformat(), "information_available_at": information.isoformat(),
                        "decision_as_of": as_of, "log_return": math.log(row["close"] / previous["close"]) if continuous else None,
                        "source_mode": "SYNTHETIC", "data_revision": revision, "split": split})
        previous = row
    return results


def replay_endpoints(observations, calibration, *, sessions, model_run, emitted_at, expected_sha256=None):
    """Attempt each initially observed bar once; later delivery cannot rewrite it.

    Decision schedule is the *provided* revision-0 receive time, not bar_end plus
    an invented delay. A late bar still emits at its actual logical receive time;
    earlier/later endpoint snapshots retain their separate as-of views.
    """
    expected = expected_sha256 or content_hash(calibration)
    if expected != content_hash(calibration):
        raise DataContractError("Calibration hash mismatch before replay")
    selected = [row for row in observations if row["session_id"] in sessions]
    initial = sorted([row for row in selected if int(row["bar_revision"]) == 0],
                     key=lambda row: (utc_time(row["available_at"]), row["symbol"], utc_time(row["timestamp"])))
    output = []
    for delivery in initial:
        as_of = utc_time(delivery["available_at"]).isoformat()
        # Bound the view to a single session/symbol; no train/test crossing.
        scoped = [row for row in selected if row["symbol"] == delivery["symbol"] and row["session_id"] == delivery["session_id"]]
        view = return_view(scoped, as_of=as_of, split="evaluation")
        end = utc_time(delivery["timestamp"])
        prefix = [row for row in view if utc_time(row["timestamp"]) <= end]
        if not prefix or utc_time(prefix[-1]["timestamp"]) != end:
            raise DataContractError("Delivered endpoint absent from its visible view")
        current = prefix[-1]
        prefix = [row for row in prefix if row["segment_id"] == current["segment_id"]][-12:]
        result = score_suffixes(prefix, calibration, expected_sha256=expected, model_run=model_run,
                               emitted_at=emitted_at, latest_only=True)[0]
        result["delivery_revision"] = int(delivery["bar_revision"])
        output.append(result)
    return output
