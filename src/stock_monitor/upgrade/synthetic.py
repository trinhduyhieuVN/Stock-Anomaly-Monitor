"""Independent, deterministic bar fixtures; no market file or detector inputs."""
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import random

import numpy as np
import pandas as pd

from stock_monitor.io import DataContractError, read_bars
from stock_monitor.validation import assign_segments
from .contracts import LOCAL_ZONE, utc_time, validate_news

CASES = ["isolated_spike", "mild_shift_3", "mild_shift_6", "mild_shift_12",
         "session_volume", "news_context", "gap", "revision", "delayed_outcome", "censored"]
PARTITIONS = {"development": "development_seeds", "calibration": "calibration_seeds",
              "holdout": "holdout_seeds"}


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def content_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def create_spec(config):
    return {"schema_version": 2, "generator_version": 1, "source": "SYNTHETIC",
            "event_unit": "bar_5m", "timestamp_convention": "bar_end",
            "calendar_status": "artificial_weekdays_not_exchange_calendar",
            "start_date": "2025-01-06", "symbols": list(config["symbols"]),
            "sessions": config["synthetic"]["sessions"],
            "warmup_sessions": config["synthetic"]["warmup_sessions"],
            "seed_partitions": {key: list(config["synthetic"][value]) for key, value in PARTITIONS.items()},
            "case_order": list(CASES), "periods": [{"id": "am", "start": "09:00", "bars": 30},
                                                    {"id": "pm", "start": "13:00", "bars": 18}],
            "parameters": {"base_price": 100.0, "normal_return_sigma": 0.00035,
                           "mild_shift_return": 0.0012, "shift_noise_bound": 0.0002,
                           "isolated_spike_return": 0.02, "outcome_event_return": 0.008,
                           "base_volume": 1000, "pm_volume_multiplier": 2.0,
                           "logical_receive_seconds": 2, "delayed_receive_minutes": 12},
            "units": {"price": "arbitrary_synthetic_units", "volume": "synthetic_integer_counts"},
            "truth_source": "generator_injection_spec_not_detector_or_human_review",
            "default_truth": "normal_for_observed_valid_bars_outside_injections",
            "purpose": {"development": "functional_fixtures_and_design",
                        "calibration": "synthetic_threshold_calibration_only",
                        "holdout": "frozen_synthetic_benchmark_not_market_holdout"}}


def _dates(spec):
    day = date.fromisoformat(spec["start_date"])
    days = []
    while len(days) < spec["sessions"]:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _rng(seed, symbol, day):
    key = f"v1:{seed}:{symbol}:{day.isoformat()}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16], "big"))


def generate(spec, seed, partition="development"):
    """Truth is emitted from the injection plan, never inferred from a model score."""
    if partition not in PARTITIONS or seed not in spec["seed_partitions"][partition] or type(seed) is not int:
        raise DataContractError("Seed must belong to its declared partition")
    if (spec["source"] != "SYNTHETIC" or spec["case_order"] != CASES or
            spec["sessions"] < spec["warmup_sessions"] + len(CASES)):
        raise DataContractError("Invalid independent synthetic scenario specification")
    params = spec["parameters"]
    rows, truth, news, probes = [], [], [], []
    for symbol_index, symbol in enumerate(spec["symbols"]):
        for day_index, day in enumerate(_dates(spec)):
            case = "normal" if day_index < spec["warmup_sessions"] else CASES[(day_index - spec["warmup_sessions"]) % len(CASES)]
            rng = _rng(seed, symbol, day)
            price = params["base_price"] + 25 * symbol_index
            day_rows, period_stamps = [], {}
            for period in spec["periods"]:
                begin = datetime.combine(day, time.fromisoformat(period["start"]), tzinfo=LOCAL_ZONE)
                stamps = [begin + timedelta(minutes=5 * (index + 1)) for index in range(period["bars"])]
                period_stamps[period["id"]] = stamps
                for index, stamp in enumerate(stamps):
                    ret = rng.gauss(0, params["normal_return_sigma"])
                    affected = period["id"] == "am"
                    if affected and case.startswith("mild_shift_") and 14 <= index < 14 + int(case.rsplit("_", 1)[1]):
                        ret = params["mild_shift_return"] + rng.uniform(-params["shift_noise_bound"], params["shift_noise_bound"])
                    elif affected and case == "news_context" and 18 <= index < 24:
                        ret = params["mild_shift_return"] + rng.uniform(-params["shift_noise_bound"], params["shift_noise_bound"])
                    elif affected and case == "isolated_spike" and index == 18:
                        ret = params["isolated_spike_return"]
                    elif (affected and case == "delayed_outcome" and index == 18) or (period["id"] == "pm" and case == "censored" and index == period["bars"] - 1):
                        ret = params["outcome_event_return"]
                    open_, close = price, price * math.exp(ret)
                    wick = price * rng.uniform(0.00005, 0.0003)
                    price = close
                    volume_multiplier = params["pm_volume_multiplier"] if period["id"] == "pm" else 1.0
                    volume = round(params["base_volume"] * volume_multiplier * rng.uniform(0.9, 1.1))
                    available = stamp + timedelta(seconds=params["logical_receive_seconds"])
                    if affected and case == "delayed_outcome" and index == 19:
                        available += timedelta(minutes=params["delayed_receive_minutes"])
                    row = {"symbol": symbol, "timestamp": _stamp(stamp), "open": round(open_, 8),
                           "high": round(max(open_, close) + wick, 8), "low": round(min(open_, close) - wick, 8),
                           "close": round(close, 8), "volume": volume, "interval": "5m", "source": "SYNTHETIC",
                           "session_id": day.isoformat(), "period": period["id"], "quality_status": "ok",
                           "bar_revision": 0, "available_at": _stamp(available)}
                    if not (affected and case == "gap" and index == 18):
                        day_rows.append(row)
            base = pd.DataFrame(day_rows)
            base["timestamp"] = pd.to_datetime(base.timestamp, utc=True)
            segments = assign_segments(base)
            segment_by_time = {_stamp(stamp): segment for stamp, segment in zip(segments.timestamp, segments.segment_id)}
            for row in day_rows:
                row["segment_id"] = segment_by_time[row["timestamp"]]
            am, pm = period_stamps["am"], period_stamps["pm"]
            if case == "revision":
                original = next(row for row in day_rows if row["timestamp"] == _stamp(am[18]))
                revised = deepcopy(original)
                revised.update(bar_revision=1, available_at=_stamp(am[20] + timedelta(seconds=3)),
                               close=round(original["close"] * 1.003, 8))
                revised["high"] = max(revised["high"], revised["close"] + 0.01)
                day_rows.append(revised)
            rows.extend(day_rows)
            def label(kind, start, end, purpose):
                truth.append({"truth_id": f"{seed}:{symbol}:{day}:{case}:{_stamp(start)}:{kind}", "symbol": symbol,
                              "session_id": day.isoformat(), "scenario": case, "kind": kind,
                              "event_start": _stamp(start), "event_end": _stamp(end),
                              "segment_id": segment_by_time.get(_stamp(start)), "label_source": "synthetic",
                              "purpose": purpose, "partition": partition})
            if case in {"normal", "session_volume"}:
                for stamps in (am, pm):
                    label("normal", stamps[0], stamps[-1], "Normal coverage; regular AM/PM volume differs without an anomaly")
            elif case.startswith("mild_shift_"):
                length = int(case.rsplit("_", 1)[1])
                label("collective", am[14], am[14 + length - 1], "Sustained small returns; no promise about a fitted point threshold")
            elif case == "news_context":
                label("collective", am[18], am[23], "Known-later synthetic context does not erase anomaly truth")
                item = {"schema_version": 2, "event_unit": "bar_5m", "symbol": symbol, "source_mode": "SYNTHETIC",
                        "news_id": f"{seed}:{symbol}:{day}:news", "title": "SYNTHETIC scheduled announcement",
                        "url": "https://example.org/synthetic-not-a-market-source", "published_at": _stamp(am[18] - timedelta(minutes=1)),
                        "known_at": _stamp(am[18] + timedelta(minutes=7)), "available_at": _stamp(am[18] + timedelta(minutes=7, seconds=1)),
                        "time_basis": "synthetic"}
                news.append(validate_news(item))
            elif case in {"gap", "revision"}:
                label("insufficient_data", am[18], am[18], "Data quality probe, not a normal or price-anomaly label")
            elif case == "isolated_spike":
                label("point", am[18], am[18], "One injected large return, not a collective interval")
            else:
                origin = pm[-1] if case == "censored" else am[18]
                label("point", origin, origin, "Injected event with separate outcome availability probe")
                for horizon in (1, 3):
                    target = origin + timedelta(minutes=5 * horizon)
                    target_rows = [row for row in day_rows if row["timestamp"] == _stamp(target)]
                    probes.append({"probe_id": f"{seed}:{symbol}:{day}:h{horizon}", "symbol": symbol,
                                   "session_id": day.isoformat(), "origin_bar_end": _stamp(origin),
                                   "target_bar_end": _stamp(target), "horizon_bars": horizon,
                                   "status": "censored" if not target_rows else "pending_until_outcome_available",
                                   "outcome_available_at": target_rows[0]["available_at"] if target_rows else None,
                                   "reason": "cross_period_or_session" if not target_rows else "logical_fixture_receive_time",
                                   "source": "SYNTHETIC"})
    rows.sort(key=lambda row: (row["symbol"], row["timestamp"], row["bar_revision"]))
    result = {"bars": rows, "truth": truth, "news": news, "outcome_probes": probes,
              "metadata": {"source": "SYNTHETIC", "seed": seed, "partition": partition,
                           "spec_sha256": content_hash(spec), "sessions_per_symbol": spec["sessions"],
                           "warmup_sessions": spec["warmup_sessions"], "cases": ["normal", *CASES],
                           "truth_default": spec["default_truth"], "live_latency": "unavailable",
                           "fixture_receive_times": "logical_simulated_not_observed"}}
    validate_observations(pd.DataFrame(rows))
    return result


def validate_observations(frame):
    if frame.empty or not frame.source.eq("SYNTHETIC").all() or not frame.interval.eq("5m").all() or not frame.quality_status.eq("ok").all():
        raise DataContractError("Expected valid independent SYNTHETIC 5m observations")
    numeric = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or numeric[["open", "high", "low", "close"]].le(0).any().any():
        raise DataContractError("Nonfinite/nonpositive OHLCV")
    if (frame.high.lt(frame[["open", "close"]].max(axis=1)) | frame.low.gt(frame[["open", "close"]].min(axis=1)) | frame.low.gt(frame.high)).any():
        raise DataContractError("Invalid OHLC relation")
    if (frame.volume.lt(0) | frame.volume.mod(1).ne(0)).any():
        raise DataContractError("Invalid synthetic integer volume")
    if frame.duplicated(["symbol", "timestamp", "bar_revision"]).any():
        raise DataContractError("Duplicate observation version")
    for row in frame.itertuples(index=False):
        stamp, available = utc_time(str(row.timestamp)), utc_time(str(row.available_at))
        if (stamp.second or stamp.microsecond or stamp.minute % 5 or available < stamp or
                stamp.astimezone(LOCAL_ZONE).date().isoformat() != row.session_id or
                isinstance(row.bar_revision, bool) or row.bar_revision < 0 or int(row.bar_revision) != row.bar_revision):
            raise DataContractError("Invalid bar timestamp/session/revision/availability")
    return frame


def bars_as_of(rows, as_of):
    """Materialize only known revisions; recalculate gap segments at the cutoff."""
    cutoff = utc_time(as_of, "as_of")
    frame = pd.DataFrame(rows)
    validate_observations(frame)
    visible = frame.loc[frame.available_at.map(utc_time).le(cutoff)].copy()
    visible = visible.sort_values(["symbol", "timestamp", "bar_revision"]).drop_duplicates(["symbol", "timestamp"], keep="last")
    visible["timestamp"] = pd.to_datetime(visible.timestamp, utc=True)
    return assign_segments(visible)


def news_as_of(items, as_of):
    cutoff = utc_time(as_of)
    return [validate_news(item, as_of=as_of) for item in items if utc_time(item["available_at"]) <= cutoff]


def _member(root, name):
    if not isinstance(name, str) or Path(name).is_absolute() or PureWindowsPath(name).is_absolute() or "\\" in name:
        raise DataContractError("Invalid public manifest member")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DataContractError("Manifest path escapes root")
    if not (name.startswith("demo/upgrade_v2/") or (name.startswith("src/stock_monitor/") and name.endswith(".py")) or
            name in {"scripts/build_upgrade_fixture.py", "scripts/__init__.py", "configs/upgrade_v2.yaml"}):
        raise DataContractError("Only synthetic bundle and source code paths may be distributed")
    return path


def verify_bundle(root, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 2 or manifest.get("bundle_kind") != "upgrade_v2_synthetic" or
            manifest.get("data_policy") != {"source": "SYNTHETIC", "contains_real_data": False, "distributable": True}):
        raise DataContractError("Not a distributable synthetic bundle")
    files = manifest["files"]
    if len({item["path"] for item in files}) != len(files):
        raise DataContractError("Duplicate manifest member")
    for item in files:
        path = _member(Path(root), item["path"])
        role = item["role"]
        if role not in {"source_code_or_config", "synthetic_observations", "synthetic_truth_news_probes", "synthetic_spec"}:
            raise DataContractError("Unknown synthetic manifest role")
        if ((role == "source_code_or_config" and path.suffix not in {".py", ".yaml"}) or
                (role != "source_code_or_config" and not item["path"].startswith("demo/upgrade_v2/"))):
            raise DataContractError("Synthetic manifest role/path mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise DataContractError("Synthetic manifest hash mismatch")
        if role == "synthetic_spec" and item["sha256"] != manifest["spec_sha256"]:
            raise DataContractError("Spec hash mismatch")
        if item["role"] == "synthetic_observations":
            frame = read_bars(path, symbols=["FPT", "HPG", "VCB"], source="SYNTHETIC")
            validate_observations(frame)
    if sum(item["role"] == "synthetic_spec" for item in files) != 1:
        raise DataContractError("Expected exactly one frozen spec")
    return {"verified_files": len(files), "spec_sha256": manifest["spec_sha256"],
            "seed_partitions": manifest["seed_partitions"], "data_policy": manifest["data_policy"]}
