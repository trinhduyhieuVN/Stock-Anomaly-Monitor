"""Offline causal mean-shift suffixes; not CAPA, likelihood or point grouping."""
from bisect import bisect_left, bisect_right
from datetime import timedelta
import math

import numpy as np

from stock_monitor.io import DataContractError
from .contracts import MODES, _session_bar, utc_time, validate_candidate, validate_event
from .synthetic import content_hash

ALGORITHM = "causal_return_suffix_heuristic_v1"
LENGTHS = (3, 6, 12)
PENALTY = 0.5
FIELDS = {"symbol", "session_id", "segment_id", "period", "timestamp", "available_at",
          "information_available_at", "decision_as_of", "log_return", "source_mode",
          "data_revision", "split"}


def _rows(rows, split):
    result, seen = [], set()
    for value in rows:
        if not isinstance(value, dict) or set(value) != FIELDS:
            raise DataContractError("Expected residual fields only; no truth/label/alert columns")
        row = dict(value)
        if row["split"] != split or row["source_mode"] not in MODES:
            raise DataContractError("Invalid residual split/source mode")
        if row["symbol"] not in {"FPT", "HPG", "VCB"} or row["period"] not in {"am", "pm"}:
            raise DataContractError("Invalid symbol/period")
        for name in ("segment_id", "data_revision"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise DataContractError(f"Missing {name}")
        times = {name: utc_time(row[name], name) for name in
                 ("timestamp", "available_at", "information_available_at", "decision_as_of")}
        _session_bar(times["timestamp"], row["session_id"])
        if not times["timestamp"] <= times["available_at"] <= times["information_available_at"]:
            raise DataContractError("Residual availability must cover the bar and its inputs")
        if times["decision_as_of"] < times["timestamp"]:
            raise DataContractError("Decision precedes bar close")
        raw = row["log_return"]
        if raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
            raise DataContractError("log_return must be numeric or missing")
        if raw is not None and math.isinf(raw):
            raise DataContractError("Infinite log_return")
        row["log_return"] = None if raw is None or math.isnan(raw) else float(raw)
        key = (row["symbol"], times["timestamp"])
        if key in seen:
            raise DataContractError("Duplicate bar/revision; materialize one as-of view first")
        seen.add(key)
        result.append(row)
    result.sort(key=lambda row: (row["symbol"], utc_time(row["timestamp"])))
    if len({row["source_mode"] for row in result}) != 1:
        raise DataContractError("Require nonempty single-source residual data")
    for previous, current in zip(result, result[1:]):
        if previous["symbol"] == current["symbol"] and (
                utc_time(previous["decision_as_of"]) > utc_time(current["decision_as_of"])):
            raise DataContractError("Decision chronology is not monotone")
    return result


def fit_return_normalizer(rows):
    """Fit new symbol median/MAD from train only, without touching v1 models."""
    return fit_return_normalizer_replicates([rows])


def fit_return_normalizer_replicates(replicates):
    """Pool independent seeds without mistaking replicas for duplicate revisions."""
    prepared = [_rows(rows, "train") for rows in replicates]
    if not prepared or len({rows[0]["source_mode"] for rows in prepared}) != 1:
        raise DataContractError("Require nonempty single-source train replicates")
    rows = [row for replica in prepared for row in replica]
    if any(utc_time(row["information_available_at"]) > utc_time(row["decision_as_of"]) for row in rows):
        raise DataContractError("Train input unavailable at its decision")
    symbols = {}
    for symbol in sorted({row["symbol"] for row in rows}):
        values = [row["log_return"] for row in rows if row["symbol"] == symbol and row["log_return"] is not None]
        median = float(np.median(values)) if values else None
        scale = float(1.4826 * np.median(np.abs(np.asarray(values) - median))) if values else None
        if values and (not math.isfinite(median) or not math.isfinite(scale)):
            raise DataContractError("Nonfinite fitted normalization")
        symbols[symbol] = {"count": len(values), "median": median, "scale": scale,
                           "ready": len(values) >= 2 and scale is not None and scale > 0}
    return {"algorithm": ALGORITHM, "source_mode": rows[0]["source_mode"], "symbols": symbols,
            "train_event_end": max((row["timestamp"] for row in rows), key=utc_time),
            "train_available_end": max((row["information_available_at"] for row in rows), key=utc_time),
            "train_sha256": content_hash(prepared), "train_replicate_hashes": [content_hash(replica) for replica in prepared]}


def _after_fit(rows, normalizer):
    if normalizer.get("algorithm") != ALGORITHM or rows[0]["source_mode"] != normalizer.get("source_mode"):
        raise DataContractError("Normalizer algorithm/source mismatch")
    if (min(utc_time(row["timestamp"]) for row in rows) <= utc_time(normalizer["train_event_end"]) or
            min(utc_time(row["decision_as_of"]) for row in rows) <= utc_time(normalizer["train_available_end"])):
        raise DataContractError("Scoring/calibration overlaps train event or availability boundary")


def _continuous(previous, current):
    return (all(previous[key] == current[key] for key in ("symbol", "session_id", "segment_id", "period"))
            and utc_time(current["timestamp"]) - utc_time(previous["timestamp"]) == timedelta(minutes=5))


def _statistic(values):
    total = float(sum(values))
    magnitude = float(sum(abs(value) for value in values))
    largest = max(range(len(values)), key=lambda index: abs(values[index]))
    dominance = abs(values[largest]) / magnitude if magnitude else 1.0
    coherence = abs(total) / magnitude if magnitude else 0.0
    retained = abs(total - values[largest]) / abs(total) if total else 0.0
    return {"raw_score": max(0.0, abs(total) / math.sqrt(len(values)) - PENALTY * math.log(len(values))),
            "dominance": dominance, "coherence": coherence, "retained_signed_fraction": retained,
            "admissible": dominance <= 0.60 and coherence >= 0.50 and retained >= 0.50}


def _scan(rows, normalizer):
    history, results = [], []
    for row in rows:
        if history and not _continuous(history[-1][0], row):
            history = []
        fitted = normalizer["symbols"].get(row["symbol"], {"ready": False})
        z = ((row["log_return"] - fitted["median"]) / fitted["scale"]
             if fitted["ready"] and row["log_return"] is not None else None)
        if z is not None and not math.isfinite(z):
            raise DataContractError("Nonfinite standardized residual")
        history.append((row, z))
        history = history[-max(LENGTHS):]
        as_of = utc_time(row["decision_as_of"])
        ready = z is not None and utc_time(row["information_available_at"]) <= as_of
        candidates = []
        for length in LENGTHS:
            suffix = history[-length:]
            if (len(suffix) == length and all(value is not None and
                    utc_time(item["information_available_at"]) <= as_of for item, value in suffix)):
                candidates.append({"length": length, "rows": [item for item, _ in suffix],
                                   **_statistic([value for _, value in suffix])})
        mask = ",".join(str(candidate["length"]) for candidate in candidates)
        maximum = max((candidate["raw_score"] for candidate in candidates if candidate["admissible"]), default=0.0)
        results.append({"row": row, "ready": ready, "point_score": abs(z) if ready else None,
                        "candidates": candidates, "coverage_mask": mask, "joint_score": maximum})
    return results


def calibrate_suffixes(rows, normalizer, *, normalizer_sha256, minimum_samples=100, quantile=0.98):
    """Calibrate guarded maximum over all suffix lengths, per symbol/mask."""
    return calibrate_suffix_replicates([rows], normalizer, normalizer_sha256=normalizer_sha256,
                                      minimum_samples=minimum_samples, quantile=quantile)


def calibrate_suffix_replicates(replicates, normalizer, *, normalizer_sha256, minimum_samples=100, quantile=0.98):
    """Keep suffix history separate for each independent calibration seed."""
    if content_hash(normalizer) != normalizer_sha256:
        raise DataContractError("Normalizer hash mismatch")
    if type(minimum_samples) is not int or minimum_samples < 1 or isinstance(quantile, bool) or not 0 < quantile < 1:
        raise DataContractError("Invalid calibration controls")
    prepared = [_rows(rows, "validation") for rows in replicates]
    if not prepared:
        raise DataContractError("Require calibration replicates")
    points, maxima = {}, {}
    for rows in prepared:
        _after_fit(rows, normalizer)
        for endpoint in _scan(rows, normalizer):
            if not endpoint["ready"]:
                continue
            symbol, mask = endpoint["row"]["symbol"], endpoint["coverage_mask"]
            points.setdefault(symbol, []).append(endpoint["point_score"])
            if mask:
                maxima.setdefault(symbol, {}).setdefault(mask, []).append(endpoint["joint_score"])
    rows = [row for replica in prepared for row in replica]
    return {"algorithm": ALGORITHM, "normalizer": normalizer, "normalizer_sha256": normalizer_sha256,
            "lengths": list(LENGTHS), "penalty": PENALTY, "minimum_samples": minimum_samples,
            "quantile": float(quantile), "points": {key: sorted(values) for key, values in points.items()},
            "maxima": {key: {mask: sorted(values) for mask, values in masks.items()} for key, masks in maxima.items()},
            "validation_event_end": max((row["timestamp"] for row in rows), key=utc_time),
            "validation_available_end": max((row["information_available_at"] for row in rows), key=utc_time),
            "validation_sha256": content_hash(prepared),
            "validation_replicate_hashes": [content_hash(replica) for replica in prepared]}


def _rank(value, references, minimum, quantile):
    if len(references) < minimum:
        return None, None
    rank = (bisect_left(references, value) + 0.5 * (bisect_right(references, value) - bisect_left(references, value))) / len(references)
    return float(rank), float(np.quantile(references, quantile))


def _record(rows, calibration, model_hash, model_run, emitted_at, kind, rank, reasons, *, candidate=False):
    last = rows[-1]
    data_hash = content_hash(rows)
    identifier = content_hash({"data": data_hash, "model": model_hash, "kind": kind, "as_of": last["decision_as_of"]})
    record = {"schema_version": 2, "event_unit": "bar_5m", "symbol": last["symbol"],
              "session_id": last["session_id"], "segment_id": last["segment_id"], "version": 1,
              "event_start": rows[0]["timestamp"], "event_end": last["timestamp"],
              "detected_at": emitted_at, "available_at": emitted_at, "as_of": last["decision_as_of"],
              "stage": "initial", "kind": kind, "model_run": model_run, "model_sha256": model_hash,
              "data_sha256": data_hash, "data_revision": "|".join(row["data_revision"] for row in rows),
              "source_mode": last["source_mode"], "reason_codes": reasons, "score": rank,
              "threshold": calibration["quantile"] if rank is not None else None,
              "decision_method": "heuristic", "candidate_id" if candidate else "event_id": identifier}
    return validate_candidate(record) if candidate else validate_event(record)


def score_suffixes(rows, calibration, *, expected_sha256, model_run, emitted_at, latest_only=False):
    """Return append-only endpoint snapshots; callers must retain old emissions."""
    if content_hash(calibration) != expected_sha256:
        raise DataContractError("Calibration hash mismatch")
    normalizer = calibration["normalizer"]
    if (content_hash(normalizer) != calibration["normalizer_sha256"] or
            calibration["algorithm"] != ALGORITHM or calibration["lengths"] != list(LENGTHS)
            or calibration["penalty"] != PENALTY):
        raise DataContractError("Frozen detector binding mismatch")
    rows = _rows(rows, "evaluation")
    _after_fit(rows, normalizer)
    if (min(utc_time(row["timestamp"]) for row in rows) <= utc_time(calibration["validation_event_end"]) or
            min(utc_time(row["decision_as_of"]) for row in rows) <= utc_time(calibration["validation_available_end"])):
        raise DataContractError("Evaluation overlaps validation or late reference availability")
    if utc_time(emitted_at, "emitted_at") < max(utc_time(row["decision_as_of"]) for row in rows):
        raise DataContractError("Emission precedes evidence as-of")
    if type(latest_only) is not bool:
        raise DataContractError("latest_only must be boolean")
    output = []
    endpoints = _scan(rows, normalizer)
    for endpoint in endpoints[-1:] if latest_only else endpoints:
        row, symbol, mask = endpoint["row"], endpoint["row"]["symbol"], endpoint["coverage_mask"]
        minimum, q = calibration["minimum_samples"], calibration["quantile"]
        point_rank, point_threshold = _rank(endpoint["point_score"], calibration["points"].get(symbol, []), minimum, q) if endpoint["ready"] else (None, None)
        references = calibration["maxima"].get(symbol, {}).get(mask, [])
        segment_rank, segment_threshold = _rank(endpoint["joint_score"], references, minimum, q) if mask else (None, None)
        # lean-ctx: one bounded suffix per length; event dedup belongs to U07.
        candidates = []
        if point_rank is not None:
            candidates.append({"record": _record([row], calibration, expected_sha256, model_run, emitted_at,
                                                  "point", point_rank, ["point_residual_rank"], candidate=True),
                               "raw_score": endpoint["point_score"], "admissible": True})
        for candidate in endpoint["candidates"]:
            diagnostic = {key: value for key, value in candidate.items() if key != "rows"}
            rank, _ = _rank(candidate["raw_score"], references, minimum, q)
            diagnostic["rank"] = rank
            reasons = ["joint_suffix_rank", "estimated_current_suffix",
                       "influence_guard_pass" if candidate["admissible"] else "influence_guard_reject"]
            if rank is None:
                reasons.append("insufficient_joint_reference")
            diagnostic.update({"event_start": candidate["rows"][0]["timestamp"],
                               "event_end": row["timestamp"], "as_of": row["decision_as_of"],
                               "data_sha256": content_hash(candidate["rows"]), "reason_codes": reasons,
                               "boundary_status": "estimated_current_suffix"})
            diagnostic["record"] = (_record(candidate["rows"], calibration, expected_sha256, model_run,
                                             emitted_at, "collective", rank, reasons,
                                             candidate=True) if rank is not None else None)
            candidates.append(diagnostic)
        admissible = [candidate for candidate in endpoint["candidates"] if candidate["admissible"]]
        best = min(admissible, key=lambda candidate: (-candidate["raw_score"], candidate["length"])) if admissible else None
        selected_rows, kind, selected_rank = [row], "normal", None
        if point_rank is None or segment_rank is None:
            kind, reasons = "insufficient_data", ["missing_scale_history_or_reference"]
        elif best is not None and segment_rank > q and segment_rank >= point_rank:
            kind, selected_rank, selected_rows = "collective", segment_rank, best["rows"]
            reasons = ["joint_suffix_rank", "influence_guard_pass", "estimated_current_suffix"]
        elif point_rank > q:
            kind, selected_rank, reasons = "point", point_rank, ["point_residual_rank"]
        else:
            selected_rank, reasons = max(point_rank, segment_rank), ["no_rank_exceeds_threshold"]
        output.append({"event": _record(selected_rows, calibration, expected_sha256, model_run, emitted_at,
                                         kind, selected_rank, reasons), "candidates": candidates,
                       "coverage_mask": mask, "candidate_lengths_considered": len(endpoint["candidates"]),
                       "point_raw_score": endpoint["point_score"], "point_rank": point_rank,
                       "joint_raw_score": endpoint["joint_score"], "joint_rank": segment_rank,
                       "point_reference_count": len(calibration["points"].get(symbol, [])),
                       "joint_reference_count": len(references), "point_raw_threshold": point_threshold,
                       "joint_raw_threshold": segment_threshold, "boundary_status": "estimated_current_suffix"})
    return output
