"""Honest denominators and deterministic one-to-one bar-interval matching."""
from collections import defaultdict
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from stock_monitor.io import DataContractError
from .contracts import utc_time


def counts(tp, fp, fn):
    if any(type(value) is not int or value < 0 for value in (tp, fp, fn)):
        raise DataContractError("Counts must be nonnegative integers")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0) \
        if precision is not None and recall is not None else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def point_metrics(predicted, truth):
    predicted, truth = list(predicted), list(truth)
    if len(predicted) != len(truth) or any(type(x) is not bool for x in predicted + truth):
        raise DataContractError("Aligned explicit binary labels required")
    return {**counts(sum(p and t for p, t in zip(predicted, truth)),
                     sum(p and not t for p, t in zip(predicted, truth)),
                     sum(not p and t for p, t in zip(predicted, truth))),
            "eligible": len(truth), "positives": sum(truth)}


def _interval(row, identifier):
    start, end = utc_time(row["event_start"]), utc_time(row["event_end"])
    if start > end or any(stamp.second or stamp.microsecond or stamp.minute % 5 for stamp in (start, end)):
        raise DataContractError("Expected ordered five-minute bar-end interval")
    if not isinstance(row.get(identifier), str) or not row[identifier]:
        raise DataContractError("Stable interval ID required")
    return int(start.timestamp() // 300), int(end.timestamp() // 300) + 1


def interval_iou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / (a[1] - a[0] + b[1] - b[0] - intersection)


def match_segments(predictions, truth, *, minimum_iou=.3):
    if isinstance(minimum_iou, bool) or not math.isfinite(minimum_iou) or not 0 < minimum_iou <= 1:
        raise DataContractError("Invalid IoU threshold")
    predictions, truth = list(predictions), list(truth)
    groups = defaultdict(lambda: [[], []])
    for values, id_field, side in [(predictions, "event_id", 0), (truth, "truth_id", 1)]:
        ids = set()
        for row in values:
            interval = _interval(row, id_field)
            if row[id_field] in ids:
                raise DataContractError("Duplicate interval ID")
            ids.add(row[id_field])
            groups[(row["symbol"], row["session_id"])][side].append((row, interval))
    matches, unmatched_predictions, misses = [], [], []
    for key in sorted(groups):
        p, t = groups[key]
        p.sort(key=lambda row: row[0]["event_id"])
        t.sort(key=lambda row: row[0]["truth_id"])
        paired_p, paired_t = set(), set()
        if p and t:
            overlaps = np.array([[interval_iou(a[1], b[1]) for b in t] for a in p])
            # Cardinality dominates every possible sum-IoU gain. Sorted stable IDs
            # and deterministic scipy assignment resolve exact ties reproducibly.
            weights = np.where(overlaps >= minimum_iou, min(len(p), len(t)) + 1 + overlaps, 0.)
            rows, columns = linear_sum_assignment(weights, maximize=True)
            for i, j in zip(rows.tolist(), columns.tolist()):
                if not weights[i, j]:
                    continue
                paired_p.add(i)
                paired_t.add(j)
                matches.append({"event_id": p[i][0]["event_id"], "truth_id": t[j][0]["truth_id"],
                    "symbol": key[0], "session_id": key[1], "iou": float(overlaps[i, j]),
                    "start_error_bars": p[i][1][0] - t[j][1][0], "end_error_bars": p[i][1][1] - t[j][1][1],
                    "predicted_length": p[i][1][1] - p[i][1][0], "truth_length": t[j][1][1] - t[j][1][0]})
        unmatched_predictions += [row[0]["event_id"] for i, row in enumerate(p) if i not in paired_p]
        misses += [row[0]["truth_id"] for j, row in enumerate(t) if j not in paired_t]
    return {**counts(len(matches), len(unmatched_predictions), len(misses)),
            "minimum_iou": minimum_iou, "matches": matches, "unmatched_predictions": unmatched_predictions,
            "misses": misses, "point_adjusted": False}
