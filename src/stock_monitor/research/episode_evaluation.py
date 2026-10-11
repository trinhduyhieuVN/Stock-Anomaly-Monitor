"""Registered exact point / IoU segment metrics; require explicit complete review."""
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

from stock_monitor.io import DataContractError
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.metrics import match_segments, counts


def _keys(record):
    start, end = utc_time(record['event_start']), utc_time(record['event_end'])
    if start > end or (end - start).total_seconds() % 300:
        raise DataContractError('Invalid review interval')
    if record['kind'] not in {'point', 'collective'} or (record['kind'] == 'point' and start != end):
        raise DataContractError('Invalid review unit')
    if any(t.minute % 5 or t.second or t.microsecond for t in (start, end)):
        raise DataContractError('Expected closed 5m boundaries')
    if any(t.astimezone(ZoneInfo('Asia/Ho_Chi_Minh')).date().isoformat() != record['session_id'] for t in (start, end)):
        raise DataContractError('Cross-session review interval')
    return {(record['symbol'], int(start.timestamp()) + 300 * i) for i in range(int((end - start).total_seconds() // 300) + 1)}


def evaluate_episodes(predictions, labels, *, reviewed_keys, eligible_keys, complete_sessions):
    """No synthetic labels generated here. Callers must bind reviewed packet IDs.

    Ordinary coverage requires explicit complete sessions. Uncertain/data_issue
    intervals are excluded, counted, and never relabeled ordinary. Positive
    intervals not fully score-eligible still contribute FN in all-reviewed scope.
    """
    reviewed, eligible = set(reviewed_keys), set(eligible_keys)
    complete = set(complete_sessions)
    if not reviewed or not complete or any(r['session_id'] not in complete for r in predictions + labels):
        raise DataContractError('Explicit nonempty fully-reviewed sessions required; never infer recall')
    if not eligible.issubset(reviewed): raise DataContractError('Eligible denominator outside reviewed scope')
    grey, positives, seen = set(), [], set()
    label_counts = defaultdict(int)
    for row in labels:
        if row['label'] not in {'noteworthy', 'ordinary', 'uncertain', 'data_issue'}:
            raise DataContractError('Unknown adjudicated label')
        if row['reviewer_kind'] not in {'human', 'domain_expert', 'ai_assisted'}:
            raise DataContractError('Missing actual reviewer provenance')
        if row['truth_id'] in seen: raise DataContractError('Duplicate review identity')
        seen.add(row['truth_id']); label_counts[row['label']] += 1
        interval = _keys(row)
        if not interval.issubset(reviewed): raise DataContractError('Label outside reviewed scope')
        if row['label'] in {'uncertain', 'data_issue'}: grey.update(interval)
        elif row['label'] == 'noteworthy': positives.append(row)
    if any(_keys(r) & grey for r in positives): raise DataContractError('Conflicting positive/uncertain review: adjudicate first')
    kept, excluded = [], []
    ids = set()
    for row in predictions:
        if row['event_id'] in ids: raise DataContractError('Duplicate prediction identity')
        ids.add(row['event_id'])
        if not _keys(row).issubset(reviewed): raise DataContractError('Prediction outside reviewed scope')
        (excluded if _keys(row) & grey else kept).append(row)
    def match(pred, truth):
        groups = defaultdict(lambda: [[], []])
        for values, side in ((pred, 0), (truth, 1)):
            for r in values: groups[(r['symbol'], r['session_id'], r['segment_id'], r['kind'])][side].append(r)
        results = []
        for scope in sorted(groups):
            p, t = groups[scope]
            # Existing matcher: maximum-cardinality then IoU. Timestamp-prefixed
            # stable IDs satisfy registered chronological tie resolution.
            def ordered(row, field): return {**row, field: row['event_start'] + '|' + row[field]}
            results.append(match_segments([ordered(r, 'event_id') for r in p], [ordered(r, 'truth_id') for r in t],
                                          minimum_iou=1. if scope[-1] == 'point' else .5))
        return {**counts(sum(r['tp'] for r in results), sum(r['fp'] for r in results), sum(r['fn'] for r in results)),
                'matching_groups': results}
    common_truth = [r for r in positives if _keys(r).issubset(eligible)]
    common_pred = [r for r in kept if _keys(r).issubset(eligible)]
    return {'all_reviewed': match(kept, positives), 'common_eligible': match(common_pred, common_truth),
        'coverage': len(eligible - grey) / len(reviewed - grey) if reviewed - grey else None,
        'reviewed_bars': len(reviewed), 'eligible_bars': len(eligible), 'uncertain_or_issue_bars': len(grey),
        'excluded_predictions': [r['event_id'] for r in excluded], 'label_counts': dict(label_counts),
        'reviewer_kind_counts': {k: sum(r['reviewer_kind'] == k for r in labels) for k in ('human', 'domain_expert', 'ai_assisted')},
        'positives_without_full_score_coverage': len(positives) - len(common_truth)}


def paired_precision_interval(by_session, *, replicates=1000, seed=42):
    if replicates != 1000 or seed != 42 or not by_session:
        raise DataContractError('Registered paired bootstrap requires 1000, seed42 and sessions')
    if len(by_session) < 2:
        return {'interval': None, 'undefined_replicates': 1000, 'reason': 'fewer_than_two_session_clusters'}
    array = np.asarray(by_session)
    if array.shape != (len(by_session), 2, 2) or (array < 0).any() or not np.isfinite(array).all() or (array % 1).any():
        raise DataContractError('Require paired per-session [new/comparator][TP,FP] counts')
    rng = np.random.default_rng(seed); values, undefined = [], 0
    for _ in range(replicates):
        totals = array[rng.integers(0, len(array), size=len(array))].sum(axis=0)
        denominators = totals.sum(axis=1)
        if (denominators == 0).any(): undefined += 1
        else: values.append(totals[0, 0] / denominators[0] - totals[1, 0] / denominators[1])
    return {'interval': np.quantile(values, [.025, .975]).tolist() if values else None,
            'undefined_replicates': undefined, 'defined_replicates': len(values),
            'session_clusters': len(array), 'replicates': replicates, 'seed': seed}
