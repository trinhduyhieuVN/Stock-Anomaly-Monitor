"""Frozen conditional residual costs and causal episode selection, not posteriors."""
from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd

from stock_monitor.io import DataContractError
from stock_monitor.research.density_samples import AVAILABILITY
from stock_monitor.research.protocol import digest

ALGORITHM = 'conditional_student_t_cost_selector_v1'


def validate_design(design):
    # This algorithm version fixes the formula; edits require a new registered version.
    required = {'algorithm': ALGORITHM, 'point_scale': 10., 'suffix_lengths': [3, 6, 12],
                'shift_grid': {'minimum': -8., 'maximum': 8., 'step': .5},
                'largest_positive_contribution_limit': .6,
                'minimum_gain_fraction_after_largest_removed': .5}
    if any(design.get(k) != v for k, v in required.items()):
        raise DataContractError('Unregistered selector formula')
    if design.get('episodes', {}).get('maximum_new_per_symbol_session') != 3:
        raise DataContractError('Unregistered episode budget')
    cal = design.get('calibration', {})
    if cal.get('quantile') != .98 or cal.get('minimum_samples_per_symbol') != 100:
        raise DataContractError('Unregistered selector calibration')


def loss(residual):
    return 3. * np.log1p(np.square(residual) / 5.).sum(axis=-1)


def compare_costs(residuals, design, *, segments=True):
    """Every cost explains the same region; alternative suffix leaves outside H0.

    H1 uses a proper broad Student-t (scale10) at one position. H2 fits a
    shared four-dimensional residual location on a fixed finite grid. Costs
    condition on frozen network parameters; temporal independence is an
    approximation, not evidence of a real generative market model.
    """
    validate_design(design)
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 4 or not 1 <= len(r) <= 12 or not np.isfinite(r).all():
        raise DataContractError('Expected 1..12 finite four-feature residuals')
    if np.abs(r).max() > 1e100:
        raise DataContractError('Residual outside safe numerical range')
    base = loss(r)
    h0 = float(base.sum())
    broad = loss(r / 10.) + 4. * math.log(10.)
    point_costs = h0 - base + broad + math.log(len(r)) + math.log(2.)
    point = int(np.argmin(point_costs))
    h1 = float(point_costs[point])
    grid = np.arange(-8., 8.25, .5)
    candidates = []
    if segments:
        for length in (3, 6, 12):
            if len(r) < length:
                continue
            suffix = r[-length:]
            # N x grid x feature; independently minimize each location parameter.
            per_feature = (3. * np.log1p((suffix[:, None, :] - grid[None, :, None]) ** 2 / 5.)).sum(axis=0)
            shift = grid[np.argmin(per_feature, axis=0)]
            shifted = loss(suffix - shift)
            contributions = base[-length:] - shifted
            positive = np.maximum(contributions, 0.)
            raw_gain = float(contributions.sum())
            largest = int(np.argmax(positive))
            dominance = float(positive[largest] / positive.sum()) if positive.sum() else 1.
            reduced_gain = raw_gain - float(contributions[largest])
            retained = reduced_gain / raw_gain if raw_gain > 0 else 0.
            penalty = 2. * math.log(length) + math.log(3.)
            cost = h0 - raw_gain + penalty
            candidates.append({'length': length, 'cost': cost, 'shift': shift.tolist(),
                               'raw_gain': raw_gain, 'penalty': penalty, 'dominance': dominance,
                               'gain_after_largest_removed': reduced_gain,
                               'retained_gain_fraction': retained,
                               'admissible': raw_gain > 0 and dominance <= .6 and retained >= .5})
    admissible = [v for v in candidates if v['admissible']]
    best = min(admissible, key=lambda v: (v['cost'], v['length'])) if admissible else None
    costs = [h0, h1, best['cost'] if best else math.inf]
    winner = int(np.argmin(costs))
    return {'h0_cost': h0, 'h1_cost': h1, 'h2_cost': best['cost'] if best else None,
            'point_index': point, 'segment_length': best['length'] if best else None,
            'classification': ('normal', 'point', 'collective')[winner],
            'joint_gain': max(0., h0 - min(costs)), 'candidates': candidates,
            'coverage_mask': ','.join(str(v['length']) for v in candidates) or 'point_only'}


def _stamp(raw):
    try:
        value = pd.Timestamp(raw)
        if pd.isna(value) or value.tzinfo is None or value.utcoffset().total_seconds() != 0:
            raise ValueError('Not UTC')
        return value
    except (TypeError, ValueError) as exc:
        raise DataContractError('Selector time must be explicit UTC') from exc


def scan_residuals(rows, residuals, design, *, as_of, revision, segments=True):
    """Rebuilding a prefix is the restart protocol: identical IDs, no hidden state."""
    validate_design(design)
    cutoff = _stamp(as_of)
    residuals = np.asarray(residuals)
    if residuals.shape != (len(rows), 4) or not isinstance(revision, str) or not revision:
        raise DataContractError('Missing residual provenance/count')
    known = [(dict(row), residuals[i]) for i, row in enumerate(rows) if _stamp(row['target_time']) <= cutoff]
    known.sort(key=lambda pair: (pair[0]['symbol'], _stamp(pair[0]['target_time'])))
    seen, history, output = set(), [], []
    for row, residual in known:
        t = _stamp(row['target_time'])
        if (row['symbol'] not in {'FPT', 'HPG', 'VCB'} or row['availability_basis'] != AVAILABILITY or
                _stamp(row['as_of']) != t or not _stamp(row['input_start']) <= _stamp(row['input_end']) < t or
                t.tz_convert('Asia/Ho_Chi_Minh').date().isoformat() != row['session_id']):
            raise DataContractError('Invalid residual availability/conditioning scope')
        key = (row['symbol'], t)
        if key in seen:
            raise DataContractError('Duplicate residual revision; materialize one view first')
        seen.add(key)
        if history:
            previous = history[-1][0]
            continuous = (all(previous[k] == row[k] for k in ('symbol', 'session_id', 'segment_id', 'split')) and
                          t - _stamp(previous['target_time']) == pd.Timedelta(minutes=5))
            if not continuous:
                history = []
        history = (history + [(row, residual)])[-12:]
        result = compare_costs([v for _, v in history], design, segments=segments)
        point = history[result['point_index']][0]['target_time']
        start = history[-result['segment_length']][0]['target_time'] if result['classification'] == 'collective' else point
        end = row['target_time'] if result['classification'] == 'collective' else point
        result.update({k: row[k] for k in ('symbol', 'session_id', 'segment_id', 'split', 'target_time', 'availability_basis')})
        result.update({'region_start': history[0][0]['target_time'], 'region_end': row['target_time'],
                       'region_count': len(history), 'start': start, 'end': end,
                       'decision_as_of': row['target_time'], 'revision': revision, 'algorithm': ALGORITHM,
                       'source_mode': 'HISTORICAL_REPLAY', 'decision_method': 'conditional_penalized_likelihood',
                       'segments_enabled': segments})
        result['emission_id'] = digest(result)
        output.append(result)
    return output


def calibrate(emissions, design):
    validate_design(design)
    if not emissions or any(v['split'] != 'validation' for v in emissions):
        raise DataContractError('Calibration requires validation emissions only')
    grouped = defaultdict(list)
    for row in emissions:
        grouped[row['symbol']].append(row)
    references = {}
    for symbol, group in grouped.items():
        if len(group) < 100:
            raise DataContractError('Too few validation selector references')
        references[symbol] = {'threshold': float(np.quantile([v['joint_gain'] for v in group], .98)),
                              'count': len(group), 'coverage_masks': dict(Counter(v['coverage_mask'] for v in group)),
                              'scores': sorted(v['joint_gain'] for v in group)}
    return {'algorithm': ALGORITHM, 'design_sha256': digest(design), 'symbols': references,
            'segments_enabled': emissions[0]['segments_enabled'],
            'validation_end': max(v['target_time'] for v in emissions),
            'emissions_sha256': digest(emissions), 'effectiveness_status': 'unavailable'}


def select_episodes(emissions, calibration, design):
    """All emissions retained; cap new episodes chronologically, never daily top-k."""
    validate_design(design)
    if calibration['algorithm'] != ALGORITHM or calibration['design_sha256'] != digest(design):
        raise DataContractError('Selector calibration/design mismatch')
    if len({v['emission_id'] for v in emissions}) != len(emissions):
        raise DataContractError('Duplicate emission')
    episodes, output, counts = [], [], Counter()
    for row in sorted(emissions, key=lambda v: (v['symbol'], _stamp(v['target_time']))):
        if row['segments_enabled'] != calibration['segments_enabled']:
            raise DataContractError('Segment ablation calibration mismatch')
        reference = calibration['symbols'].get(row['symbol'])
        ready = reference is not None and row['coverage_mask'] in reference['coverage_masks']
        result = {**row, 'threshold': reference['threshold'] if ready else None,
                  'reference_count': reference['count'] if ready else 0,
                  'state': 'normal' if ready else 'insufficient_reference', 'episode_id': None}
        if ready:
            values = reference['scores']
            result['empirical_rank'] = float((np.searchsorted(values, row['joint_gain'], 'left') +
                                               np.searchsorted(values, row['joint_gain'], 'right')) / (2 * len(values)))
        if ready and row['classification'] != 'normal' and row['joint_gain'] > reference['threshold']:
            scope = (row['symbol'], row['session_id'])
            overlapping = [e for e in episodes if (e['symbol'], e['session_id']) == scope and
                           e['segment_id'] == row['segment_id'] and
                           _stamp(row['start']) <= _stamp(e['end']) and _stamp(row['end']) >= _stamp(e['start'])]
            episode = overlapping[-1] if overlapping else None
            if episode is None and counts[scope] < 3:
                episode = {'episode_id': digest({'initial_emission_id': row['emission_id']}),
                           'symbol': row['symbol'], 'session_id': row['session_id'], 'segment_id': row['segment_id'],
                           'start': row['start'], 'end': row['end'], 'first_detection': row['decision_as_of'],
                           'initial_emission': dict(row), 'emission_ids': []}
                episodes.append(episode)
                counts[scope] += 1
            if episode is None:
                result['state'] = 'budget_suppressed'
            else:
                result['state'] = 'alert'
                result['episode_id'] = episode['episode_id']
                episode['start'] = min(episode['start'], row['start'], key=_stamp)
                episode['end'] = max(episode['end'], row['end'], key=_stamp)
                episode['emission_ids'].append(row['emission_id'])
        output.append(result)
    return {'emissions': output, 'episodes': episodes, 'state_counts': dict(Counter(v['state'] for v in output))}
