"""Five-symbol development data contract; no provider, fit or live claims."""
from collections import Counter
from copy import deepcopy

import pandas as pd

from stock_monitor.features import compute_features
from stock_monitor.io import DataContractError
from stock_monitor.research.density_samples import AVAILABILITY, _assemble_density_features
from stock_monitor.research.protocol import blind_bars, digest, validate_split_registry
from stock_monitor.validation import clock_minutes

SYMBOLS = ('VNM', 'ACB', 'TCB', 'MBB', 'SSI')
EXPOSED_DAY = '2026-10-01'


def build_protocol(calendar, previous_splits):
    """Reuse temporal boundaries, not the old symbols' fitted artifacts."""
    splits = validate_split_registry(previous_splits, explored_dates=[])
    dates = calendar['dates']
    if (not dates or dates != sorted(set(dates)) or calendar['timezone'] != 'Asia/Ho_Chi_Minh' or
            calendar['status'] != 'observed_only' or calendar['complete'] is not False or
            dates[0] != '2026-04-01' or dates[-1] != EXPOSED_DAY or
            set(sum(splits.values(), [])) - set(dates) or
            splits['train'][0] != '2026-05-04' or splits['train'][-1] != '2026-07-29' or
            splits['validation'][0] != '2026-07-30' or splits['validation'][-1] != '2026-08-27'):
        raise DataContractError('Expected pinned observed calendar and existing development boundaries')
    warmup = [d for d in dates if d < splits['train'][0]]
    if len(warmup) != 20:
        raise DataContractError('Require twenty registered reference sessions')
    return {'schema_version': 1, 'protocol_id': 'watchlist_history_development_20261013',
        'symbols': list(SYMBOLS), 'source': 'VCI', 'source_interval': '1m', 'interval': '5m',
        'requested_start': warmup[0], 'requested_end': splits['validation'][-1],
        'dates': {'reference': warmup, 'train': deepcopy(splits['train']),
                  'validation': deepcopy(splits['validation'])},
        'exposed_demo_sessions': [EXPOSED_DAY],
        'independent_test': {'status': 'pending_separate_future_registration', 'dates': [],
            'must_follow': EXPOSED_DAY, 'requires': ['observed session calendar', 'unexposed dataset pins',
                'frozen model and thresholds before opening scores', 'blind reviewer labels for effectiveness']},
        'acquisition_budget': {'maximum_requests': 5, 'requests_per_symbol': 1, 'max_attempts': 1,
            'timeout_seconds': 10, 'retry_wait_seconds': 0, 'wall_seconds': 120,
            'count_back': 33916, 'maximum_source_rows': 169580,
            'maximum_response_bytes_per_request': 10485760, 'maximum_total_response_bytes': 52428800,
            'stop_on_access_quota_timeout_or_invalid_payload': True},
        'coverage_floors': {'common_usable_sessions': 80, 'reference_sessions_per_symbol': 20,
            'train_sessions_per_symbol': 50, 'validation_sessions_per_symbol': 15,
            'minimum_session_bar_fraction': .8, 'train_samples_per_symbol': 100,
            'validation_samples_per_symbol': 100},
        'features': {'history_sessions': 20, 'min_history_sessions': 10, 'warmup_sessions': 20,
            'conditioning_bars': 6, 'segment_and_session_contiguous': True,
            'availability_basis': AVAILABILITY},
        'fit_budget': {'maximum_fits_in_this_registration': 0,
            'next_gate': 'New five-symbol model/scaler/calibration and finite CPU fit registration after actual coverage audit'},
        'quality': {'missing_bars': 'do_not_fill', 'incomplete_minutes': 'reject_and_break_segment',
            'source_rights_status': 'pending', 'distributable': False, 'live_certified': False,
            'profile_usage': 'Pinned research assumptions only; no source confirmation promoted'},
        'selection': {'scaler_fit': 'train only', 'checkpoint_selection': 'validation only',
            'threshold': 'per-symbol validation quantile .98, minimum100 references; not alert accuracy',
            'old_three_symbol_model_reuse': False, 'effectiveness_status': 'unavailable'}}


def validate_protocol(protocol, calendar, previous_splits):
    # Exact comparison also rejects added keys, wider budgets and hidden fit permission.
    if digest(protocol) != digest(build_protocol(calendar, previous_splits)):
        raise DataContractError('Registered watchlist protocol differs from frozen scope or budget')
    return protocol


def build_development_samples(bars, protocol, calendar, previous_splits, *, as_of):
    """Separate symbol adapter using existing causal math; no old model loader."""
    validate_protocol(protocol, calendar, previous_splits)
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None or cutoff.utcoffset().total_seconds() != 0:
        raise DataContractError('Development as_of must be explicit UTC')
    times = pd.to_datetime(bars.timestamp, utc=False)
    if str(times.dt.tz) != 'UTC' or times.isna().any():
        raise DataContractError('Development bar timestamps must be explicit UTC')
    allowed = set(sum(protocol['dates'].values(), []))
    # Future/exposed rows are excluded BEFORE OHLCV inspection or feature calculation.
    scoped = bars.loc[times.le(cutoff) & bars.session_id.isin(allowed)].copy()
    date_split = {d: split for split in ('train', 'validation') for d in protocol['dates'][split]}
    if scoped.empty:
        return {'rows': [], 'samples': 0, 'bars_known': 0, 'fits': 0,
                'availability_basis': AVAILABILITY, 'exclusion_counts': {'no_development_bars': len(bars)}}
    blind_bars(scoped)
    if (set(scoped.symbol) - set(SYMBOLS) or not scoped.source.eq('VCI').all() or
            not scoped.interval.eq('5m').all() or
            {'available_at', 'received_at', 'decision_as_of'} & set(scoped.columns)):
        raise DataContractError('Require historical checked bars for registered new symbols only')
    geometry = protocol['features']
    features = compute_features(scoped, calendar, history_sessions=geometry['history_sessions'],
        min_history_sessions=geometry['min_history_sessions'], warmup_sessions=geometry['warmup_sessions']).frame
    return _assemble_density_features(features, date_split, bars_known=len(scoped),
        history_length=geometry['conditioning_bars'], availability_basis=AVAILABILITY)


def audit_coverage(bars, protocol, calendar, previous_splits):
    validate_protocol(protocol, calendar, previous_splits)
    checked = blind_bars(bars)
    if set(checked.symbol) - set(SYMBOLS) or set(checked.session_id) - set(calendar['dates']):
        raise DataContractError('Coverage contains an unregistered symbol or calendar date')
    cutoff = (pd.Timestamp(protocol['requested_end'], tz=calendar['timezone']) + pd.Timedelta(hours=16)).tz_convert('UTC')
    samples = build_development_samples(bars, protocol, calendar, previous_splits, as_of=cutoff)
    floors = protocol['coverage_floors']
    by_symbol, usable_sets = {}, []
    sample_counts = Counter((r['symbol'], r['split']) for r in samples['rows'])
    for symbol in SYMBOLS:
        rows = checked.loc[checked.symbol.eq(symbol)]
        usable = set()
        for day, group in rows.groupby('session_id'):
            sessions = calendar.get('overrides', {}).get(day, calendar['sessions'])
            expected = sum((clock_minutes(s['end']) - clock_minutes(s['start'])) // 5 for s in sessions)
            if expected <= 0 or len(group) > expected:
                raise DataContractError('Unexpected per-session bar count')
            if len(group) / expected >= floors['minimum_session_bar_fraction']:
                usable.add(day)
        roles = {role: len(usable & set(days)) for role, days in protocol['dates'].items()}
        eligible = {role: sample_counts[symbol, role] for role in ('train', 'validation')}
        met = (roles['reference'] >= floors['reference_sessions_per_symbol'] and
            roles['train'] >= floors['train_sessions_per_symbol'] and
            roles['validation'] >= floors['validation_sessions_per_symbol'] and
            all(eligible[role] >= floors[role + '_samples_per_symbol'] for role in eligible))
        by_symbol[symbol] = {'observed_bars': len(rows), 'observed_sessions': rows.session_id.nunique(),
            'usable_development_sessions': roles, 'eligible_development_samples': eligible,
            'coverage_floors_met': met}
        usable_sets.append(usable & set(sum(protocol['dates'].values(), [])))
    common = len(set.intersection(*usable_sets))
    return {'by_symbol': by_symbol, 'common_usable_development_sessions': common,
        'coverage_floors_met': common >= floors['common_usable_sessions'] and
            all(v['coverage_floors_met'] for v in by_symbol.values()),
        'excluded_sessions': sorted(set(checked.session_id) - set(sum(protocol['dates'].values(), []))),
        'samples': samples['samples'], 'model_fits': 0, 'training_authorized': False,
        'independent_test_available': False, 'effectiveness_status': 'unavailable', 'live_certified': False}
