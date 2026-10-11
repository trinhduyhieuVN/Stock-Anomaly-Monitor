"""Purged historical-bar forecasts; no synthetic or live certification crossover."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
import math

import numpy as np
import pandas as pd

from stock_monitor.io import DataContractError
from stock_monitor.research.density_samples import AVAILABILITY
from stock_monitor.research.protocol import digest, blind_bars
from stock_monitor.upgrade.contracts import LOCAL_ZONE, utc_time, _number, _hash
from stock_monitor.upgrade.forecast import FORECAST_FIELDS


def build_real_forecast_samples(rows, arrays, bars):
    """Reuse registered six-return Ridge inputs; purge the complete h3 path."""
    blind_bars(bars)
    prices = {(r.symbol, r.timestamp.isoformat()): r for r in bars.itertuples()}
    samples, purged = [], Counter()
    for i, row in enumerate(rows):
        if row['split'] not in {'train', 'validation'}:
            continue
        if row['availability_basis'] != AVAILABILITY:
            raise DataContractError('Historical forecast fit requires explicit bar-end assumption')
        origin_time = pd.Timestamp(row['target_time'])
        origin = prices.get((row['symbol'], origin_time.isoformat()))
        if origin is None or origin.segment_id != row['segment_id'] or origin.source != 'VCI':
            raise DataContractError('Forecast origin binding differs from density sample')
        path = [prices.get((row['symbol'], (origin_time + pd.Timedelta(minutes=5 * h)).isoformat())) for h in (1, 2, 3)]
        if any(v is None or v.session_id != origin.session_id or v.segment_id != origin.segment_id or
               v.quality_status != 'ok' or v.source != 'VCI' for v in path):
            purged['cross_gap_session_period_or_split'] += 1
            continue
        features = arrays['X'][i, -5:, 0].tolist() + [float(arrays['y'][i, 0])]
        ends = [path[0].timestamp.isoformat(), path[2].timestamp.isoformat()]
        samples.append({'sample_id': digest({'symbol': row['symbol'], 'time': row['target_time']}),
            'symbol': row['symbol'], 'session_id': row['session_id'], 'segment_id': row['segment_id'],
            'origin_bar_end': row['target_time'], 'split': row['split'], 'feature_end': row['target_time'],
            'input_available_at': row['target_time'], 'features': features,
            'targets_h1_h3': [math.log(float(v.close) / float(origin.close)) for v in (path[0], path[2])],
            'target_bar_ends': ends, 'outcome_available_h1_h3': ends,
            'input_sha256': digest({'row': row, 'features': features}), 'source_mode': 'HISTORICAL_REPLAY',
            'availability_basis': AVAILABILITY})
    return {'samples': samples, 'purged': dict(purged)}


def period_end(origin, calendar):
    if calendar.get('timezone') != 'Asia/Ho_Chi_Minh' or origin['session_id'] not in calendar['dates']:
        raise DataContractError('Forecast date outside frozen observed calendar')
    stamp = utc_time(origin['timestamp'])
    day = stamp.astimezone(LOCAL_ZONE).date()
    for p in calendar['sessions']:
        start = datetime.combine(day, time.fromisoformat(p['start']), LOCAL_ZONE)
        end = datetime.combine(day, time.fromisoformat(p['end']), LOCAL_ZONE)
        if start < stamp.astimezone(LOCAL_ZONE) <= end:
            return p['id'], end.astimezone(timezone.utc).isoformat()
    raise DataContractError('Forecast origin outside closed continuous session')


def validate_real_forecast(record):
    fields = FORECAST_FIELDS | {'algorithm', 'availability_basis'}
    if (set(record) != fields or record['schema_version'] != 3 or record['source_mode'] != 'HISTORICAL_REPLAY' or
            record['clock_basis'] != 'historical_bar_end_assumption' or record['availability_basis'] != AVAILABILITY or
            record['algorithm'] not in {'ridge_h1_h3', 'zero_return'}):
        raise DataContractError('Expected historical forecast contract3, not LIVE')
    h = record['horizon_bars']
    if type(h) is not int or h not in {1, 3}:
        raise DataContractError('Unknown forecast horizon')
    origin, target, created, cutoff, available, end, origin_available = (utc_time(record[k]) for k in
        ('origin_bar_end', 'target_bar_end', 'created_at', 'as_of', 'available_at', 'period_end', 'origin_available_at'))
    if (not origin == cutoff == created == available == origin_available or
            target != origin + timedelta(minutes=5 * h) or origin > end or
            record['session_id'] != origin.astimezone(LOCAL_ZONE).date().isoformat() or
            origin.minute % 5 or origin.second or origin.microsecond):
        raise DataContractError('Forecast must be created at origin under declared historical assumption')
    if record['symbol'] not in {'FPT', 'HPG', 'VCB'} or not record['segment_id'] or not record['period']:
        raise DataContractError('Invalid forecast identity')
    if record['origin_revision'] != 0 or type(record['origin_revision']) is not int:
        raise DataContractError('Historical frozen origin revision required')
    _number(record['prediction'], 'prediction'); _number(record['origin_close'], 'origin_close')
    if record['origin_close'] <= 0 or (record['algorithm'] == 'zero_return' and record['prediction'] != 0):
        raise DataContractError('Invalid origin/prediction')
    for k in ('model_sha256', 'input_sha256', 'calendar_sha256', 'forecast_id'): _hash(record[k], k)
    expected = 'censored_boundary' if target > end else 'awaiting_outcome'
    if record['horizon_status'] != expected or record['forecast_id'] != digest({k: v for k, v in record.items() if k != 'forecast_id'}):
        raise DataContractError('Forecast hash/horizon mismatch')
    return deepcopy(record)


def create_real_forecast(origin, features, *, horizon, prediction, algorithm, model_sha256, calendar):
    if origin.get('source') != 'VCI' or origin.get('quality_status') != 'ok':
        raise DataContractError('Require checked frozen VCI origin')
    if len(features) != 6 or not np.isfinite(features).all():
        raise DataContractError('Forecast needs six known returns')
    period, end = period_end(origin, calendar)
    stamp = utc_time(origin['timestamp']).isoformat()
    target = (utc_time(stamp) + timedelta(minutes=5 * horizon)).isoformat()
    record = {'schema_version': 3, 'symbol': origin['symbol'], 'session_id': origin['session_id'],
        'segment_id': origin['segment_id'], 'period': period, 'source_mode': 'HISTORICAL_REPLAY',
        'origin_bar_end': stamp, 'horizon_bars': horizon, 'target_bar_end': target, 'created_at': stamp,
        'available_at': stamp, 'as_of': stamp, 'prediction': float(prediction), 'origin_close': float(origin['close']),
        'origin_revision': 0, 'origin_available_at': stamp,
        'input_sha256': digest({'origin': origin, 'features': list(features)}), 'model_sha256': model_sha256,
        'calendar_sha256': digest(calendar), 'period_end': end,
        'horizon_status': 'censored_boundary' if utc_time(target) > utc_time(end) else 'awaiting_outcome',
        'clock_basis': 'historical_bar_end_assumption', 'availability_basis': AVAILABILITY, 'algorithm': algorithm}
    record['forecast_id'] = digest(record)
    return validate_real_forecast(record)


def assess_real_forecast(record, bars, *, as_of):
    forecast = validate_real_forecast(record)
    cutoff = utc_time(as_of)
    origin_time = utc_time(forecast['origin_bar_end'])
    if cutoff < origin_time:
        raise DataContractError('Assessment precedes forecast')
    if forecast['horizon_status'] == 'censored_boundary':
        return {'status': 'censored_boundary', 'post_evidence': None}
    if cutoff < utc_time(forecast['target_bar_end']):
        return {'status': 'pending', 'post_evidence': None}
    view = bars.loc[pd.to_datetime(bars.timestamp, utc=True).le(cutoff)]
    outcomes = []
    for step in range(1, forecast['horizon_bars'] + 1):
        t = origin_time + timedelta(minutes=5 * step)
        match = view.loc[view.symbol.eq(forecast['symbol']) & view.timestamp.eq(t)]
        if len(match) != 1:
            return {'status': 'censored_missing_bar' if cutoff >= utc_time(forecast['period_end']) else 'pending_missing_bar', 'post_evidence': None}
        row = match.iloc[0]
        if (row.source != 'VCI' or row.quality_status != 'ok' or row.session_id != forecast['session_id'] or
                row.segment_id != forecast['segment_id'] or not math.isfinite(float(row.close)) or row.close <= 0):
            raise DataContractError('Outcome path crossed data source/quality/segment')
        outcomes.append({'timestamp': t.isoformat(), 'close': float(row.close), 'segment_id': row.segment_id})
    realized = math.log(outcomes[-1]['close'] / forecast['origin_close'])
    evidence = {'horizon_bars': forecast['horizon_bars'], 'origin_bar_end': forecast['origin_bar_end'],
        'forecast_created_at': forecast['created_at'], 'target_bar_end': forecast['target_bar_end'],
        'outcome_available_at': forecast['target_bar_end'], 'prediction': forecast['prediction'],
        'realized_return': realized, 'residual': realized - forecast['prediction'], 'forecast_sha256': digest(forecast)}
    return {'status': 'mature', 'post_evidence': evidence, 'outcome_sha256': digest(outcomes)}
