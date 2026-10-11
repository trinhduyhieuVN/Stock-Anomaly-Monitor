"""Blind OHLCV packets and fail-closed review/split gates for AI research."""
from datetime import date, timedelta
import hashlib
import json

import numpy as np
import pandas as pd

from stock_monitor.io import DataContractError

BAR_FIELDS = ('symbol', 'timestamp', 'session_id', 'segment_id', 'open', 'high', 'low', 'close', 'volume', 'quality_status')
LABELS = {'noteworthy', 'ordinary', 'uncertain', 'data_issue'}
REVIEW_FIELDS = {'packet_id', 'unit', 'start', 'end', 'label', 'reason', 'reviewer', 'reviewer_kind', 'reviewed_at', 'expert_evidence'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise DataContractError('Expected SHA256')


def _utc(value):
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise DataContractError('Invalid UTC timestamp') from exc
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise DataContractError('Timestamp must be explicit UTC')
    return stamp


def validate_split_registry(splits, *, explored_dates):
    """Panel split dates are shared by all symbols; explored test is rejected."""
    if not isinstance(splits, dict) or set(splits) != {'train', 'validation', 'test'}:
        raise DataContractError('Require train/validation/test session registry')
    parsed = {}
    for name, values in splits.items():
        if not isinstance(values, list) or not values or values != sorted(set(values)):
            raise DataContractError('Split dates must be nonempty, unique and sorted')
        try:
            if any(date.fromisoformat(v).isoformat() != v for v in values):
                raise ValueError('Noncanonical date')
        except (ValueError, TypeError) as exc:
            raise DataContractError('Invalid split date') from exc
        parsed[name] = values
    if parsed['train'][-1] >= parsed['validation'][0] or parsed['validation'][-1] >= parsed['test'][0]:
        raise DataContractError('Split session boundaries overlap or are out of order')
    if set(parsed['test']) & set(explored_dates):
        raise DataContractError('Previously explored sessions cannot be independent test')
    # Unobserved old dates require a separate independent-source audit, not this future-session gate.
    if explored_dates and parsed['test'][0] <= max(explored_dates):
        raise DataContractError('Future holdout must follow the explored date range')
    return parsed


def blind_bars(frame):
    """An allowlist discards scores, predictions, truth and arbitrary extra fields."""
    if set(BAR_FIELDS) - set(frame.columns) or frame.empty:
        raise DataContractError('Missing or empty OHLCV packet input')
    bars = frame.loc[:, BAR_FIELDS].copy()
    bars['timestamp'] = pd.to_datetime([_utc(v) for v in bars.timestamp], utc=True)
    for name in ('symbol', 'session_id', 'segment_id', 'quality_status'):
        if not bars[name].map(lambda x: isinstance(x, str) and bool(x.strip()) and x == x.strip()).all():
            raise DataContractError('Invalid bar identity/quality')
    local = bars.timestamp.dt.tz_convert('Asia/Ho_Chi_Minh')
    if (not bars.session_id.eq(local.dt.strftime('%Y-%m-%d')).all() or
            not bars.timestamp.dt.minute.mod(5).eq(0).all() or
            not bars.timestamp.dt.second.eq(0).all() or
            not bars.timestamp.dt.microsecond.eq(0).all() or
            not bars.timestamp.dt.nanosecond.eq(0).all()):
        raise DataContractError('Bar date/boundary differs from declared five-minute close')
    for name in ('open', 'high', 'low', 'close', 'volume'):
        if bars[name].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise DataContractError('Boolean OHLCV is invalid')
        try:
            bars[name] = pd.to_numeric(bars[name], errors='raise')
        except (ValueError, TypeError) as exc:
            raise DataContractError('Invalid numeric OHLCV') from exc
    values = bars[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any() or
            (values[:, 4] % 1 != 0).any() or
            (bars.high < bars[['open', 'close', 'low']].max(axis=1)).any() or
            (bars.low > bars[['open', 'close', 'high']].min(axis=1)).any()):
        raise DataContractError('Invalid OHLCV bounds')
    if bars.duplicated(['symbol', 'timestamp']).any() or not bars.quality_status.eq('ok').all():
        raise DataContractError('Require unique checked bars')
    return bars.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def build_blind_packets(frame, *, sessions, dataset_sha256, rubric_sha256, context_bars=12):
    _hash(dataset_sha256)
    _hash(rubric_sha256)
    if type(context_bars) is not int or not 1 <= context_bars <= 12:
        raise DataContractError('Invalid context length')
    if not sessions or sessions != sorted(set(sessions)):
        raise DataContractError('Select fixed unique ordered sessions before scores')
    bars = blind_bars(frame)
    if set(sessions) - set(bars.session_id):
        raise DataContractError('Requested packet session missing')
    packets = []

    def packet(rows, task, focus=None):
        records = rows.copy()
        records['timestamp'] = records.timestamp.map(pd.Timestamp.isoformat)
        value = {'schema_version': 1, 'dataset_sha256': dataset_sha256, 'rubric_sha256': rubric_sha256,
                 'source_mode': 'HISTORICAL_REPLAY', 'data_role': 'development_rehearsal_not_holdout',
                 'availability_basis': 'retrospective_bar_end_assumption_no_receive_receipts',
                 'task': task, 'symbol': rows.symbol.iloc[0], 'session_id': rows.session_id.iloc[0],
                 'focus': focus, 'as_of': rows.timestamp.iloc[-1].isoformat(), 'bars': records.to_dict('records')}
        value['packet_id'] = digest(value)
        packets.append(value)

    for _, rows in bars.loc[bars.session_id.isin(sessions)].groupby(['symbol', 'session_id'], sort=True):
        history = []
        for index in rows.index:
            row = bars.loc[index]
            if history:
                previous = bars.loc[history[-1]]
                if row.segment_id != previous.segment_id or row.timestamp - previous.timestamp != pd.Timedelta(minutes=5):
                    history = []
            history = (history + [index])[-(context_bars + 1):]
            packet(bars.loc[history], 'initial_point', row.timestamp.isoformat())
        packet(rows, 'retrospective_segment')
    return packets


def verify_packet(packet):
    """Revalidate structure, not only the hash supplied by a caller."""
    expected = {'schema_version', 'dataset_sha256', 'rubric_sha256', 'source_mode', 'data_role',
                'availability_basis', 'task', 'symbol', 'session_id', 'focus', 'as_of', 'bars', 'packet_id'}
    if not isinstance(packet, dict) or set(packet) != expected:
        raise DataContractError('Unknown/missing packet fields')
    _hash(packet['dataset_sha256'])
    _hash(packet['rubric_sha256'])
    if packet['packet_id'] != digest({k: v for k, v in packet.items() if k != 'packet_id'}):
        raise DataContractError('Packet checksum mismatch')
    if any(set(row) != set(BAR_FIELDS) for row in packet['bars']):
        raise DataContractError('Blind packet contains unknown bar fields')
    bars = blind_bars(pd.DataFrame(packet['bars']))
    if (packet['schema_version'] != 1 or packet['source_mode'] != 'HISTORICAL_REPLAY' or
            packet['data_role'] != 'development_rehearsal_not_holdout' or
            packet['availability_basis'] != 'retrospective_bar_end_assumption_no_receive_receipts' or
            not bars.symbol.eq(packet['symbol']).all() or not bars.session_id.eq(packet['session_id']).all() or
            bars.timestamp.max() != _utc(packet['as_of'])):
        raise DataContractError('Packet scope/time mismatch')
    if packet['task'] == 'initial_point':
        if (len(bars) > 13 or _utc(packet['focus']) != _utc(packet['as_of']) or bars.segment_id.nunique() != 1 or
                not bars.timestamp.diff().dropna().eq(pd.Timedelta(minutes=5)).all()):
            raise DataContractError('Initial packet crosses a gap or contains future context')
    elif packet['task'] != 'retrospective_segment' or packet['focus'] is not None:
        raise DataContractError('Unknown packet task/focus')
    return bars


def annotation_template(packet):
    verify_packet(packet)
    return {name: packet['packet_id'] if name == 'packet_id' else '' for name in sorted(REVIEW_FIELDS)}


def validate_annotation(record, packet):
    bars = verify_packet(packet)
    if not isinstance(record, dict) or set(record) != REVIEW_FIELDS or record['packet_id'] != packet['packet_id']:
        raise DataContractError('Annotation must bind to its packet without extra fields')
    if record['label'] not in LABELS or record['reviewer_kind'] not in {'human', 'domain_expert', 'ai_assisted'}:
        raise DataContractError('Missing or invalid label/reviewer kind')
    if any(not isinstance(record[name], str) or not record[name].strip() for name in ('reviewer', 'reason')):
        raise DataContractError('Reviewer and reason required')
    if not isinstance(record['expert_evidence'], str) or (record['reviewer_kind'] == 'domain_expert' and not record['expert_evidence'].strip()):
        raise DataContractError('Expert status requires evidence reference')
    start, end = _utc(record['start']), _utc(record['end'])
    selected = bars.loc[bars.timestamp.between(start, end)]
    if start > end or start not in set(bars.timestamp) or end not in set(bars.timestamp):
        raise DataContractError('Annotation endpoints outside packet')
    unit = record['unit']
    if packet['task'] == 'initial_point' and (unit != 'point' or start != _utc(packet['focus']) or end != start):
        raise DataContractError('Initial review is only the focus point')
    if unit == 'point':
        if start != end:
            raise DataContractError('Point must have one bar')
    elif unit == 'segment':
        if len(selected) < 2 or selected.segment_id.nunique() != 1 or not selected.timestamp.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
            raise DataContractError('Segment must be continuous within one segment')
    elif unit == 'session':
        if packet['task'] != 'retrospective_segment' or start != bars.timestamp.min() or end != bars.timestamp.max() or record['label'] == 'noteworthy':
            raise DataContractError('Session summary scope/label invalid')
    else:
        raise DataContractError('Unknown annotation unit')
    # Actual review creation is audited separately; reject historical backdating here.
    if _utc(record['reviewed_at']) < _utc(packet['as_of']):
        raise DataContractError('Review time predates available observations')
    return dict(record)


def development_inventory(frame, split):
    bars = blind_bars(frame)
    dates = sorted(bars.session_id.unique())
    old = split['dates']
    validate_split_registry(old, explored_dates=[])
    assigned = {day: name for name, days in old.items() for day in days}
    if len(assigned) != sum(len(days) for days in old.values()):
        raise DataContractError('Original split overlaps')
    return {'symbols': sorted(bars.symbol.unique()), 'sessions': len(dates), 'dates': dates,
            'bars': len(bars), 'by_symbol': {symbol: len(rows) for symbol, rows in bars.groupby('symbol')},
            'all_dates_development_only': True, 'independent_test_sessions': [],
            'split_bar_counts': {name: int(bars.session_id.isin(days).sum()) for name, days in old.items()},
            'warmup_or_unassigned_bars': int((~bars.session_id.isin(assigned)).sum()),
            'availability_basis': 'retrospective_bar_end_assumption_no_receive_receipts'}
