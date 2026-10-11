"""Causal bar-density inputs; historical availability is an explicit assumption."""
from collections import Counter

import numpy as np
import pandas as pd

from stock_monitor.features import FEATURE_ORDER, compute_features
from stock_monitor.io import DataContractError
from stock_monitor.research.protocol import blind_bars, validate_split_registry

AVAILABILITY = 'historical_bar_end_assumption_without_per_bar_receive_receipts'
CONTEXT_ORDER = ('lagged_volatility', 'past_trend', 'lagged_relative_volume', 'time_sin', 'time_cos')


def build_density_samples(bars, calendar, splits, *, as_of, availability_basis, history_length=12):
    """Conditioning ends at t-1; target is observed feature vector at closed t.

    No scalers or models are fit here. Only the selected historical bar-end
    assumption is supported; this function is not a live receipt adapter.
    """
    if availability_basis != AVAILABILITY or type(history_length) is not int or history_length != 12:
        raise DataContractError('Explicit historical availability and registered history length12 required')
    split = validate_split_registry(splits, explored_dates=[])
    try:
        cutoff = pd.Timestamp(as_of)
    except (ValueError, TypeError) as exc:
        raise DataContractError('Invalid density as_of') from exc
    if pd.isna(cutoff) or cutoff.tzinfo is None or cutoff.utcoffset().total_seconds() != 0:
        raise DataContractError('Density as_of must be explicit UTC')
    # Validate only known prefix. Data after T is neither inspected nor used for features.
    if 'timestamp' not in bars:
        raise DataContractError('Missing timestamps')
    times = []
    for raw in bars.timestamp:
        stamp = pd.Timestamp(raw)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise DataContractError('Bar timestamps must be explicit UTC')
        times.append(stamp)
    scoped = bars.loc[pd.Series(times, index=bars.index).le(cutoff)].copy()
    if scoped.empty:
        raise DataContractError('No closed bars at requested T')
    blind_bars(scoped)
    if (set(scoped.symbol) - {'FPT', 'HPG', 'VCB'} or 'source' not in scoped or not scoped.source.eq('VCI').all() or
            'interval' not in scoped or not scoped.interval.eq('5m').all()):
        raise DataContractError('Density development adapter requires checked VCI five-minute bars')
    # A caller cannot substitute recorded delays for the declared historical assumption.
    if {'available_at', 'received_at', 'decision_as_of'} & set(scoped.columns):
        raise DataContractError('Observed receipts need a separate causal-availability adapter')
    features = compute_features(scoped, calendar).frame
    date_split = {day: key for key, dates in split.items() for day in dates}
    return _assemble_density_features(features, date_split, bars_known=len(scoped),
        history_length=history_length, availability_basis=AVAILABILITY)


def _assemble_density_features(features, date_split, *, bars_known, history_length, availability_basis):
    """Shared tensor math; public adapters must validate their availability first."""
    X, y, contexts, rows = [], [], [], []
    reasons = Counter()
    for _, group in features.groupby(['symbol', 'session_id'], sort=True):
        sequence = []
        for index in group.index:
            current = features.loc[index]
            if sequence:
                previous = features.loc[sequence[-1]]
                if current.segment_id != previous.segment_id or current.timestamp - previous.timestamp != pd.Timedelta(minutes=5):
                    sequence = []
            past = sequence[-history_length:]
            sequence.append(index)
            name = date_split.get(current.session_id)
            if name is None:
                reasons['warmup_or_unassigned_session'] += 1
                continue
            if not current.feature_eligible:
                reasons['target_feature_unavailable'] += 1
                continue
            if len(past) != history_length:
                reasons['insufficient_contiguous_history'] += 1
                continue
            before = features.loc[past]
            if not before.feature_eligible.all():
                reasons['history_feature_unavailable'] += 1
                continue
            if not before.timestamp.lt(current.timestamp).all():
                raise DataContractError('Conditioning contains target or future')
            # Model timestamp context must come from a previous observation too.
            local = before.timestamp.iloc[-1].tz_convert('Asia/Ho_Chi_Minh')
            angle = 2 * np.pi * (local.hour * 60 + local.minute) / (24 * 60)
            context = [before.log_return.std(ddof=0), before.log_return.mean(), before.volume_surprise.iloc[-1],
                       np.sin(angle), np.cos(angle)]
            inputs = before[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
            target = current[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)
            if not np.isfinite(inputs).all() or not np.isfinite(target).all() or not np.isfinite(context).all():
                raise DataContractError('Eligible density inputs must be finite')
            X.append(inputs)
            y.append(target)
            contexts.append(context)
            rows.append({'symbol': current.symbol, 'session_id': current.session_id,
                         'segment_id': current.segment_id, 'split': name, 'target_time': current.timestamp.isoformat(),
                         'input_start': before.timestamp.iloc[0].isoformat(),
                         'input_end': before.timestamp.iloc[-1].isoformat(),
                         'as_of': current.timestamp.isoformat(), 'availability_basis': availability_basis})
    return {'X': np.asarray(X, dtype=np.float64).reshape(-1, history_length, len(FEATURE_ORDER)),
            'y': np.asarray(y, dtype=np.float64).reshape(-1, len(FEATURE_ORDER)),
            'context': np.asarray(contexts, dtype=np.float64).reshape(-1, len(CONTEXT_ORDER)),
            'rows': rows, 'feature_order': list(FEATURE_ORDER), 'context_order': list(CONTEXT_ORDER),
            'bars_known': bars_known, 'samples': len(rows), 'exclusion_counts': dict(reasons),
            'availability_basis': availability_basis, 'fits': 0}
