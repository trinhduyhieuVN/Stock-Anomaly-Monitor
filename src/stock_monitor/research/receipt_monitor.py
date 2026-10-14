"""Frozen AI decisions on received closed bars, never historical backdating.

This seam owns no network. Replay contracts remain unchanged. Schema4 lives in
an independent append-only journal; initial decisions and forecasts survive
source revisions and restart. A receipt rehearsal is not a live certificate.
"""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from stock_monitor.config import load_config
from stock_monitor.features import compute_features
from stock_monitor.io import DataContractError
from stock_monitor.realtime import closed_bar_readiness
from stock_monitor.session_trial import trial_history
from stock_monitor.validation import load_profile
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.journal import Journal, JournalConflict
from stock_monitor.research.ai_demo import AI_SHA, verify_ai
from stock_monitor.research.density_samples import AVAILABILITY, _assemble_density_features
from stock_monitor.research.density_selector import ALGORITHM, compare_costs, select_episodes
from stock_monitor.research.frozen_density import infer_density
from stock_monitor.research.history import verify_history_binding
from stock_monitor.research.protocol import digest
from stock_monitor.research.real_forecast import period_end
from stock_monitor.research.receipt_density import RECEIPT_BASIS, saved_poll_prefix

MODES = {'OFFLINE_RECEIPT_REHEARSAL', 'RECEIVED_MARKET_DATA'}


def _validate_record(value):
    record = deepcopy(value)
    if (record.get('schema_version') != 4 or record.get('source_mode') not in MODES or
            record.get('availability_basis') != RECEIPT_BASIS or record.get('symbol') not in {'FPT', 'HPG', 'VCB'} or
            record.get('source_rights_status') != 'pending' or record.get('distributable') is not False):
        raise DataContractError('Expected local-only receipt contract4')
    available, cutoff = utc_time(record['available_at']), utc_time(record['data_as_of'])
    if available < cutoff:
        raise DataContractError('Decision availability precedes received input cutoff')
    if record['kind'] == 'decision':
        target, received = utc_time(record['target_time']), utc_time(record['received_at_utc'])
        if not target <= cutoff or not received <= cutoff or record['version'] != 1:
            raise DataContractError('Initial decision has unavailable input or invalid version')
        if record['model_sha256'] != AI_SHA or record['input_sha256'] != digest(record['inputs']):
            raise DataContractError('Initial model/input binding mismatch')
        if record['event_id'] != digest({k: record[k] for k in ('symbol', 'session_id', 'model', 'target_time', 'model_sha256')}):
            raise DataContractError('Initial decision identity mismatch')
        emission = record['emission']
        if (emission['target_time'] != record['target_time'] or emission['symbol'] != record['symbol'] or
                emission['decision_as_of'] != record['available_at'] or
                not math.isfinite(record['density_score']) or not math.isfinite(emission['joint_gain'])):
            raise DataContractError('Initial score/emission binding mismatch')
    elif record['kind'] == 'forecast':
        origin, target = utc_time(record['origin_bar_end']), utc_time(record['target_bar_end'])
        h = record['horizon_bars']
        if type(h) is not int or h not in (1, 3) or target != origin + timedelta(minutes=5*h):
            raise DataContractError('Invalid receipt forecast horizon')
        expected = ('censored_boundary' if target > utc_time(record['period_end']) else
                    'censored_late_creation' if available >= target else 'awaiting_outcome')
        if (origin > cutoff or record['horizon_status'] != expected or
                record['algorithm'] not in {'zero_return', 'ridge_h1_h3'} or
                not math.isfinite(record['prediction']) or record['origin_close'] <= 0 or
                record['algorithm'] == 'zero_return' and record['prediction'] != 0 or
                record['forecast_id'] != digest({k: v for k, v in record.items() if k != 'forecast_id'})):
            raise DataContractError('Forecast creation time/hash/policy mismatch')
    elif record['kind'] not in {'outcome', 'source_revision'}:
        raise DataContractError('Unknown receipt record kind')
    return record


class ReceiptJournal(Journal):
    """Use a NEW path: no migrations or reclassification of old Replay records."""
    def _validate(self, conn, collection, value):
        r = _validate_record(value)
        expected = {'decision': 'events', 'forecast': 'forecasts',
                    'outcome': 'assessments', 'source_revision': 'assessments'}[r['kind']]
        if collection != expected:
            raise DataContractError('Wrong receipt collection')
        event = r.get('event_id')
        identity = event if collection == 'events' else r.get('forecast_id') if collection == 'forecasts' else r['record_id']
        if collection != 'events':
            original = self._find(conn, 'events', event)
            if original is None or any(r[k] != original[k] for k in ('symbol', 'source_mode', 'model')):
                raise DataContractError('Receipt evidence requires the persisted initial decision')
            if utc_time(r['available_at']) < utc_time(original['available_at']):
                raise DataContractError('Evidence precedes initial decision')
        if r['kind'] == 'outcome':
            forecast = self._find(conn, 'forecasts', r['forecast_id'])
            if forecast is None or forecast['horizon_status'] != 'awaiting_outcome':
                raise DataContractError('Outcome requires a timely persisted forecast')
            if (utc_time(r['outcome_received_at']) > utc_time(r['data_as_of']) or
                    utc_time(r['target_bar_end']) > utc_time(r['data_as_of']) or
                    r['target_bar_end'] != forecast['target_bar_end'] or
                    r['forecast_sha256'] != digest(forecast) or
                    not math.isclose(r['realized_return'], math.log(r['target_close']/forecast['origin_close'])) or
                    not math.isclose(r['residual'], r['realized_return']-forecast['prediction'])):
                raise DataContractError('Outcome receipt/formula binding mismatch')
        if r['kind'] == 'source_revision' and r['initial_input_sha256'] != original['input_sha256']:
            raise DataContractError('Revision must reference the unchanged initial input')
        return r, identity, 1, event, 1, r['symbol'], r['available_at']

    def _append(self, conn, collection, value):
        # The base class special-cases schema2 assessments; contract4 uses exact
        # immutable content, so do not inherit that equivalence relaxation.
        r, identity, version, event, event_version, symbol, available = self._validate(conn, collection, value)
        previous = self._find(conn, collection, identity, version)
        if previous is not None:
            if previous != r:
                raise JournalConflict('Receipt ID already has different immutable content')
            return False
        import json
        from stock_monitor.upgrade.synthetic import content_hash
        conn.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)', (collection, identity, version, event,
            event_version, symbol, utc_time(available).isoformat(),
            json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False), content_hash(r)))
        return True


def bound_history(bundle, profile, history_evidence=None):
    root = bundle['root']
    original_profile = load_profile(load_config(root / 'configs/research.yaml'))
    if history_evidence is None:
        if profile != original_profile:
            raise DataContractError('Extended profile requires an independently pinned trial history')
        binding = verify_history_binding(root, 'demo/manifest.local.json', bundle['manifest']['history_manifest_sha256'])
        return pd.read_parquet(binding['dataset_directory'] / 'bars.parquet')
    _, trial_profile, bars = trial_history(root, history_evidence['path'], history_evidence['sha256'])
    if profile != trial_profile:
        raise DataContractError('Trial profile differs from its history manifest')
    # Extending observed dates does not authorize changing units, sessions or
    # feature math used by the frozen network.
    for key in ('price', 'volume', 'source', 'bar_label', 'diagnostic', 'research', 'confirmations_pending'):
        if profile.get(key) != original_profile.get(key):
            raise DataContractError('Trial source or unit convention changed')
    conventions = ('timezone', 'sessions', 'overrides', 'status', 'complete')
    if any(profile['calendar'].get(k) != original_profile['calendar'].get(k) for k in conventions):
        raise DataContractError('Trial session conventions changed')
    return bars


def _receipt_for(minutes, symbol, stamp):
    t = pd.Timestamp(stamp)
    rows = minutes.loc[minutes.symbol.eq(symbol) & minutes.timestamp.ge(t-pd.Timedelta(minutes=5)) & minutes.timestamp.lt(t)]
    if len(rows) != 5 or rows.timestamp.nunique() != 5:
        raise DataContractError('Closed target lacks its five persisted source minutes')
    return rows.retrieved_at_utc.max().isoformat()


def advance_receipt_monitor(bundle, store_root, profile, journal, *, symbols, observed_at_utc,
        provider_delay_seconds, provider_delay_evidence, stale_after_seconds=600,
        model='ai_context', history_evidence=None, source_mode='OFFLINE_RECEIPT_REHEARSAL', decision_clock=None):
    """Latest endpoint only: never create retroactive alerts for a whole backfill.

    Historical bars are an explicitly assumed reference, not observed receipts.
    All current-session inputs come from immutable received polls <= data_as_of.
    Forecasts use new causal features, never the saved historical sample arrays.
    """
    started = perf_counter()
    process_started = datetime.now(timezone.utc).isoformat()
    verify_ai(bundle)
    if source_mode not in MODES or model not in bundle['manifest']['configurations']:
        raise DataContractError('Unregistered receipt monitor mode/model')
    if source_mode == 'RECEIVED_MARKET_DATA' and decision_clock is None:
        raise DataContractError('Received-market decisions require an actual score-completion clock')
    observed = utc_time(pd.Timestamp(observed_at_utc).isoformat())
    historical = bound_history(bundle, profile, history_evidence)
    prefix = saved_poll_prefix(store_root, as_of=observed.isoformat(), symbols=symbols)
    ready = closed_bar_readiness(prefix['minutes'], profile, symbols=symbols, observed_at_utc=observed,
        provider_delay_seconds=provider_delay_seconds, stale_after_seconds=stale_after_seconds,
        provider_delay_evidence=provider_delay_evidence, project_root=bundle['root'], delay_measured_live=False)
    result = {'schema_version': 4, 'source_mode': source_mode, 'live': False, 'live_certified': False,
        'provider_calls': 0, 'model_fits': 0, 'model': model, 'manifest_sha256': AI_SHA,
        'source_rights_status': 'pending', 'distributable': False, 'availability_basis': RECEIPT_BASIS,
        'historical_reference_basis': AVAILABILITY, 'data_as_of': observed.isoformat(),
        'received_at_utc': prefix['received_at_utc'], 'readiness': ready.status, 'poll_pins': prefix['poll_pins'],
        'bars': [], 'events': [], 'forecasts': [], 'outcomes': [], 'new_decisions': 0}
    if not ready.status['scoring_allowed'] or ready.status['gap_count']:
        return dict(result, state='BLOCKED_READINESS', score_duration_seconds=perf_counter()-started)
    day = ready.status['calendar_date']
    config = bundle['manifest']['configurations'][model]
    calibration = bundle['calibrations']['E1' if config == 'E4' else config]
    if day <= max(bundle['forecast_policy']['validation_end'][:10], calibration['validation_end'][:10]):
        return dict(result, state='BLOCKED_BEFORE_MODEL_FREEZE', score_duration_seconds=perf_counter()-started)
    past = historical.loc[historical.symbol.isin(symbols) & historical.session_id.lt(day)]
    features = compute_features(pd.concat([past, ready.bars], ignore_index=True), profile['calendar']).frame
    samples = _assemble_density_features(features, {day: 'receipt_monitor'}, bars_known=len(ready.bars),
        history_length=12, availability_basis=RECEIPT_BASIS)
    variant = 'without_context' if config == 'E2' else 'with_context'
    inferred = infer_density(bundle['models'][variant], samples['X'], samples['y'], samples['context'],
        [r['symbol'] for r in samples['rows']], stable_prefix=True)
    available = utc_time(pd.Timestamp(decision_clock()).isoformat()) if decision_clock else observed
    if available < observed:
        raise DataContractError('Score completion clock precedes its input cutoff')
    common = {'schema_version': 4, 'source_mode': source_mode, 'availability_basis': RECEIPT_BASIS,
        'source_rights_status': 'pending', 'distributable': False, 'data_as_of': observed.isoformat(),
        'available_at': available.isoformat(), 'model': model}
    old = journal.query('events', as_of=available.isoformat())
    old_by_id = {r['event_id']: r for r in old}
    records, history = [], []
    for i, row in enumerate(samples['rows']):
        if history and (any(history[-1][1][k] != row[k] for k in ('symbol', 'session_id', 'segment_id')) or
                pd.Timestamp(row['target_time'])-pd.Timestamp(history[-1][1]['target_time']) != pd.Timedelta(minutes=5)):
            history = []
        history = (history + [(i, row)])[-12:]
        if row['target_time'] != ready.status['latest_by_symbol'].get(row['symbol']):
            continue
        inputs = {'X': samples['X'][i].tolist(), 'y': samples['y'][i].tolist(), 'context': samples['context'][i].tolist()}
        event = {**common, 'kind': 'decision', 'version': 1, 'symbol': row['symbol'], 'session_id': day,
            'segment_id': row['segment_id'], 'target_time': row['target_time'], 'model_sha256': AI_SHA,
            'inputs': inputs, 'input_sha256': digest(inputs), 'data_revision': prefix['prefix_sha256'],
            'received_at_utc': _receipt_for(prefix['minutes'], row['symbol'], row['target_time']),
            'density_score': float(inferred['scores'][i])}
        event['event_id'] = digest({k: event[k] for k in ('symbol', 'session_id', 'model', 'target_time', 'model_sha256')})
        previous = old_by_id.get(event['event_id'])
        if previous is not None:
            if previous['source_mode'] != source_mode:
                raise DataContractError('Cannot promote an offline journal to received market data')
            if previous['input_sha256'] != event['input_sha256']:
                revision = {**common, 'kind': 'source_revision', 'symbol': row['symbol'], 'event_id': event['event_id'],
                    'initial_input_sha256': previous['input_sha256'], 'revised_input_sha256': event['input_sha256'],
                    'data_revision': prefix['prefix_sha256']}
                revision['record_id'] = digest({k: revision[k] for k in ('event_id', 'revised_input_sha256')})
                if not any(v['record_id'] == revision['record_id'] for v in journal.query('assessments', as_of=available.isoformat())):
                    records.append(('assessments', revision))
            continue
        emission = compare_costs(inferred['residuals'][[idx for idx, _ in history]], bundle['design'], segments=config != 'E3')
        point = history[emission['point_index']][1]['target_time']
        emission.update({k: row[k] for k in ('symbol', 'session_id', 'segment_id', 'target_time', 'availability_basis')})
        emission.update(region_start=history[0][1]['target_time'], region_end=row['target_time'], region_count=len(history),
            start=history[-emission['segment_length']][1]['target_time'] if emission['classification'] == 'collective' else point,
            end=row['target_time'] if emission['classification'] == 'collective' else point,
            decision_as_of=available.isoformat(), algorithm=ALGORITHM, segments_enabled=config != 'E3',
            source_mode=source_mode, revision=prefix['prefix_sha256'])
        emission['emission_id'] = digest(emission)
        previous_emissions = [r['emission'] for r in old if r['model'] == model and r['symbol'] == row['symbol'] and r['session_id'] == day]
        selected = select_episodes(previous_emissions + [emission], calibration, bundle['design'])['emissions'][-1]
        event['emission'] = selected
        records.append(('events', event))
        origin = ready.bars.loc[ready.bars.symbol.eq(row['symbol']) & ready.bars.timestamp.eq(pd.Timestamp(row['target_time']))].iloc[0].to_dict()
        origin['timestamp'] = pd.Timestamp(origin['timestamp']).isoformat()
        period, end = period_end(origin, profile['calendar'])
        for h, index in ((1, 0), (3, 1)):
            algorithm = 'zero_return' if config == 'E4' else bundle['forecast_policy']['algorithm']
            returns = samples['X'][i, -5:, 0].tolist()+[float(samples['y'][i, 0])]
            prediction = 0. if algorithm == 'zero_return' else float(bundle['forecast']['model'].predict(
                bundle['forecast']['scaler'].transform([returns]))[0, index])
            target = (pd.Timestamp(row['target_time'])+pd.Timedelta(minutes=5*h)).isoformat()
            forecast = {**common, 'kind': 'forecast', 'event_id': event['event_id'], 'symbol': row['symbol'],
                'session_id': day, 'segment_id': row['segment_id'], 'origin_bar_end': row['target_time'],
                'origin_close': float(origin['close']), 'horizon_bars': h, 'target_bar_end': target,
                'period': period, 'period_end': end, 'algorithm': algorithm, 'prediction': prediction,
                'model_sha256': bundle['forecast_policy_sha256'], 'input_sha256': digest(returns),
                'horizon_status': 'censored_boundary' if utc_time(target) > utc_time(end) else
                    'censored_late_creation' if available >= utc_time(target) else 'awaiting_outcome'}
            forecast['forecast_id'] = digest(forecast)
            records.append(('forecasts', forecast))
    # Decision+forecast transaction is crash atomic. No outcome can precede it.
    verify_ai(bundle)
    journal.append_many(records)
    assessments = journal.query('assessments', as_of=available.isoformat())
    completed = {r['forecast_id'] for r in assessments if r['kind'] == 'outcome'}
    for forecast in journal.query('forecasts', as_of=available.isoformat()):
        if (forecast['model'] != model or forecast['session_id'] != day or forecast['forecast_id'] in completed or
                forecast['horizon_status'] != 'awaiting_outcome' or utc_time(forecast['target_bar_end']) > observed):
            continue
        path = []
        for step in range(1, forecast['horizon_bars']+1):
            stamp = pd.Timestamp(forecast['origin_bar_end'])+pd.Timedelta(minutes=5*step)
            match = ready.bars.loc[ready.bars.symbol.eq(forecast['symbol']) & ready.bars.timestamp.eq(stamp)]
            if len(match) != 1 or match.iloc[0].segment_id != forecast['segment_id']:
                path = []
                break
            receipt = _receipt_for(prefix['minutes'], forecast['symbol'], stamp)
            if utc_time(receipt) > observed:
                raise DataContractError('Future receipt reached outcome assessment')
            path.append({'timestamp': stamp.isoformat(), 'close': float(match.iloc[0].close), 'received': receipt})
        if path:
            realized = math.log(path[-1]['close']/forecast['origin_close'])
            outcome = {**common, 'kind': 'outcome', 'event_id': forecast['event_id'], 'symbol': forecast['symbol'],
                'forecast_id': forecast['forecast_id'], 'forecast_sha256': digest(forecast),
                'target_bar_end': forecast['target_bar_end'], 'outcome_received_at': max(v['received'] for v in path),
                'target_close': path[-1]['close'], 'realized_return': realized,
                'residual': realized-forecast['prediction'], 'path': path}
            outcome['record_id'] = digest({'forecast_id': forecast['forecast_id'], 'kind': 'first_received_outcome'})
            journal.append('assessments', outcome)
    verify_ai(bundle)
    latest_scored = all(any(r['symbol'] == s and r['target_time'] == ready.status['latest_by_symbol'].get(s)
                           for r in samples['rows']) for s in symbols)
    result.update(state='AI_READY' if latest_scored else 'INSUFFICIENT_HISTORY',
        bars=ready.bars.to_dict('records'), events=journal.query('events', as_of=available.isoformat()),
        forecasts=journal.query('forecasts', as_of=available.isoformat()),
        outcomes=journal.query('assessments', as_of=available.isoformat()),
        new_decisions=sum(collection == 'events' for collection, _ in records),
        score_duration_seconds=perf_counter()-started,
        score_finished_at_utc=available.isoformat() if decision_clock else None,
        process_started_at_utc=process_started, process_finished_at_utc=datetime.now(timezone.utc).isoformat(),
        decision_clock_basis='actual_score_clock' if decision_clock else 'offline_receipt_cutoff_not_wall_clock',
        exclusion_counts=samples['exclusion_counts'], warmup_history_bars_required=12,
        input_cutoff_utc=observed.isoformat(), prefix_sha256=prefix['prefix_sha256'])
    return result
