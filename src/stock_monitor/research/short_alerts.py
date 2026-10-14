"""Opt-in six-bar POINT alerts, isolated from the frozen full selector.

No forecast/segment claim. Decisions include normal and suppressed points;
revisions append evidence and never rewrite or re-emit the initial decision.
"""
from copy import deepcopy
import math
from pathlib import Path
from time import perf_counter

import pandas as pd

from stock_monitor.io import DataContractError
from stock_monitor.research.protocol import digest
from stock_monitor.research.receipt_density import RECEIPT_BASIS, saved_poll_prefix
from stock_monitor.research.receipt_monitor import ReceiptJournal
from stock_monitor.research.short_history import score_short_receipt, verify_short_bundle
from stock_monitor.session_sampling import utc_now
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.journal import Journal, JournalConflict
from stock_monitor.workbench.service import clean

MODEL = 'gru_context_6_candidate'
MODES = {'OFFLINE_RECEIPT_REHEARSAL', 'RECEIVED_MARKET_DATA'}
POLICY = {'id': 'gru6_point_policy_1', 'comparison': 'score > frozen per-symbol validation quantile0.98',
    'maximum_alert_points_per_symbol_session': 3, 'ordering': 'latest approved endpoint, chronological',
    'duplicates': 'one initial decision per model/symbol/session/target',
    'revision': 'append only; no new alert for an already decided endpoint',
    'role': 'operational point cap, not an episode/calibration/effectiveness claim'}
POLICY_SHA = digest(POLICY)


def journal_path(root, manifest_sha256, source_mode):
    if source_mode not in MODES or not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 or any(c not in '0123456789abcdef' for c in manifest_sha256):
        raise DataContractError('Invalid point journal namespace')
    root = Path(root).resolve()
    # Windows MAX_PATH: do not nest two64-char hashes beneath a long workspace.
    # Full bindings stay in every record; any namespace collision fails closed.
    namespace = digest({'model_sha256': manifest_sha256, 'source_mode': source_mode, 'policy_sha256': POLICY_SHA})[:24]
    path = (root / '.local/short_monitor' / namespace / 'decisions.sqlite').resolve()
    if not path.is_relative_to(root):
        raise DataContractError('Point journal escapes project')
    return path


class PointJournal(ReceiptJournal):
    """Reuse exact append/transactions/triggers, not the schema4 validator."""
    def __init__(self, path, bundle, source_mode):
        verify_short_bundle(bundle)
        if source_mode not in MODES or not Path(path).resolve().is_relative_to(bundle['root']):
            raise DataContractError('Point journal path/mode invalid')
        self.manifest_sha = bundle['manifest_pin']['sha256']
        self.thresholds = deepcopy(bundle['thresholds'])
        self.source_mode = source_mode
        super().__init__(path)
        with self.transaction() as conn:
            for row in conn.execute('SELECT * FROM records'):
                self._decode(row)  # Refuse old/mixed journals; never migrate them.

    def _decode(self, row):
        result = Journal._decode(row)
        self._common(result)
        return result

    def _common(self, record):
        required = {'schema_version', 'model', 'model_sha256', 'source_mode', 'policy_sha256', 'availability_basis',
            'source_rights_status', 'distributable', 'symbol', 'session_id', 'target_time', 'input_cutoff_utc',
            'available_at', 'received_at_utc', 'target_received_at_utc', 'score_finished_at_utc', 'score_clock_basis',
            'density_score', 'threshold', 'input_sha256', 'inputs', 'event_id', 'version', 'kind'}
        if not isinstance(record, dict) or required - set(record):
            raise DataContractError('Incomplete point record')
        if (record.get('schema_version') != 5 or record.get('model') != MODEL or
                record.get('model_sha256') != self.manifest_sha or record.get('source_mode') != self.source_mode or
                record.get('policy_sha256') != POLICY_SHA or record.get('availability_basis') != RECEIPT_BASIS or
                record.get('source_rights_status') != 'pending' or record.get('distributable') is not False or
                record.get('symbol') not in self.thresholds):
            raise DataContractError('Point journal schema/model/source/policy mismatch')
        cutoff = utc_time(record['input_cutoff_utc']); available = utc_time(record['available_at'])
        target = utc_time(record['target_time']); received = utc_time(record['received_at_utc'])
        if not target <= cutoff <= available or not utc_time(record['target_received_at_utc']) <= received <= cutoff or record['score_finished_at_utc'] != record['available_at']:
            raise DataContractError('Point decision receipt/cutoff/score chronology invalid')
        if pd.Timestamp(target).tz_convert('Asia/Ho_Chi_Minh').date().isoformat() != record['session_id']:
            raise DataContractError('Point session does not match target')
        if record['score_clock_basis'] not in {'actual_utc_completion', 'injected_offline_clock'} or self.source_mode == 'RECEIVED_MARKET_DATA' and record['score_clock_basis'] != 'actual_utc_completion':
            raise DataContractError('Received-market point requires actual completion clock')
        score = record['density_score']; threshold = record['threshold']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or threshold != self.thresholds[record['symbol']]['threshold']:
            raise DataContractError('Point score/threshold mismatch')
        if record['input_sha256'] != digest(record['inputs']):
            raise DataContractError('Point input hash mismatch')
        inputs = record['inputs']; minutes = inputs.get('minutes', [])
        if (inputs.get('feature_order') != ['log_return', 'range_ratio', 'body_ratio', 'volume_surprise'] or
                len(minutes) != 40 or any(v.get('symbol') != record['symbol'] for v in minutes)):
            raise DataContractError('Point minute provenance shape/symbol mismatch')
        stamps = [utc_time(v['timestamp']) for v in minutes]
        expected = [target - pd.Timedelta(minutes=40-i) for i in range(40)]
        if stamps != expected:
            raise DataContractError('Point minute provenance is not the causal40-minute window')
        identity = digest({k: record[k] for k in ('symbol', 'session_id', 'target_time', 'model', 'model_sha256', 'policy_sha256')})
        if record['event_id'] != identity:
            raise DataContractError('Point decision ID mismatch')

    def _validate(self, conn, collection, value):
        r = deepcopy(value); self._common(r)
        initial = self._find(conn, 'events', r['event_id'])
        identity = r['event_id'] if collection == 'events' else r.get('record_id')
        if collection == 'events' and r['kind'] == 'point_decision':
            if type(r['version']) is not int or r['version'] != 1 or r['emission']['target_time'] != r['target_time']:
                raise DataContractError('Initial point identity/version mismatch')
            if initial is None:
                prior = [self._decode(row) for row in conn.execute("SELECT * FROM records WHERE collection='events' AND symbol=?", (r['symbol'],))]
                if any(utc_time(v['input_cutoff_utc']) > utc_time(r['input_cutoff_utc']) or utc_time(v['available_at']) > utc_time(r['available_at']) for v in prior):
                    raise JournalConflict('Point journal cannot rewind cutoff or decision clock; use a new rehearsal journal')
                count = sum(v['session_id'] == r['session_id'] and v['emission']['state'] == 'alert' for v in prior)
                expected = 'normal' if r['density_score'] <= r['threshold'] else 'alert' if count < 3 else 'budget_suppressed'
                if r['emission']['state'] != expected:
                    raise DataContractError('Point emission violates chronological frozen cap')
            if (r['emission']['joint_gain'] != r['density_score'] or r['emission']['threshold'] != r['threshold'] or
                    r['emission']['decision_as_of'] != r['available_at'] or r['emission']['kind'] != 'point'):
                raise DataContractError('Point emission score/time mismatch')
        elif collection == 'assessments' and r['kind'] == 'source_revision':
            if initial is None or r['initial_input_sha256'] != initial['input_sha256'] or utc_time(r['available_at']) < utc_time(initial['available_at']):
                raise DataContractError('Point revision requires immutable initial decision')
            past = [self._decode(row) for row in conn.execute("SELECT * FROM records WHERE collection='assessments' AND event_id=? ORDER BY version", (r['event_id'],))]
            existing = self._find(conn, collection, identity)
            if existing is None:
                previous = max(past, key=lambda v: v['revision_index']) if past else initial
                if (type(r['revision_index']) is not int or r['revision_index'] != len(past)+1 or
                        r['previous_input_sha256'] != previous['input_sha256'] or r['input_sha256'] == previous['input_sha256'] or
                        utc_time(r['input_cutoff_utc']) < utc_time(previous['input_cutoff_utc']) or
                        utc_time(r['available_at']) < utc_time(previous['available_at'])):
                    raise DataContractError('Point revision ordering/content invalid')
            expected_id = digest({'event_id': r['event_id'], 'revision_index': r['revision_index'], 'input_sha256': r['input_sha256']})
            if r['record_id'] != expected_id or r['alerts_emitted'] is not False:
                raise DataContractError('Point revision cannot emit or change its identity')
        else:
            raise DataContractError('Point journal accepts only initial points and source revisions')
        return r, identity, 1, r['event_id'], 1, r['symbol'], r['available_at']


def _inputs(prefix, row, history):
    # Six feature histories + target need the bar preceding the first history
    # bar for its log-return: eight bars /40 source minutes, not just35minutes.
    start = utc_time(row['input_start']) - pd.Timedelta(minutes=10)
    end = utc_time(row['target_time'])
    minutes = prefix['minutes']
    selected = minutes.loc[minutes.symbol.eq(row['symbol']) & minutes.timestamp.ge(start) & minutes.timestamp.lt(end)]
    if len(selected) != 40 or selected.timestamp.nunique() != 40:
        raise DataContractError('Point input provenance requires40 contiguous source minutes')
    # Do not hash refreshed receipt times: identical OHLCV in a later poll is
    # not a data revision. Persisted poll pins remain in separate provenance.
    values = selected.sort_values('timestamp')[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return {'minutes': clean(values), 'history_binding': history,
        'feature_order': ['log_return', 'range_ratio', 'body_ratio', 'volume_surprise']}


def advance_point_alerts(bundle, store_root, profile, journal, *, symbols, observed_at_utc,
        provider_delay_seconds, provider_delay_evidence, history_evidence=None,
        source_mode='OFFLINE_RECEIPT_REHEARSAL', decision_clock=utc_now):
    started = perf_counter()
    if source_mode not in MODES or source_mode == 'RECEIVED_MARKET_DATA' and decision_clock is not utc_now:
        raise DataContractError('Received-market point adapter requires actual UTC completion clock')
    if journal.source_mode != source_mode or journal.manifest_sha != bundle['manifest_pin']['sha256']:
        raise DataContractError('Point journal differs from scoring model/source')
    result = score_short_receipt(bundle, store_root, profile, symbols=symbols, observed_at_utc=observed_at_utc,
        provider_delay_seconds=provider_delay_seconds, provider_delay_evidence=provider_delay_evidence,
        history_evidence=history_evidence)
    if digest(POLICY) != POLICY_SHA:
        raise DataContractError('Point policy mutated')
    result.update(schema_version=5, source_mode=source_mode, policy=deepcopy(POLICY), policy_sha256=POLICY_SHA,
        events=[], outcomes=[], forecasts=[], new_decisions=0, new_alerts=0, new_revisions=0,
        model_feature_score_duration_seconds=result['score_duration_seconds'])
    if result['state'] not in {'DENSITY_READY', 'INSUFFICIENT_HISTORY'}:
        return result
    prefix = saved_poll_prefix(store_root, as_of=result['input_cutoff_utc'], symbols=symbols)
    if prefix['prefix_sha256'] != result['prefix_sha256']:
        raise DataContractError('Point input prefix changed during scoring')
    finished = pd.Timestamp(utc_time(pd.Timestamp(decision_clock()).isoformat()))
    if finished < utc_time(result['input_cutoff_utc']):
        raise DataContractError('Point completion precedes input cutoff')
    old = journal.query('events', as_of=finished.isoformat())
    revisions = journal.query('assessments', as_of=finished.isoformat())
    old_by_id = {v['event_id']: v for v in old}
    records = []
    history = history_evidence or {'path': bundle['manifest']['history_manifest'], 'sha256': bundle['manifest']['history_manifest_sha256']}
    for row in result['latest_scores']:
        inputs = _inputs(prefix, row, history)
        base = {k: row[k] for k in ('symbol', 'session_id', 'segment_id', 'target_time')}
        base.update(schema_version=5, model=MODEL, model_sha256=bundle['manifest_pin']['sha256'],
            source_mode=source_mode, policy_sha256=POLICY_SHA, availability_basis=RECEIPT_BASIS,
            source_rights_status='pending', distributable=False, version=1,
            input_cutoff_utc=result['input_cutoff_utc'], received_at_utc=result['received_at_utc'],
            target_received_at_utc=row['target_received_at_utc'], score_finished_at_utc=finished.isoformat(),
            available_at=finished.isoformat(), inputs=inputs, input_sha256=digest(inputs),
            score_clock_basis='actual_utc_completion' if decision_clock is utc_now else 'injected_offline_clock',
            poll_pins=result['poll_pins'], density_score=row['score'], threshold=row['threshold'])
        base['event_id'] = digest({k: base[k] for k in ('symbol', 'session_id', 'target_time', 'model', 'model_sha256', 'policy_sha256')})
        prior = old_by_id.get(base['event_id'])
        if prior:
            previous_revisions = [v for v in revisions if v['event_id'] == base['event_id']]
            previous = max(previous_revisions, key=lambda v: v['revision_index']) if previous_revisions else prior
            if previous['input_sha256'] == base['input_sha256']:
                continue
            base.update(kind='source_revision', revision_index=len(previous_revisions)+1,
                initial_input_sha256=prior['input_sha256'], previous_input_sha256=previous['input_sha256'], alerts_emitted=False)
            base['record_id'] = digest({'event_id': base['event_id'], 'revision_index': base['revision_index'], 'input_sha256': base['input_sha256']})
            records.append(('assessments', base))
        else:
            count = sum(v['symbol'] == row['symbol'] and v['session_id'] == row['session_id'] and v['emission']['state'] == 'alert' for v in old)
            state = 'normal' if row['score'] <= row['threshold'] else 'alert' if count < 3 else 'budget_suppressed'
            base.update(kind='point_decision', emission={'kind': 'point', 'state': state,
                'symbol': row['symbol'], 'target_time': row['target_time'], 'joint_gain': row['score'],
                'threshold': row['threshold'], 'decision_as_of': finished.isoformat()})
            records.append(('events', base))
    verify_short_bundle(bundle)
    with journal.transaction() as conn:
        for collection, record in records:
            if journal._append(conn, collection, record):
                result['new_revisions' if collection == 'assessments' else 'new_decisions'] += 1
                if collection == 'events' and record['emission']['state'] == 'alert':
                    result['new_alerts'] += 1
        verify_short_bundle(bundle)  # An artifact change rolls the whole commit back.
        committed = pd.Timestamp(utc_time(pd.Timestamp(decision_clock()).isoformat()))
        if committed < finished:
            raise DataContractError('Point completion clock moved backwards')
    # Completion of scoring includes durable journal commit, not just forward.
    day = result['readiness']['calendar_date']
    visible = lambda collection: [v for v in journal.query(collection, as_of=committed.isoformat()) if v['symbol'] in symbols and v['session_id'] == day]
    events, outcomes = visible('events'), visible('assessments')
    result.update(state='AI_READY' if result['state'] == 'DENSITY_READY' else 'INSUFFICIENT_HISTORY',
        events=events[-48:], outcomes=outcomes[-48:], event_count=len(events), revision_count=len(outcomes),
        snapshot_record_limit=48, score_duration_seconds=perf_counter()-started,
        score_finished_at_utc=committed.isoformat(), decision_available_at_utc=finished.isoformat(),
        alerts_emitted=result['new_alerts'] > 0,
        score_clock_basis='actual_utc_completion' if decision_clock is utc_now else 'injected_offline_clock')
    return result
