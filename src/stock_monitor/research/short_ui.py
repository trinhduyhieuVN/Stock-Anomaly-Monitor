"""Read-only projection of the opt-in GRU6 candidate. No model execution."""
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from stock_monitor.io import DataContractError
from stock_monitor.realtime import closed_bar_readiness
from stock_monitor.research.frozen_density import verify_pin
from stock_monitor.research.protocol import digest
from stock_monitor.research.receipt_density import saved_poll_prefix
from stock_monitor.research.short_alerts import MODEL, POLICY_SHA, PointJournal, journal_path
from stock_monitor.research.short_history import verify_short_bundle
from stock_monitor.session_sampling import utc_now
from stock_monitor.session_trial import trial_history
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.journal import APPLICATION_ID, SCHEMA_VERSION, Journal
from stock_monitor.workbench.service import clean

MODEL_PIN = {'path': 'artifacts/research/short_history/gru6_20261012/manifest.json',
    'sha256': '55b61a7546b45d372580043fc0348a5f9d7d13e50e3040bc25465f09c5fd3bfe'}


def _snapshot_path(root):
    path = (root / '.local/short_monitor/current.json').resolve()
    if not path.is_relative_to(root):
        raise DataContractError('Candidate snapshot escapes project')
    return path


def _bundle(root):
    manifest = json.loads(verify_pin(root, MODEL_PIN).read_text(encoding='utf-8'))
    bundle = {'root': root, 'manifest_pin': MODEL_PIN, 'manifest': manifest}
    verify_short_bundle(bundle)  # Hashes only: no torch.load/joblib/model inference.
    thresholds_pin = next(p for p in manifest['members'] if Path(p['path']).name == 'thresholds.json')
    return bundle, json.loads(verify_pin(root, thresholds_pin).read_text(encoding='utf-8'))


def _verify_records(root, value, thresholds):
    """Compare selected records to durable journal using a read-only connection."""
    path = journal_path(root, MODEL_PIN['sha256'], value['source_mode'])
    if value.get('journal_path') != path.relative_to(root).as_posix() or not path.is_file():
        raise DataContractError('Candidate journal binding unavailable')
    # Reuse the record validator, not its constructor (which creates/opens a writer).
    validator = object.__new__(PointJournal)
    validator.manifest_sha = MODEL_PIN['sha256']; validator.source_mode = value['source_mode']
    validator.thresholds = thresholds
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=2)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        if (conn.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID or
                conn.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION):
            raise DataContractError('Candidate journal schema mismatch')
        for collection, key in (('events', 'event_id'), ('outcomes', 'record_id')):
            records = value.get(collection, [])
            if not isinstance(records, list) or len(records) > 48:
                raise DataContractError('Candidate snapshot records exceed bounded size')
            identities = set()
            for record in records:
                name = 'events' if collection == 'events' else 'assessments'
                identity = record[key]
                if identity in identities:
                    raise DataContractError('Duplicate candidate decision identity')
                identities.add(identity)
                row = conn.execute('SELECT * FROM records WHERE collection=? AND record_id=? AND version=1',
                    (name, identity)).fetchone()
                if row is None or Journal._decode(row) != record:
                    raise DataContractError('Candidate snapshot differs from durable decision')
                validator._validate(conn, name, record)
                if (record['symbol'] != value['symbol'] or record['session_id'] != value['calendar_date'] or
                        utc_time(record['input_cutoff_utc']) > utc_time(value['input_cutoff_utc']) or
                        utc_time(record['available_at']) > utc_time(value['published_at_utc'])):
                    raise DataContractError('Candidate record outside snapshot symbol/session/cutoff')


def _context(root, value):
    if value['source_mode'] == 'OFFLINE_RECEIPT_REHEARSAL':
        report = verify_pin(root, value['saved_trial'])
        if not report.is_relative_to(root / 'reports/session_trials'):
            raise DataContractError('Candidate saved trial outside registered namespace')
        trial = json.loads(report.read_text(encoding='utf-8'))
        if trial.get('kind') != 'measured_session_trial' or not re.fullmatch(r'live_[a-f0-9]{24}', trial['cycle_id']):
            raise DataContractError('Candidate saved trial identity mismatch')
        evidence = trial['history_evidence']
        delay = trial['preflight']['provider_delay_seconds']
        store = (root / 'data/raw/live_trial' / trial['cycle_id']).resolve()
    else:
        if value.get('schema_version') == 6:
            from stock_monitor.research.short_context import verified_context
            bundle, _ = _bundle(root)
            return verified_context(root, value, bundle)
        # A received snapshot without a verified history binding cannot expose bars.
        records = value.get('events', []) + value.get('outcomes', [])
        if not records:
            raise DataContractError('Candidate received snapshot lacks durable history binding')
        evidence = records[-1]['inputs']['history_binding']
        delay = value['readiness']['provider_delay_seconds']
        store = (journal_path(root, MODEL_PIN['sha256'], value['source_mode']).parent / 'store').resolve()
    history, profile, _ = trial_history(root, evidence['path'], evidence['sha256'])
    if history['symbol_scope'] != [value['symbol']] or not store.is_relative_to(root):
        raise DataContractError('Candidate history/store scope mismatch')
    if value['source_mode'] == 'RECEIVED_MARKET_DATA' and value.get('config_path') != history['config_path']:
        raise DataContractError('Candidate received config differs from its history binding')
    for record in value.get('events', []) + value.get('outcomes', []):
        if record['inputs']['history_binding'] != evidence:
            raise DataContractError('Candidate decision history binding changed')
    return store, profile, delay


def read_candidate_projection(root, *, now=None):
    root = Path(root).resolve(); path = _snapshot_path(root)
    base = {'schema_version': 1, 'state': 'NOT_STARTED', 'model': MODEL,
        'model_role': 'opt_in_point_candidate_not_full_selector', 'source_mode': 'NO_SESSION',
        'live': False, 'live_certified': False, 'source_rights_status': 'pending', 'distributable': False,
        'bars': [], 'events': [], 'forecasts': [], 'outcomes': [], 'provider_calls': 0,
        'score_semantics': 'Point NLL; higher is rarer relative to the frozen validation threshold, not a probability',
        'message': 'Chưa có snapshot GRU6. Chọn mô hình không khởi chạy nguồn hay chấm lại.'}
    if not path.exists():
        return base
    try:
        if path.stat().st_size > 4_000_000:
            raise DataContractError('Candidate snapshot exceeds bounded size')
        envelope = json.loads(path.read_text(encoding='utf-8'))
        value = envelope['payload']
        if (envelope.get('schema_version') != 1 or not isinstance(value, dict) or
                envelope.get('sha256') != digest(value) or value.get('model') != MODEL or
                value.get('model_manifest') != MODEL_PIN or value.get('source_rights_status') != 'pending' or
                value.get('live_certified') is not False or value.get('live') is not False or value.get('forecasts')):
            raise DataContractError('Candidate snapshot hash/model/source policy mismatch')
        bundle, thresholds = _bundle(root)
        mode = value.get('source_mode')
        if mode == 'NO_SESSION':
            if value.get('events') or value.get('outcomes') or value.get('bars'):
                raise DataContractError('Candidate preflight cannot contain decisions or bars')
            return clean({**base, **value, 'artifact_status': 'verified', 'bars': []})
        if (mode not in {'OFFLINE_RECEIPT_REHEARSAL', 'RECEIVED_MARKET_DATA'} or
                value.get('schema_version') not in {5, 6} or value.get('manifest_sha256') != MODEL_PIN['sha256'] or
                value.get('policy_sha256') != POLICY_SHA or value.get('distributable') is not False):
            raise DataContractError('Candidate source/model/policy mismatch')
        if value['schema_version'] == 6 and mode != 'RECEIVED_MARKET_DATA':
            raise DataContractError('Explicit schema6 context is for received snapshots only')
        cutoff = utc_time(value['input_cutoff_utc']); receive = utc_time(value['received_at_utc'])
        published = utc_time(value['published_at_utc'])
        if not receive <= cutoff <= published:
            raise DataContractError('Candidate snapshot receipt/cutoff/publication chronology invalid')
        if value.get('score_finished_at_utc') and not cutoff <= utc_time(value['score_finished_at_utc']) <= published:
            raise DataContractError('Candidate completion clock outside snapshot chronology')
        if mode == 'OFFLINE_RECEIPT_REHEARSAL' and (value.get('provider_calls') != 0 or value.get('actual_receive_to_score_seconds') is not None):
            raise DataContractError('Candidate offline snapshot cannot claim live requests/latency')
        _verify_records(root, value, thresholds)
        store, profile, delay = _context(root, value)
        prefix = saved_poll_prefix(store, as_of=cutoff.isoformat(), symbols=[value['symbol']])
        if (prefix['prefix_sha256'] != value['prefix_sha256'] or prefix['poll_pins'] != value['poll_pins'] or
                utc_time(prefix['received_at_utc']) != receive):
            raise DataContractError('Candidate immutable receipt prefix mismatch')
        ready = closed_bar_readiness(prefix['minutes'], profile, symbols=[value['symbol']],
            observed_at_utc=cutoff, provider_delay_seconds=delay, stale_after_seconds=600,
            provider_delay_evidence=value['readiness']['provider_delay_evidence'], project_root=root,
            delay_measured_live=False)  # No live certificate is issued by a browser GET.
        for key in ('calendar_date', 'eligibility_cutoff_utc', 'latest_by_symbol', 'gap_count', 'scoring_allowed'):
            if ready.status.get(key) != value['readiness'].get(key):
                raise DataContractError('Candidate readiness differs from immutable closed-bar view')
        if value['calendar_date'] != ready.status['calendar_date']:
            raise DataContractError('Candidate calendar date mismatch')
        bars = clean(ready.bars.to_dict('records'))
        cutoff_bar = utc_time(ready.status['eligibility_cutoff_utc'])
        if any(utc_time(b['timestamp']) > cutoff_bar for b in bars):
            raise DataContractError('Candidate projection includes a forming/future bar')
        result = {**base, **value, 'bars': bars, 'artifact_status': 'verified', 'provider_delay_seconds': delay,
            'delay_basis': 'offline_hypothesis_not_current_measurement' if mode == 'OFFLINE_RECEIPT_REHEARSAL' else 'saved_receipt_basis_not_new_live_certification'}
        if not ready.status['scoring_allowed'] or ready.status['gap_count']:
            result.update(state='WAITING_FIRST_CLOSED_BAR' if ready.status['state'] == 'WAITING_FIRST_CLOSED_BAR' else 'BLOCKED_READINESS', events=[], outcomes=[])
        if value.get('monitor_running') is True:
            age = (utc_time(now if now is not None else utc_now().isoformat()) - published).total_seconds()
            interval = value.get('interval_seconds', 60)
            if type(interval) is not int or not 15 <= interval <= 300:
                raise DataContractError('Candidate monitor interval invalid')
            if age < 0 or age > max(180, interval * 3):
                result.update(state='SNAPSHOT_STALE', live=False,
                    message='Monitor GRU6 chưa cập nhật. Đang giữ snapshot đã xác minh gần nhất.')
        verify_short_bundle(bundle)
        return clean(result)
    except DataContractError:
        raise  # Preserve actionable checksum/policy diagnostics (it is a ValueError).
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error, StopIteration) as exc:
        raise DataContractError('Cannot verify candidate local snapshot') from exc
