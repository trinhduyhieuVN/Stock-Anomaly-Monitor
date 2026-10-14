"""Finite local market monitor, using the existing guarded VCI executor.

No unattended infinite loop, retraining, calendar invention or source-rights
promotion. The browser only reads an atomic projection and never starts a feed.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

import pandas as pd

from stock_monitor.config import load_config
from stock_monitor.io import DataContractError
from stock_monitor.live_trial import execute_vci_one_cycle, preflight_live_trial
from stock_monitor.providers import public_transport
from stock_monitor.realtime import closed_bar_readiness, verify_frozen_scoring_manifest
from stock_monitor.research.ai_demo import AI_SHA, load_ai, verify_ai
from stock_monitor.research.protocol import digest
from stock_monitor.research.receipt_monitor import ReceiptJournal, advance_receipt_monitor
from stock_monitor.session_sampling import session_gate, utc_now, verified_delay_basis
from stock_monitor.session_trial import trial_history
from stock_monitor.validation import load_profile
from stock_monitor.workbench.service import clean


def _directory(root):
    root = Path(root).resolve()
    directory = (root / '.local/near_realtime').resolve()
    if not directory.is_relative_to(root):
        raise DataContractError('Monitor directory escapes the project')
    return directory


def atomic_projection(root, payload):
    directory = _directory(root)
    directory.mkdir(parents=True, exist_ok=True)
    data = clean(payload)
    envelope = {'schema_version': 1, 'payload': data, 'sha256': digest(data)}
    path = directory/'current.json'
    temp = directory/('current.'+uuid.uuid4().hex+'.next')
    with temp.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(envelope, stream, ensure_ascii=False, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return data


def read_projection(root, *, now=None):
    path = _directory(root)/'current.json'
    if not path.exists():
        return {'schema_version': 1, 'state': 'NOT_STARTED', 'source_mode': 'NO_SESSION',
            'live': False, 'live_certified': False, 'source_rights_status': 'pending',
            'bars': [], 'events': [], 'forecasts': [], 'outcomes': [], 'provider_calls': 0,
            'message': 'Chưa chạy monitor. Replay vẫn hoạt động độc lập.'}
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise DataContractError('Monitor projection escapes the project')
    if path.stat().st_size > 4_000_000:
        raise DataContractError('Monitor projection exceeds its bounded size')
    envelope = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(envelope, dict) or not isinstance(envelope.get('payload'), dict):
        raise DataContractError('Invalid monitor projection envelope')
    value = envelope['payload']
    if (envelope.get('schema_version') != 1 or envelope['sha256'] != digest(value) or
            value.get('source_rights_status') != 'pending' or value.get('live_certified') is not False):
        raise DataContractError('Monitor projection hash/policy mismatch')
    if value.get('source_mode') not in {'RECEIVED_MARKET_DATA', 'OFFLINE_RECEIPT_REHEARSAL', 'NO_SESSION'}:
        raise DataContractError('Unregistered monitor source mode')
    if value.get('ai_manifest_sha256') == AI_SHA or value.get('manifest_sha256') == AI_SHA:
        # Read-only status must not keep saying "verified" after an artifact was
        # replaced. Hash-check without deserializing or running model weights.
        manifest = json.loads((Path(root)/'demo/ai_outline.local.json').read_text(encoding='utf-8'))
        members = manifest['members']
        pins = members['selector']+members['forecast']+members['samples']+sum(members['density'].values(),[])
        verify_ai({'root':Path(root).resolve(),'pins':pins})
        verify_frozen_scoring_manifest(root,'demo/manifest.local.json')
    current = pd.Timestamp(now or utc_now())
    stamp = value.get('published_at_utc')
    if stamp and value.get('monitor_running') is True:
        age = (current-pd.Timestamp(stamp)).total_seconds()
        if age < 0 or age > max(180, value.get('interval_seconds', 60)*3):
            value = {**value, 'state': 'SNAPSHOT_STALE', 'live': False,
                'message': 'Monitor chưa cập nhật. Đang giữ dữ liệu lần nhận gần nhất.'}
    return value


def rehearse_saved_trial(root, report_path, expected_sha256):
    """Run new frozen AI over a real old receipt, NOT a new provider trial."""
    from stock_monitor.io import sha256
    from stock_monitor.incremental import load_incremental_current, load_incremental_poll
    root = Path(root).resolve(); path = (root/report_path).resolve()
    if not path.is_relative_to(root/'reports/session_trials') or sha256(path) != expected_sha256:
        raise DataContractError('Saved trial report path/checksum mismatch')
    trial = json.loads(path.read_text(encoding='utf-8'))
    if trial.get('kind') != 'measured_session_trial' or not trial.get('history_evidence'):
        raise DataContractError('Pinned measured trial receipt required for rehearsal')
    evidence = trial['history_evidence']
    history, profile, _ = trial_history(root, evidence['path'], evidence['sha256'])
    cycle = trial['cycle_id']
    import re
    if not re.fullmatch(r'live_[a-f0-9]{24}', cycle):
        raise DataContractError('Unsafe saved trial cycle identity')
    store = root/'data/raw/live_trial'/cycle
    minutes, pointer = load_incremental_current(store)
    if pointer is None or minutes.empty:
        raise DataContractError('Saved trial store unavailable')
    symbol = history['symbol_scope'][0]
    # Old materialized CSVs round receipt time to seconds. The immutable raw
    # manifests retain full precision; ONLY those verified receipts define T.
    receipts = [load_incremental_poll(store, poll_id, symbol)[1] for poll_id in pointer['applied_poll_ids']]
    observed = max(receipts).isoformat()
    with monitor_lock(root):
        run_id = 'rehearsal_'+uuid.uuid4().hex[:20]
        directory = _directory(root)/'runs'/run_id
        journal = ReceiptJournal(directory/'decisions.sqlite')
        value = advance_receipt_monitor(load_ai(root), store, profile, journal, symbols=[symbol],
            observed_at_utc=observed, provider_delay_seconds=trial['preflight']['provider_delay_seconds'],
            provider_delay_evidence='Old trial delay replayed as an offline hypothesis, not fresh live evidence',
            history_evidence=evidence)
        value.update(criteria_met=False, symbol=symbol, run_id=run_id, calendar_date=value['readiness']['calendar_date'],
            published_at_utc=utc_now(), interval_seconds=60, provider_calls=0,
            saved_trial={'path': str(report_path), 'sha256': expected_sha256},
            ai_manifest_sha256=AI_SHA, message='Receipt thật lưu từ phiên cũ; chấm lại offline bằng AI mới. Không mở nguồn, không đo latency live.')
        return atomic_projection(root, value)


@contextmanager
def monitor_lock(root):
    """OS lock releases on exit/crash; never delete another writer's lock file."""
    directory = _directory(root); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'writer.lock').open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise DataContractError('A near-realtime writer is already running; stop it before starting another') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise DataContractError('A near-realtime writer is already running') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_monitor(root, *, history_evidence=None, delay_evidence=None, observed_at_utc=None,
        max_polls=6, interval_seconds=60):
    if type(max_polls) is not int or not 1 <= max_polls <= 12:
        raise DataContractError('Monitor requires 1..12 polls')
    if type(interval_seconds) is not int or not 15 <= interval_seconds <= 300:
        raise DataContractError('Monitor interval must be 15..300 seconds')
    if max_polls*40+(max_polls-1)*interval_seconds > 3600:
        raise DataContractError('Maximum HTTP/retry/wait schedule must not exceed one hour')
    root = Path(root).resolve(); observed = observed_at_utc or utc_now()
    frozen = verify_frozen_scoring_manifest(root, 'demo/manifest.local.json'); bundle = load_ai(root)
    if history_evidence:
        history, profile, _ = trial_history(root, history_evidence['path'], history_evidence['sha256'])
        symbol = history['symbol_scope'][0]; config = history['config_path']
    else:
        profile = load_profile(load_config(root/'configs/research.yaml'))
        symbol = None; config = 'configs/research.yaml'
    state, day = session_gate(profile, observed)
    report = {'schema_version': 1, 'criteria_met': False, 'state': state,
        'source_mode': 'NO_SESSION', 'live': False, 'live_certified': False,
        'provider_calls': 0, 'model_fits': 0, 'source_rights_status': 'pending',
        'calendar_date': day, 'symbol': symbol, 'config_path': config,
        'frozen_manifest_sha256': frozen['manifest_sha256'], 'ai_manifest_sha256': AI_SHA,
        'max_polls': max_polls, 'interval_seconds': interval_seconds,
        'max_attempts_per_poll': 3, 'max_retry_wait_seconds_per_poll': 10, 'timeout_seconds': 10,
        'maximum_transport_requests': max_polls*3, 'execution_enabled': False,
        'bars': [], 'events': [], 'forecasts': [], 'outcomes': []}
    if state != 'SESSION_OPEN':
        return report, profile, bundle
    if not history_evidence:
        return {**report, 'state': 'BLOCKED_HISTORY_EVIDENCE'}, profile, bundle
    if not delay_evidence:
        return {**report, 'state': 'BLOCKED_MEASURED_DELAY'}, profile, bundle
    basis = verified_delay_basis(root, delay_evidence['path'], delay_evidence['sha256'],
        config_path=config, symbol=symbol, now=observed)
    return {**report, 'state': 'VALIDATED_NO_NETWORK', 'provider_delay_seconds': basis['empirical_delay_seconds']}, profile, bundle


def run_live_monitor(root, *, history_evidence, delay_evidence, max_polls=6, interval_seconds=60,
        transport=public_transport, clock=utc_now, sleep=time.sleep, cancelled=lambda: False):
    """Real network opt-in only. Injected transports cannot certify a live run."""
    if transport is not public_transport or clock is not utc_now:
        raise DataContractError('Live monitor requires the actual VCI transport and UTC clock; use receipt rehearsal for tests')
    root = Path(root).resolve()
    with monitor_lock(root):
        report, profile, bundle = validate_monitor(root, history_evidence=history_evidence,
            delay_evidence=delay_evidence, observed_at_utc=clock(), max_polls=max_polls, interval_seconds=interval_seconds)
        run_id = 'monitor_'+uuid.uuid4().hex[:20]
        directory = _directory(root)/'runs'/run_id
        directory.mkdir(parents=True, exist_ok=False)
        journal = ReceiptJournal(directory/'decisions.sqlite')
        report.update(run_id=run_id, published_at_utc=pd.Timestamp(clock()).isoformat(),
            run_directory=directory.relative_to(root).as_posix(), history_evidence=history_evidence,
            delay_evidence=delay_evidence)
        atomic_projection(root, report)
        if report['state'] != 'VALIDATED_NO_NETWORK':
            return report
        report['monitor_running'] = True
        report['execution_enabled'] = True
        requests = 0
        # Identity checks inside execute_vci_one_cycle require public_transport.
        # Count actual attempts from returned receipts rather than wrapping it.
        for index in range(max_polls):
            if cancelled():
                report.update(state='STOPPED', live=False); break
            observed = clock()
            try:
                refreshed, latest_profile, _ = validate_monitor(root, history_evidence=history_evidence,
                    delay_evidence=delay_evidence, observed_at_utc=observed, max_polls=max_polls, interval_seconds=interval_seconds)
                if refreshed['state'] != 'VALIDATED_NO_NETWORK' or latest_profile != profile:
                    report.update(state=refreshed['state'], live=False); break
                verify_ai(bundle)
                preflight = preflight_live_trial(root, config_path=report['config_path'], observed_at_utc=observed,
                    provider_delay_seconds=report['provider_delay_seconds'],
                    provider_delay_evidence=f"{delay_evidence['path']} sha256={delay_evidence['sha256']}",
                    max_polls=1, max_attempts=3, max_wait_seconds=10)
                preflight['symbols'] = [report['symbol']]
                report['context'] = preflight.get('context')
                def score(readiness, received):
                    # Readiness is rechecked against post-receive time. The data
                    # cutoff and score completion time remain separate.
                    if not readiness.status['scoring_allowed'] or readiness.status['gap_count']:
                        raise DataContractError('Readiness has not approved this endpoint')
                    value = advance_receipt_monitor(bundle, directory/'store', profile, journal,
                        symbols=[report['symbol']], observed_at_utc=clock(),
                        provider_delay_seconds=report['provider_delay_seconds'],
                        provider_delay_evidence=preflight['provider_delay_evidence'], history_evidence=history_evidence,
                        source_mode='RECEIVED_MARKET_DATA', decision_clock=clock)
                    if value['state'] not in {'AI_READY', 'INSUFFICIENT_HISTORY'}:
                        raise DataContractError('New AI readiness failed closed')
                    return value
                # If an unexpected scorer failure prevents the executor from
                # returning its attempt ledger, the request count is UNKNOWN,
                # never an invented zero. Normal results replace this value.
                report['provider_calls'] = 'chưa xác minh'
                result = execute_vci_one_cycle(preflight, store_root=directory/'store', cycle_id=f'{run_id}_{index:02}',
                    profile=profile, project_root=root, transport=transport, clock=clock, sleep=sleep,
                    timeout_seconds=10, readiness_fn=closed_bar_readiness, score_fn=score, delay_observations=delay_evidence)
                requests += sum(a.get('attempt', 0) > 0 for a in result.attempts)
                score_result = result.score_result or {}
                if score_result:
                    report.update(clean(score_result))
                report.update(state=result.state, live=False, live_certified=False,
                    provider_calls=requests, polls_completed=index+1, attempts=list(result.attempts),
                    readiness=result.readiness.status if result.readiness else {},
                    live_candidate=result.live_evidence, message=result.operator_message)
                if score_result:
                    end = pd.Timestamp(score_result['score_finished_at_utc'])
                    receive = pd.Timestamp(score_result['received_at_utc'])
                    report['actual_receive_to_score_seconds'] = (end-receive).total_seconds()
                report['published_at_utc'] = pd.Timestamp(clock()).isoformat()
                atomic_projection(root, report)
                with (directory/'cycles.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(clean(report), ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                if result.state not in {'SCORED_LIVE_CANDIDATE', 'READY_FOR_SCORE', 'WAITING_FIRST_CLOSED_BAR'}:
                    break
                if index+1 < max_polls:
                    # Finite short waits allow prompt Ctrl+C and cancellation.
                    for _ in range(interval_seconds):
                        if cancelled():
                            break
                        sleep(1)
            except DataContractError:
                report.update(state='BLOCKED_CONTEXT_OR_ARTIFACT', live=False,
                    message='Calendar, delay, history or frozen artifacts failed verification. No automatic override.')
                break
            except KeyboardInterrupt:
                report.update(state='STOPPED', live=False); break
            except Exception as exc:
                report.update(state='BLOCKED_RUNTIME', live=False, error_type=type(exc).__name__,
                    message='Monitor stopped after an unexpected processing error; inspect local logs before retrying.')
                break
        else:
            report.update(state='FINISHED', live=False)
        report['published_at_utc'] = pd.Timestamp(clock()).isoformat()
        report['monitor_running'] = False
        atomic_projection(root, report)
        return report
