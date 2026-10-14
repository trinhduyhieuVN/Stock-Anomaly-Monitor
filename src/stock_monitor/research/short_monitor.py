"""Finite opt-in candidate monitor. No change to the old monitor/UI default."""
import json
import os
from pathlib import Path
import re
import time
import uuid

import pandas as pd

from stock_monitor.incremental import load_incremental_current, load_incremental_poll
from stock_monitor.io import DataContractError
from stock_monitor.live_trial import preflight_live_trial, execute_vci_one_cycle
from stock_monitor.near_realtime import monitor_lock, validate_monitor
from stock_monitor.providers import public_transport
from stock_monitor.realtime import closed_bar_readiness
from stock_monitor.research.frozen_density import verify_pin
from stock_monitor.research.protocol import digest
from stock_monitor.research.receipt_monitor import bound_history
from stock_monitor.research.short_alerts import PointJournal, advance_point_alerts, journal_path, POLICY, POLICY_SHA
from stock_monitor.research.short_history import load_short_density, verify_short_bundle
from stock_monitor.session_sampling import utc_now
from stock_monitor.session_trial import trial_history
from stock_monitor.workbench.service import clean


def _directory(root):
    root = Path(root).resolve()
    directory = (root / '.local/short_monitor').resolve()
    if not directory.is_relative_to(root):
        raise DataContractError('Candidate monitor directory escapes project')
    return directory


def write_snapshot(root, value):
    directory = _directory(root); directory.mkdir(parents=True, exist_ok=True)
    payload = clean(value)
    envelope = {'schema_version': 1, 'payload': payload, 'sha256': digest(payload)}
    encoded = json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(encoded) > 4_000_000:
        raise DataContractError('Candidate snapshot exceeds bounded size')
    temp = directory / ('current.' + uuid.uuid4().hex + '.next')
    with temp.open('xb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, directory / 'current.json')
    return payload


def validate_candidate(root, manifest_pin, *, history_evidence=None, delay_evidence=None,
        observed_at_utc=None, max_polls=6, interval_seconds=60):
    bundle = load_short_density(root, manifest_pin['path'], manifest_pin['sha256'])
    report, profile, _ = validate_monitor(root, history_evidence=history_evidence, delay_evidence=delay_evidence,
        observed_at_utc=observed_at_utc, max_polls=max_polls, interval_seconds=interval_seconds)
    bound_history(bundle, profile, history_evidence)  # Unit/feature compatibility before ANY transport.
    report.update(model='gru_context_6_candidate', model_manifest=manifest_pin, policy=POLICY,
        policy_sha256=POLICY_SHA, model_role='opt_in_point_candidate_not_full_selector',
        ui_default_changed=False, new_alerts=0, new_decisions=0)
    return report, profile, bundle


def execute_candidate_cycle(bundle, preflight, *, store_root, cycle_id, journal, profile,
        history_evidence=None, delay_evidence=None, source_mode='OFFLINE_RECEIPT_REHEARSAL',
        transport, clock, sleep):
    """Injected offline cycle or actual measured opt-in cycle, never a fake live."""
    if source_mode == 'RECEIVED_MARKET_DATA':
        if transport is not public_transport or clock is not utc_now or not delay_evidence:
            raise DataContractError('Received candidate cycle requires actual transport/UTC clock and fresh measured delay')
    elif source_mode != 'OFFLINE_RECEIPT_REHEARSAL' or transport is public_transport or delay_evidence is not None:
        raise DataContractError('Offline candidate cycles require injected transport without live delay certificates')
    limits = preflight.get('limits', {})
    if limits.get('max_polls') != 1 or type(limits.get('max_attempts')) is not int or not 1 <= limits['max_attempts'] <= 3 or not 0 <= limits.get('max_wait_seconds', -1) <= 10:
        raise DataContractError('Candidate cycle requires one poll, at most3attempts/10s retry wait')
    verify_short_bundle(bundle); bound_history(bundle, profile, history_evidence)

    def score(readiness, received):
        if not readiness.status['scoring_allowed'] or readiness.status['gap_count']:
            raise DataContractError('Candidate scorer cannot bypass readiness')
        # Snapshot post-receive UTC once; do not backdate inference to bar end.
        return advance_point_alerts(bundle, store_root, profile, journal, symbols=preflight['symbols'],
            observed_at_utc=clock(), provider_delay_seconds=preflight['provider_delay_seconds'],
            provider_delay_evidence=preflight['provider_delay_evidence'], history_evidence=history_evidence,
            source_mode=source_mode, decision_clock=clock)

    return execute_vci_one_cycle(preflight, store_root=store_root, cycle_id=cycle_id, profile=profile,
        project_root=bundle['root'], transport=transport, clock=clock, sleep=sleep, count_back=500,
        timeout_seconds=10, readiness_fn=closed_bar_readiness, score_fn=score, delay_observations=delay_evidence)


def rehearse_candidate(root, manifest_pin, trial_pin):
    """Persist new candidate decisions on an old receipt, no provider or LIVE."""
    root = Path(root).resolve()
    path = verify_pin(root, trial_pin)
    if not path.is_relative_to(root / 'reports/session_trials'):
        raise DataContractError('Saved candidate trial report outside allowed namespace')
    trial = json.loads(path.read_text(encoding='utf-8'))
    if trial.get('kind') != 'measured_session_trial' or not trial.get('history_evidence'):
        raise DataContractError('Pinned saved real trial required')
    history, profile, _ = trial_history(root, trial['history_evidence']['path'], trial['history_evidence']['sha256'])
    cycle = trial['cycle_id']
    if not re.fullmatch(r'live_[a-f0-9]{24}', cycle):
        raise DataContractError('Unsafe saved trial cycle')
    store = root / 'data/raw/live_trial' / cycle
    _, pointer = load_incremental_current(store)
    if not pointer:
        raise DataContractError('Saved candidate trial store unavailable')
    symbol = history['symbol_scope'][0]
    observed = max(load_incremental_poll(store, p, symbol)[1] for p in pointer['applied_poll_ids']).isoformat()
    with monitor_lock(root):
        bundle = load_short_density(root, manifest_pin['path'], manifest_pin['sha256'])
        journal = PointJournal(journal_path(root, manifest_pin['sha256'], 'OFFLINE_RECEIPT_REHEARSAL'), bundle, 'OFFLINE_RECEIPT_REHEARSAL')
        result = advance_point_alerts(bundle, store, profile, journal, symbols=[symbol], observed_at_utc=observed,
            provider_delay_seconds=trial['preflight']['provider_delay_seconds'], history_evidence=trial['history_evidence'],
            provider_delay_evidence='Saved trial delay as offline hypothesis, not a fresh live certificate')
        result.update(model_manifest=manifest_pin, saved_trial=trial_pin, symbol=symbol,
            calendar_date=result['readiness']['calendar_date'], published_at_utc=utc_now(),
            provider_calls=0, live_certified=False, actual_receive_to_score_seconds=None,
            journal_path=journal.path.relative_to(root).as_posix(),
            message='Chấm receipt thật lưu trước bằng candidate GRU6. Không mở nguồn hay thay mô hình mặc định.')
        return write_snapshot(root, result)


def run_candidate_monitor(root, manifest_pin, *, history_evidence, delay_evidence,
        max_polls=6, interval_seconds=60, transport=public_transport, clock=utc_now,
        sleep=time.sleep, cancelled=lambda: False):
    if transport is not public_transport or clock is not utc_now:
        raise DataContractError('Candidate monitor requires actual VCI transport and UTC clock; use offline cycles in tests')
    root = Path(root).resolve()
    with monitor_lock(root):
        report, profile, bundle = validate_candidate(root, manifest_pin, history_evidence=history_evidence,
            delay_evidence=delay_evidence, observed_at_utc=clock(), max_polls=max_polls, interval_seconds=interval_seconds)
        report.update(published_at_utc=pd.Timestamp(clock()).isoformat(), monitor_running=False,
            actual_receive_to_score_seconds=None)
        if report['state'] != 'VALIDATED_NO_NETWORK':
            return write_snapshot(root, report)
        journal = PointJournal(journal_path(root, manifest_pin['sha256'], 'RECEIVED_MARKET_DATA'), bundle, 'RECEIVED_MARKET_DATA')
        store = journal.path.parent / 'store'  # Stable across restarts: no alert-cap reset.
        run_id = 'gru6_' + uuid.uuid4().hex[:20]
        directory = _directory(root) / 'runs' / run_id; directory.mkdir(parents=True, exist_ok=False)
        report.update(run_id=run_id, run_directory=directory.relative_to(root).as_posix(),
            journal_path=journal.path.relative_to(root).as_posix(), monitor_running=True, execution_enabled=True)
        write_snapshot(root, report)
        requests = 0
        for index in range(max_polls):
            if cancelled():
                report.update(state='STOPPED'); break
            try:
                fresh, latest_profile, _ = validate_candidate(root, manifest_pin, history_evidence=history_evidence,
                    delay_evidence=delay_evidence, observed_at_utc=clock(), max_polls=max_polls, interval_seconds=interval_seconds)
                if fresh['state'] != 'VALIDATED_NO_NETWORK' or latest_profile != profile:
                    report.update(state=fresh['state']); break
                verify_short_bundle(bundle)
                preflight = preflight_live_trial(root, config_path=report['config_path'], observed_at_utc=clock(),
                    provider_delay_seconds=report['provider_delay_seconds'],
                    provider_delay_evidence=f"{delay_evidence['path']} sha256={delay_evidence['sha256']}",
                    max_polls=1, max_attempts=3, max_wait_seconds=10)
                preflight['symbols'] = [report['symbol']]
                report['provider_calls'] = 'unknown_after_unreturned_executor'
                result = execute_candidate_cycle(bundle, preflight, store_root=store, cycle_id=f'{run_id}_{index:02}',
                    journal=journal, profile=profile, history_evidence=history_evidence, delay_evidence=delay_evidence,
                    source_mode='RECEIVED_MARKET_DATA', transport=transport, clock=clock, sleep=sleep)
                requests += sum(a.get('attempt', 0) > 0 for a in result.attempts)
                if result.score_result:
                    report.update(clean(result.score_result))
                    end = pd.Timestamp(result.score_result['score_finished_at_utc'])
                    receive = pd.Timestamp(result.score_result['received_at_utc'])
                    report['actual_receive_to_score_seconds'] = (end-receive).total_seconds()
                elif result.readiness:
                    # A newer receipt without a score must not inherit older
                    # decision timestamps, prefix hashes or score latency.
                    for key in ('input_cutoff_utc','score_finished_at_utc','decision_available_at_utc'):
                        report.pop(key, None)
                    report.update(events=[], outcomes=[], new_decisions=0, new_alerts=0,
                        score_duration_seconds=None, actual_receive_to_score_seconds=None)
                ready = (result.score_result or {}).get('readiness') or (result.readiness.status if result.readiness else report.get('readiness', {}))
                report.update(state=result.state, live=False, live_certified=False, live_candidate=result.live_evidence,
                    provider_calls=requests, polls_completed=index+1, attempts=list(result.attempts),
                    readiness=ready, message=result.operator_message,
                    published_at_utc=pd.Timestamp(clock()).isoformat())
                if result.readiness:
                    from stock_monitor.research.short_context import bind_received_snapshot
                    report = bind_received_snapshot(report, bundle, store=store,
                        history_evidence=history_evidence, delay_evidence=delay_evidence)
                write_snapshot(root, report)
                with (directory / 'cycles.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(clean(report), ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                if result.state not in {'SCORED_LIVE_CANDIDATE', 'READY_FOR_SCORE', 'WAITING_FIRST_CLOSED_BAR'}:
                    break
                if index+1 < max_polls:
                    for _ in range(interval_seconds):
                        if cancelled():
                            break
                        sleep(1)
            except DataContractError:
                report.update(state='BLOCKED_CONTEXT_OR_ARTIFACT'); break
            except KeyboardInterrupt:
                report.update(state='STOPPED'); break
            except Exception as exc:
                report.update(state='BLOCKED_RUNTIME', error_type=type(exc).__name__); break
        else:
            report.update(state='FINISHED')
        report.update(published_at_utc=pd.Timestamp(clock()).isoformat(), monitor_running=False,
            execution_enabled=False, live=False, live_certified=False)
        return write_snapshot(root, report)
