"""Offline saved-poll receipt rehearsal, separate from historical Replay.

No provider, training, alert journal or LIVE claim. This evaluates a verified
poll prefix at one supplied decision time; it does not reconstruct old alerts
using the latest mutable minute view. Live selector/forecast integration remains
a separate gate before a provider pilot.
"""
import json
from pathlib import Path
import re
from time import perf_counter

import pandas as pd

from stock_monitor.features import compute_features
from stock_monitor.config import load_config
from stock_monitor.incremental import load_incremental_current, load_incremental_poll, merge_incremental_minutes, REQUIRED_COLUMNS
from stock_monitor.io import DataContractError, sha256
from stock_monitor.providers import parse_vci_minutes
from stock_monitor.realtime import closed_bar_readiness
from stock_monitor.validation import load_profile
from stock_monitor.research.ai_demo import AI_SHA, verify_ai
from stock_monitor.research.density_samples import AVAILABILITY, _assemble_density_features
from stock_monitor.research.frozen_density import infer_density
from stock_monitor.research.history import verify_history_binding
from stock_monitor.research.protocol import digest
from stock_monitor.upgrade.contracts import utc_time

RECEIPT_BASIS = 'saved_poll_receipts_at_T_with_declared_historical_reference_assumption'


def saved_poll_prefix(store_root, *, as_of, symbols):
    """Rebuild immutable polls <=T; never filter the latest revised view to fake T."""
    cutoff = pd.Timestamp(utc_time(as_of))
    root = Path(store_root).resolve()
    _, pointer = load_incremental_current(root)
    if pointer is None:
        raise DataContractError('Verified persisted poll store required')
    poll_ids = pointer.get('applied_poll_ids')
    if (not isinstance(poll_ids, list) or
            any(not isinstance(p, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', p) for p in poll_ids) or
            len(poll_ids) != len(set(poll_ids))):
        raise DataContractError('Unsafe or duplicate persisted poll identity')
    version = json.loads((root / pointer['version_manifest']).read_text(encoding='utf-8'))
    if poll_ids != version.get('applied_poll_ids'):
        raise DataContractError('Poll list differs from verified materialized version')
    wanted = set(symbols)
    if not wanted or wanted - {'FPT', 'HPG', 'VCB'}:
        raise DataContractError('Registered symbol scope required')
    frame = pd.DataFrame(columns=REQUIRED_COLUMNS)
    pins, receipts = [], []
    previous = None
    for poll_id in poll_ids:
        # load_incremental_poll validates IDs, paths, raw bytes and immutable
        # receipt/view bindings, including a changed receipt after restart.
        manifest_path = (root / 'polls' / poll_id / 'manifest.json').resolve()
        if not manifest_path.is_relative_to(root):
            raise DataContractError('Saved receipt manifest escapes the store')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        received = pd.Timestamp(utc_time(manifest['retrieved_at_utc']))
        if received > cutoff or manifest['symbol'] not in wanted:
            continue
        if previous is not None and received < previous:
            raise DataContractError('Saved poll application chronology is not monotone')
        previous = received
        saved = load_incremental_poll(root, poll_id, manifest['symbol'])
        if saved is None:
            raise DataContractError('Applied poll missing at requested T')
        raw, verified_received = saved
        if verified_received != received:
            raise DataContractError('Receipt changed during prefix loading')
        incoming = parse_vci_minutes(json.loads(raw), manifest['symbol'])
        incoming['retrieved_at_utc'] = received
        frame = merge_incremental_minutes(frame, incoming[REQUIRED_COLUMNS]).minutes
        receipts.append(received)
        pins.append({'poll_id': poll_id, 'symbol': manifest['symbol'], 'manifest_sha256': sha256(manifest_path),
                     'raw_sha256': manifest['raw_sha256'], 'received_at_utc': received.isoformat()})
    if not pins:
        raise DataContractError('No persisted receipt in requested prefix')
    return {'minutes': frame, 'poll_pins': pins, 'received_at_utc': max(receipts).isoformat(),
            'prefix_sha256': digest(pins), 'as_of': cutoff.isoformat(), 'live_evidence': False}


def score_saved_poll_density(bundle, store_root, profile, *, symbols, observed_at_utc,
        provider_delay_seconds, stale_after_seconds, provider_delay_evidence, model='ai_context'):
    """Score closed current-session targets only; no initial/delayed alerts yet."""
    started = perf_counter()
    verify_ai(bundle)
    if model not in bundle['manifest']['configurations']:
        raise DataContractError('Unknown registered AI configuration')
    observed = pd.Timestamp(utc_time(observed_at_utc))
    root = bundle['root']
    if profile != load_profile(load_config(root / 'configs/research.yaml')):
        raise DataContractError('Receipt rehearsal cannot silently change the pinned source/unit/calendar profile')
    store = Path(store_root).resolve()
    if not store.is_relative_to(root):
        raise DataContractError('Receipt store must stay in the local project')
    prefix = saved_poll_prefix(store, as_of=observed.isoformat(), symbols=symbols)
    readiness = closed_bar_readiness(prefix['minutes'], profile, symbols=symbols,
        observed_at_utc=observed.isoformat(), provider_delay_seconds=provider_delay_seconds,
        stale_after_seconds=stale_after_seconds, provider_delay_evidence=provider_delay_evidence,
        project_root=root, delay_measured_live=False)
    base = {'schema_version': 1, 'source_mode': 'OFFLINE_SAVED_POLL_REHEARSAL', 'live': False,
        'provider_calls': 0, 'model_fits': 0, 'live_receive_to_score_seconds': None,
        'received_at_utc': prefix['received_at_utc'], 'decision_as_of': observed.isoformat(),
        'readiness': readiness.status, 'poll_pins': prefix['poll_pins'], 'manifest_sha256': AI_SHA,
        'availability_basis': RECEIPT_BASIS, 'historical_reference_basis': AVAILABILITY,
        'source_rights_status': 'pending', 'distributable': False, 'rows': [], 'scores': [],
        'alerts_emitted': False, 'initial_and_post_event_receipt_journal': 'not_implemented'}
    # Conservative new seam: no gap/stale/calendar override, even when the old
    # scorer could process an independently eligible latest segment.
    if not readiness.status['scoring_allowed'] or readiness.status['gap_count']:
        return dict(base, state='BLOCKED_READINESS', score_duration_seconds=perf_counter() - started)
    day = readiness.status['calendar_date']
    if day <= bundle['forecast_policy']['validation_end'][:10]:
        return dict(base, state='BLOCKED_BEFORE_MODEL_FREEZE', score_duration_seconds=perf_counter() - started)
    binding = verify_history_binding(root, 'demo/manifest.local.json', bundle['manifest']['history_manifest_sha256'])
    historical = pd.read_parquet(binding['dataset_directory'] / 'bars.parquet')
    past = historical.loc[historical.symbol.isin(symbols) & historical.session_id.lt(day)].copy()
    current = readiness.bars.copy()
    calendar = profile['calendar']
    # Feature math stays identical to training: past calendar-session volume
    # slots are needed; no synthetic volume history or forward-filling.
    features = compute_features(pd.concat([past, current], ignore_index=True), calendar).frame
    samples = _assemble_density_features(features, {day: 'offline_receipt_rehearsal'},
        bars_known=len(current), history_length=12, availability_basis=RECEIPT_BASIS)
    latest_receipt = pd.Timestamp(prefix['received_at_utc'])
    for row in samples['rows']:
        stamp = pd.Timestamp(row['target_time'])
        minutes = prefix['minutes'].loc[prefix['minutes'].symbol.eq(row['symbol']) &
            prefix['minutes'].timestamp.ge(stamp - pd.Timedelta(minutes=5)) & prefix['minutes'].timestamp.lt(stamp)]
        if len(minutes) != 5 or minutes.timestamp.nunique() != 5:
            raise DataContractError('Target lacks all five persisted source minutes')
        target_receipt = minutes.retrieved_at_utc.max()
        if target_receipt > observed or stamp > observed - pd.Timedelta(seconds=provider_delay_seconds):
            raise DataContractError('Unavailable target reached frozen density scoring')
        row.update(as_of=observed.isoformat(), target_received_at_utc=target_receipt.isoformat(),
            information_available_at=max(latest_receipt, stamp).isoformat(),
            data_revision=prefix['prefix_sha256'], historical_reference_basis=AVAILABILITY)
    config = bundle['manifest']['configurations'][model]
    variant = 'without_context' if config == 'E2' else 'with_context'
    inferred = infer_density(bundle['models'][variant], samples['X'], samples['y'], samples['context'],
        [r['symbol'] for r in samples['rows']], stable_prefix=True)
    return dict(base, state='DENSITY_READY' if samples['rows'] else 'INSUFFICIENT_HISTORY',
        rows=samples['rows'], scores=inferred['scores'].tolist(), residuals=inferred['residuals'].tolist(),
        exclusion_counts=samples['exclusion_counts'], score_duration_seconds=perf_counter() - started)
