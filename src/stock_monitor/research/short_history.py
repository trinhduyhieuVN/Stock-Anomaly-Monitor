"""Independent six-bar density candidate; never replaces the frozen 12-bar AI.

The shorter conditioning window is an operational coverage experiment, not a
claim of better anomaly detection. Source rights and independent labels remain
pending. Receipt scoring below does not emit or backdate live alerts.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from stock_monitor.features import FEATURE_ORDER, compute_features
from stock_monitor.io import DataContractError
from stock_monitor.realtime import closed_bar_readiness
from stock_monitor.research.density_model import ConditionalDensity
from stock_monitor.research.density_samples import AVAILABILITY, CONTEXT_ORDER, build_density_samples, _assemble_density_features
from stock_monitor.research.frozen_density import verify_pin
from stock_monitor.research.receipt_density import RECEIPT_BASIS, saved_poll_prefix
from stock_monitor.research.receipt_monitor import bound_history, _receipt_for
from stock_monitor.upgrade.contracts import utc_time

HISTORY = 6
SYMBOLS = ('FPT', 'HPG', 'VCB')


class ShortHistoryDensity(ConditionalDensity):
    """Reuse the registered GRU/head and training loss, not its saved weights."""
    def __init__(self):
        super().__init__(use_context=True)

    def forward(self, history, context, symbols):
        if history.ndim != 3 or tuple(history.shape[1:]) != (HISTORY, 4):
            raise DataContractError('Short-history density requires N x6x4')
        if context.shape != (len(history), 5) or symbols.shape != (len(history),):
            raise DataContractError('Short-history context/symbol shape mismatch')
        if symbols.dtype != torch.long or ((symbols < 0) | (symbols > 2)).any():
            raise DataContractError('Unknown short-history symbol')
        if not torch.isfinite(history).all() or not torch.isfinite(context).all():
            raise DataContractError('Nonfinite short-history input')
        _, hidden = self.encoder(history)
        state = torch.cat((hidden[-1], context, self.symbol_embedding(symbols)), dim=1)
        parameters = self.head(state)
        return parameters[:, :4], torch.nn.functional.softplus(parameters[:, 4:]) + .001


def build_short_samples(bars, calendar, splits, *, as_of):
    # Reuse all existing split/source/OHLCV/availability validation. The second
    # feature pass is intentional; no changes to the pinned 12-bar adapter.
    build_density_samples(bars, calendar, splits, as_of=as_of, availability_basis=AVAILABILITY)
    cutoff = pd.Timestamp(utc_time(as_of))
    scoped = bars.loc[bars.timestamp.le(cutoff)].copy()
    features = compute_features(scoped, calendar).frame
    date_split = {day: key for key, dates in splits.items() for day in dates}
    return _assemble_density_features(features, date_split, bars_known=len(scoped),
        history_length=HISTORY, availability_basis=AVAILABILITY)


def _shapes(X, context):
    if X.ndim != 3 or X.shape[1:] != (HISTORY, 4) or context.shape != (len(X), 5):
        raise DataContractError('Short-history preprocessing shape mismatch')


def fit_short_scalers(X, context, splits):
    _shapes(X, context)
    labels = np.asarray(splits)
    if labels.shape != (len(X),) or not np.isin(labels, ['train', 'validation', 'test']).all():
        raise DataContractError('Short-history split labels invalid')
    selected = labels == 'train'
    if not selected.any() or not np.isfinite(X[selected]).all() or not np.isfinite(context[selected]).all():
        raise DataContractError('Finite train-only short-history inputs required')
    return {'feature': StandardScaler().fit(X[selected].reshape(-1, 4)),
        'context': StandardScaler().fit(context[selected]), 'history_length': HISTORY,
        'fit_samples': int(selected.sum()),
        'rule': 'train conditioning histories only; weighted by window occurrence'}


def short_tensors(X, y, context, symbols, scalers):
    _shapes(X, context)
    if scalers.get('history_length') != HISTORY or y.shape != (len(X), 4) or len(symbols) != len(X):
        raise DataContractError('Short-history target/scaler mismatch')
    if any(s not in SYMBOLS for s in symbols):
        raise DataContractError('Unknown short-history symbol')
    history = scalers['feature'].transform(X.reshape(-1, 4)).reshape(-1, HISTORY, 4)
    target = scalers['feature'].transform(y)
    ctx = scalers['context'].transform(context)
    if any(not np.isfinite(v).all() for v in (history, target, ctx)):
        raise DataContractError('Nonfinite standardized short-history inputs')
    return (torch.tensor(history, dtype=torch.float32), torch.tensor(target, dtype=torch.float32),
        torch.tensor(ctx, dtype=torch.float32), torch.tensor([SYMBOLS.index(s) for s in symbols], dtype=torch.long))


def verify_short_bundle(bundle):
    verify_pin(bundle['root'], bundle['manifest_pin'])
    for record in bundle['manifest']['members'] + bundle['manifest']['input_pins']:
        verify_pin(bundle['root'], record)


def load_short_density(root, manifest_path, expected_sha256):
    root = Path(root).resolve()
    manifest_pin = {'path': str(manifest_path), 'sha256': expected_sha256}
    path = verify_pin(root, manifest_pin)
    if not path.is_relative_to(root / 'artifacts/research/short_history'):
        raise DataContractError('Short-history manifest outside registered namespace')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != 1 or manifest.get('history_length') != HISTORY or
            manifest.get('feature_order') != list(FEATURE_ORDER) or manifest.get('context_order') != list(CONTEXT_ORDER) or
            manifest.get('source_rights_status') != 'pending' or manifest.get('distributable') is not False or
            manifest.get('deployment_status') != 'research_candidate_not_default'):
        raise DataContractError('Short-history manifest contract mismatch')
    bundle = {'root': root, 'manifest': manifest, 'manifest_pin': manifest_pin}
    verify_short_bundle(bundle)
    files = {Path(v['path']).name: verify_pin(root, v) for v in manifest['members']}
    if not {'model.pt', 'scalers.joblib', 'thresholds.json', 'training.json', 'registration.json'} <= set(files):
        raise DataContractError('Short-history bundle incomplete')
    thresholds = json.loads(files['thresholds.json'].read_text(encoding='utf-8'))
    if set(thresholds) != set(SYMBOLS) or any(not np.isfinite(v['threshold']) or v['reference_count'] < 100 for v in thresholds.values()):
        raise DataContractError('Short-history validation thresholds invalid')
    model = ShortHistoryDensity().eval()
    model.load_state_dict(torch.load(files['model.pt'], weights_only=True, map_location='cpu'))
    bundle.update(model=model, scalers=joblib.load(files['scalers.joblib']), thresholds=thresholds)
    verify_short_bundle(bundle)
    return bundle


def infer_short_density(bundle, X, y, context, symbols):
    """Fixed batch geometry gives an exact prefix; padding is not an observation."""
    verify_short_bundle(bundle)
    _shapes(X, context)
    if len(X) == 0:
        return {'scores': np.empty(0), 'location': np.empty((0, 4)), 'scale': np.empty((0, 4))}
    batch = short_tensors(X, y, context, symbols, bundle['scalers'])
    locations, scales, scores = [], [], []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            h, target, ctx, names = (v[start:start+64] for v in batch)
            count = len(h)
            h = torch.cat((h, torch.zeros((64-count, HISTORY, 4))))
            target = torch.cat((target, torch.zeros((64-count, 4))))
            ctx = torch.cat((ctx, torch.zeros((64-count, 5))))
            names = torch.cat((names, torch.zeros(64-count, dtype=torch.long)))
            location, scale = bundle['model'](h, ctx, names)
            values = -torch.distributions.StudentT(5., location, scale).log_prob(target).sum(dim=1)
            locations.append(location[:count].numpy()); scales.append(scale[:count].numpy()); scores.append(values[:count].numpy())
    result = {'scores': np.concatenate(scores), 'location': np.concatenate(locations), 'scale': np.concatenate(scales)}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise DataContractError('Nonfinite short-history inference')
    verify_short_bundle(bundle)
    return result


def score_short_receipt(bundle, store_root, profile, *, symbols, observed_at_utc,
        provider_delay_seconds, provider_delay_evidence, history_evidence=None, stale_after_seconds=600):
    """Verified closed-bar scoring seam only, not a new live session/alert journal."""
    started = perf_counter()
    process_started = datetime.now(timezone.utc).isoformat()
    verify_short_bundle(bundle)
    observed = pd.Timestamp(utc_time(pd.Timestamp(observed_at_utc).isoformat()))
    store = Path(store_root).resolve()
    if not store.is_relative_to(bundle['root']):
        raise DataContractError('Short-history receipt store escapes project')
    historical = bound_history(bundle, profile, history_evidence)
    prefix = saved_poll_prefix(store, as_of=observed.isoformat(), symbols=symbols)
    ready = closed_bar_readiness(prefix['minutes'], profile, symbols=symbols, observed_at_utc=observed.isoformat(),
        provider_delay_seconds=provider_delay_seconds, provider_delay_evidence=provider_delay_evidence,
        stale_after_seconds=stale_after_seconds, project_root=bundle['root'], delay_measured_live=False)
    base = {'schema_version': 1, 'model': 'gru_context_6_candidate', 'history_length': HISTORY,
        'source_mode': 'OFFLINE_RECEIPT_REHEARSAL', 'live': False, 'live_certified': False,
        'provider_calls': 0, 'model_fits': 0, 'alerts_emitted': False, 'live_receive_to_score_seconds': None,
        'manifest_sha256': bundle['manifest_pin']['sha256'], 'source_rights_status': 'pending', 'distributable': False,
        'received_at_utc': prefix['received_at_utc'], 'input_cutoff_utc': observed.isoformat(),
        'process_started_at_utc': process_started, 'readiness': ready.status,
        'poll_pins': prefix['poll_pins'], 'prefix_sha256': prefix['prefix_sha256'], 'latest_scores': [],
        'availability_basis': RECEIPT_BASIS, 'historical_reference_basis': AVAILABILITY,
        'deployment_status': 'research_candidate_not_default'}
    state = 'BLOCKED_READINESS'
    if ready.status['scoring_allowed'] and not ready.status['gap_count']:
        day = ready.status['calendar_date']
        state = 'BLOCKED_BEFORE_MODEL_FREEZE'
        if day > bundle['manifest']['validation_end'][:10]:
            past = historical.loc[historical.symbol.isin(symbols) & historical.session_id.lt(day)]
            features = compute_features(pd.concat([past, ready.bars], ignore_index=True), profile['calendar']).frame
            samples = _assemble_density_features(features, {day: 'receipt_candidate'}, bars_known=len(ready.bars),
                history_length=HISTORY, availability_basis=RECEIPT_BASIS)
            inferred = infer_short_density(bundle, samples['X'], samples['y'], samples['context'], [r['symbol'] for r in samples['rows']])
            for i, row in enumerate(samples['rows']):
                if row['target_time'] != ready.status['latest_by_symbol'].get(row['symbol']):
                    continue  # Never make a retrospective backfill into new alerts.
                receipt = _receipt_for(prefix['minutes'], row['symbol'], row['target_time'])
                stamp = pd.Timestamp(row['target_time'])
                if pd.Timestamp(receipt) > observed or stamp > observed-pd.Timedelta(seconds=provider_delay_seconds):
                    raise DataContractError('Unavailable short-history target')
                score = float(inferred['scores'][i])
                threshold = bundle['thresholds'][row['symbol']]['threshold']
                base['latest_scores'].append({**row, 'target_received_at_utc': receipt, 'score': score,
                    'threshold': threshold, 'alert_candidate': score > threshold,
                    'score_semantics': 'validation-calibrated point NLL, not a fraud probability'})
            base.update(eligible_endpoints=len(samples['rows']), exclusion_counts=samples['exclusion_counts'])
            state = 'DENSITY_READY' if len(base['latest_scores']) == len(symbols) else 'INSUFFICIENT_HISTORY'
    verify_short_bundle(bundle)
    return {**base, 'state': state, 'score_duration_seconds': perf_counter()-started,
        'process_finished_at_utc': datetime.now(timezone.utc).isoformat(), 'score_finished_at_utc': None}
