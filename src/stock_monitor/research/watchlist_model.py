"""Separate five-symbol GRU6 weights and calibration, with frozen inference."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from stock_monitor.io import DataContractError
from stock_monitor.research.density_model import ConditionalDensity
from stock_monitor.research.frozen_density import verify_pin
from stock_monitor.research.short_history import fit_short_scalers
from stock_monitor.research.watchlist_history import SYMBOLS

MODEL = 'watchlist_gru6'


class WatchlistDensity(ConditionalDensity):
    def __init__(self):
        super().__init__(use_context=True)
        self.symbol_embedding = nn.Embedding(len(SYMBOLS), 4)

    def forward(self, history, context, symbols):
        if (history.ndim != 3 or tuple(history.shape[1:]) != (6, 4) or
                context.shape != (len(history), 5) or symbols.shape != (len(history),)):
            raise DataContractError('Watchlist model requires N x6x4 and N x5 context')
        if symbols.dtype != torch.long or ((symbols < 0) | (symbols >= len(SYMBOLS))).any():
            raise DataContractError('Unknown watchlist symbol index')
        if not torch.isfinite(history).all() or not torch.isfinite(context).all():
            raise DataContractError('Nonfinite watchlist input')
        _, hidden = self.encoder(history)
        parameters = self.head(torch.cat((hidden[-1], context, self.symbol_embedding(symbols)), dim=1))
        return parameters[:, :4], nn.functional.softplus(parameters[:, 4:]) + .001


def fit_scalers(X, context, splits):
    return {**fit_short_scalers(X, context, splits), 'symbol_order': list(SYMBOLS)}


def tensors(X, y, context, symbols, scalers):
    if (scalers.get('history_length') != 6 or scalers.get('symbol_order') != list(SYMBOLS) or
            X.shape != (len(X), 6, 4) or y.shape != (len(X), 4) or context.shape != (len(X), 5) or len(symbols) != len(X)):
        raise DataContractError('Watchlist scaler/input contract mismatch')
    if any(s not in SYMBOLS for s in symbols): raise DataContractError('Unregistered watchlist symbol')
    history = scalers['feature'].transform(X.reshape(-1, 4)).reshape(-1, 6, 4)
    targets = scalers['feature'].transform(y); ctx = scalers['context'].transform(context)
    if any(not np.isfinite(v).all() for v in (history, targets, ctx)): raise DataContractError('Nonfinite standardized input')
    return (torch.tensor(history, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32),
        torch.tensor(ctx, dtype=torch.float32), torch.tensor([SYMBOLS.index(s) for s in symbols], dtype=torch.long))


def verify_bundle(bundle):
    verify_pin(bundle['root'], bundle['manifest_pin'])
    for item in bundle['manifest']['members'] + bundle['manifest']['input_pins']: verify_pin(bundle['root'], item)


def load_bundle(root, manifest_pin):
    root = Path(root).resolve(); path = verify_pin(root, manifest_pin)
    if not path.is_relative_to(root / 'artifacts/research/watchlist_model'):
        raise DataContractError('Watchlist model outside registered namespace')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('model_id') != MODEL or manifest.get('symbols') != list(SYMBOLS) or
            manifest.get('history_length') != 6 or manifest.get('feature_order') != ['log_return','range_ratio','body_ratio','volume_surprise'] or
            manifest.get('context_order') != ['lagged_volatility','past_trend','lagged_relative_volume','time_sin','time_cos'] or
            manifest.get('source_rights_status') != 'pending' or manifest.get('distributable') is not False):
        raise DataContractError('Watchlist model manifest mismatch')
    bundle = {'root': root, 'manifest_pin': manifest_pin, 'manifest': manifest}; verify_bundle(bundle)
    files = {Path(p['path']).name: verify_pin(root, p) for p in manifest['members']}
    thresholds = json.loads(files['thresholds.json'].read_text(encoding='utf-8'))
    if set(thresholds) != set(SYMBOLS) or any(not np.isfinite(v['threshold']) or v['reference_count'] < 100 for v in thresholds.values()):
        raise DataContractError('Invalid watchlist calibration')
    model = WatchlistDensity().eval(); model.load_state_dict(torch.load(files['model.pt'], weights_only=True, map_location='cpu'))
    bundle.update(model=model, scalers=joblib.load(files['scalers.joblib']), thresholds=thresholds)
    verify_bundle(bundle); return bundle


def infer(bundle, X, y, context, symbols):
    verify_bundle(bundle)
    if len(X) == 0: return np.empty(0)
    batch = tensors(X, y, context, symbols, bundle['scalers']); scores = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            h, target, ctx, names = (v[start:start+64] for v in batch); count = len(h)
            h = torch.cat((h, torch.zeros((64-count, 6, 4))))
            target = torch.cat((target, torch.zeros((64-count, 4))))
            ctx = torch.cat((ctx, torch.zeros((64-count, 5))))
            names = torch.cat((names, torch.zeros(64-count, dtype=torch.long)))
            location, scale = bundle['model'](h, ctx, names)
            values = -torch.distributions.StudentT(5., location, scale).log_prob(target).sum(dim=1)
            scores.append(values[:count].numpy())
    result = np.concatenate(scores)
    if not np.isfinite(result).all(): raise DataContractError('Nonfinite watchlist score')
    verify_bundle(bundle); return result
