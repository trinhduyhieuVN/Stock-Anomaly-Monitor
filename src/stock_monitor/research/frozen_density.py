"""Hash-checked inference only for explicitly pinned local density experiments."""
from pathlib import Path

import joblib
import numpy as np
import torch

from stock_monitor.io import DataContractError, sha256
from stock_monitor.research.density_model import ConditionalDensity, tensors


def verify_pin(root, record):
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha256(path) != record['sha256']:
        raise DataContractError('Frozen density checksum/path mismatch: ' + record['path'])
    return path


def load_density(root, files, *, variant):
    if variant not in {'with_context', 'without_context'}:
        raise DataContractError('Unknown registered density variant')
    paths = {Path(v['path']).name: verify_pin(root, v) for v in files}
    if not {'model.pt', 'scalers.joblib'}.issubset(paths):
        raise DataContractError('Incomplete density pins')
    model = ConditionalDensity(use_context=variant == 'with_context').eval()
    model.load_state_dict(torch.load(paths['model.pt'], weights_only=True, map_location='cpu'))
    return {'model': model, 'scalers': joblib.load(paths['scalers.joblib']), 'pins': files, 'root': root}


def infer_density(bundle, X, y, context, symbols, *, stable_prefix=False):
    for record in bundle['pins']:
        verify_pin(bundle['root'], record)
    if len(X) == 0:
        return {'residuals': np.empty((0, 4)), 'scores': np.empty(0), 'location': np.empty((0, 4)), 'scale': np.empty((0, 4))}
    batch = tensors(X, y, context, symbols, bundle['scalers'])
    locations, scales, scores = [], [], []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            history, target, ctx, names = (v[start:start + 64] for v in batch)
            count = len(history)
            if stable_prefix and count < 64:
                # Fixed batch geometry prevents BLAS rounding from rewriting an
                # immutable initial decision as later prefix rows arrive. Padding
                # is not observed data and is never scored or returned.
                history = torch.cat((history, torch.zeros((64 - count, 12, 4))))
                target = torch.cat((target, torch.zeros((64 - count, 4))))
                ctx = torch.cat((ctx, torch.zeros((64 - count, 5))))
                names = torch.cat((names, torch.zeros(64 - count, dtype=torch.long)))
            location, scale = bundle['model'](history, ctx, names)
            values = -torch.distributions.StudentT(5., location, scale).log_prob(target).sum(dim=1)
            locations.append(location[:count].numpy()); scales.append(scale[:count].numpy()); scores.append(values[:count].numpy())
    location, scale = np.concatenate(locations), np.concatenate(scales)
    residuals = (batch[1].numpy().astype(np.float64) - location) / scale
    if not np.isfinite(residuals).all():
        raise DataContractError('Nonfinite frozen density residuals')
    return {'residuals': residuals, 'scores': np.concatenate(scores), 'location': location, 'scale': scale}
