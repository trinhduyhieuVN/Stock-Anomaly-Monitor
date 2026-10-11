"""Registered small conditional density network, isolated from saved models."""
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.distributions import StudentT

from stock_monitor.io import DataContractError


class ConditionalDensity(nn.Module):
    def __init__(self, *, use_context=True):
        super().__init__()
        self.use_context = use_context
        self.encoder = nn.GRU(4, 32, num_layers=1, batch_first=True)
        self.symbol_embedding = nn.Embedding(3, 4) if use_context else None
        self.head = nn.Linear(32 + (9 if use_context else 0), 8)

    def forward(self, history, context, symbols):
        if history.ndim != 3 or tuple(history.shape[1:]) != (12, 4):
            raise DataContractError('Density requires history N x12x4')
        if context.shape != (len(history), 5) or symbols.shape != (len(history),):
            raise DataContractError('Density context/symbol shape mismatch')
        if symbols.dtype != torch.long or ((symbols < 0) | (symbols > 2)).any():
            raise DataContractError('Unknown registered density symbol')
        if not torch.isfinite(history).all() or not torch.isfinite(context).all():
            raise DataContractError('Nonfinite density input')
        _, hidden = self.encoder(history)
        state = hidden[-1]
        if self.use_context:
            state = torch.cat((state, context, self.symbol_embedding(symbols)), dim=1)
        parameters = self.head(state)
        location = parameters[:, :4]
        scale = nn.functional.softplus(parameters[:, 4:]) + .001
        return location, scale

    def distribution(self, history, context, symbols):
        location, scale = self(history, context, symbols)
        return StudentT(5., location, scale)


def fit_scalers(X, context, splits):
    labels = np.asarray(splits)
    if X.ndim != 3 or X.shape[1:] != (12, 4) or context.shape != (len(X), 5) or labels.shape != (len(X),):
        raise DataContractError('Scaler shape mismatch')
    selected = labels == 'train'
    if not selected.any():
        raise DataContractError('No train samples for scaler')
    if not np.isfinite(X[selected]).all() or not np.isfinite(context[selected]).all():
        raise DataContractError('Nonfinite train inputs')
    return {'feature': StandardScaler().fit(X[selected].reshape(-1, 4)),
            'context': StandardScaler().fit(context[selected]), 'fit_samples': int(selected.sum()),
            'rule': 'train conditioning histories only; overlapping observations weighted by window occurrence'}


def tensors(X, y, context, symbols, scalers):
    if y.shape != (len(X), 4) or len(symbols) != len(X):
        raise DataContractError('Target/symbol count mismatch')
    names = {'FPT': 0, 'HPG': 1, 'VCB': 2}
    if any(name not in names for name in symbols):
        raise DataContractError('Unregistered density symbol')
    history = scalers['feature'].transform(X.reshape(-1, 4)).reshape(-1, 12, 4)
    targets = scalers['feature'].transform(y)
    ctx = scalers['context'].transform(context)
    if any(not np.isfinite(value).all() for value in (history, targets, ctx)):
        raise DataContractError('Nonfinite standardized density inputs')
    return (torch.tensor(history, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32),
            torch.tensor(ctx, dtype=torch.float32), torch.tensor([names[s] for s in symbols], dtype=torch.long))


def nll(model, batch):
    history, target, context, symbols = batch
    values = -model.distribution(history, context, symbols).log_prob(target)
    if not torch.isfinite(values).all():
        raise DataContractError('Nonfinite density loss')
    return values.sum(dim=1)


def cpu_batch_benchmark(model, batch):
    started = time.perf_counter()
    loss = nll(model, batch).mean()
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise DataContractError('Nonfinite density gradients')
    model.zero_grad(set_to_none=True)
    return {'seconds': time.perf_counter() - started, 'batch_size': len(batch[0]),
            'loss': float(loss.detach()), 'finite_gradients': True, 'optimizer_steps': 0}


def train_density(model, train, validation, *, seed=42, epochs=30, patience=5, seconds_limit=21600):
    if len(train[0]) == 0 or len(validation[0]) == 0 or not 1 <= epochs <= 30 or not 1 <= patience <= 5:
        raise DataContractError('Registered training limits or sample counts invalid')
    if not 0 < seconds_limit <= 21600:
        raise DataContractError('Registered CPU wall-time limit invalid')
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    started = time.perf_counter()
    best, best_state, stale, history = float('inf'), None, 0, []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train[0]), generator=generator)
        total = 0.
        for index in permutation.split(64):
            if time.perf_counter() - started >= seconds_limit:
                raise DataContractError('Density CPU wall-time budget exhausted')
            batch = tuple(value[index] for value in train)
            optimizer.zero_grad(set_to_none=True)
            loss = nll(model, batch).mean()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise DataContractError('Nonfinite train gradient')
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        with torch.no_grad():
            valid = float(torch.cat([nll(model, tuple(value[i:i + 64] for value in validation))
                                      for i in range(0, len(validation[0]), 64)]).mean())
        history.append({'epoch': epoch + 1, 'train_nll': total / len(train[0]), 'validation_nll': valid})
        if valid < best:
            best, stale = valid, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise DataContractError('No finite density checkpoint')
    model.load_state_dict(best_state)
    model.eval()
    return {'epochs_run': len(history), 'best_validation_nll': best,
            'best_epoch': min(history, key=lambda row: row['validation_nll'])['epoch'],
            'wall_seconds': time.perf_counter() - started, 'history': history,
            'seed': seed, 'train_samples': len(train[0]), 'validation_samples': len(validation[0])}
