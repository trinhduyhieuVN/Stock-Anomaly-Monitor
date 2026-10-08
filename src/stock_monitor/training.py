"""Deterministic CPU training loop with validation early stopping."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .io import DataContractError
from .model import LSTMAutoencoder


@dataclass
class TrainingResult:
    history: list[dict]
    best_epoch: int
    best_validation_loss: float
    epochs_ran: int
    stopped_early: bool
    best_state: dict[str, torch.Tensor]
    device: str
    seed: int
    optimizer: str = "Adam"
    loss: str = "MSELoss_full_window"


def set_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataContractError("Training seed must be an integer")
    np.random.seed(seed)
    torch.manual_seed(seed)


def _window_tensor(values, input_size, label):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or len(array) == 0 or array.shape[1] < 1 or array.shape[2] != input_size:
        raise DataContractError(f"{label} must have shape [samples, window, {input_size}]")
    if not np.isfinite(array).all():
        raise DataContractError(f"{label} must contain only finite values")
    return torch.from_numpy(array)


def _mean_loss(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, samples = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for (batch,) in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(batch)
            samples += len(batch)
    return total / samples


def fit_autoencoder(train_values, validation_values, config):
    device_name = config.get("device", "cpu")
    if device_name != "cpu":
        raise DataContractError("Prompt 7 training is restricted to CPU")
    seed = int(config["seed"])
    set_seed(seed)
    input_size = int(config.get("input_size", np.asarray(train_values).shape[-1]))
    model = LSTMAutoencoder(input_size=input_size, hidden_size=int(config["hidden_size"]),
                            latent_size=int(config["latent_size"]), num_layers=int(config["num_layers"]))
    train = _window_tensor(train_values, input_size, "Train windows")
    validation = _window_tensor(validation_values, input_size, "Validation windows")
    if train.shape[1:] != validation.shape[1:]:
        raise DataContractError("Train and validation window shapes must match")
    batch_size, max_epochs, patience = int(config["batch_size"]), int(config["max_epochs"]), int(config["patience"])
    learning_rate = float(config["learning_rate"])
    if min(batch_size, max_epochs, patience) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise DataContractError("Invalid optimizer or early-stopping configuration")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(train), batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(TensorDataset(validation), batch_size=batch_size, shuffle=False)
    device = torch.device(device_name)
    model.to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_state = [], None
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        train_loss = _mean_loss(model, train_loader, criterion, device, optimizer)
        validation_loss = _mean_loss(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.to(device)
    result = TrainingResult(history, best_epoch, best_loss, len(history), len(history) < max_epochs,
                            best_state, device_name, seed)
    return model, result


def reconstruct(model, values, *, device="cpu"):
    if device != "cpu":
        raise DataContractError("Prompt 7 inference is restricted to CPU")
    tensor = _window_tensor(values, model.input_size, "Inference windows").to(device)
    model.to(device).eval()
    with torch.no_grad():
        return model(tensor).cpu().numpy()
