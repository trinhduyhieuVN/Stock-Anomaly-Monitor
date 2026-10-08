from io import BytesIO

import numpy as np
import pytest
import torch

from stock_monitor.io import DataContractError
from stock_monitor.model import LSTMAutoencoder
from stock_monitor.training import fit_autoencoder, reconstruct


CONFIG = {"input_size": 4, "hidden_size": 32, "latent_size": 8, "num_layers": 1,
          "batch_size": 8, "max_epochs": 2, "learning_rate": .001, "patience": 1,
          "seed": 42, "device": "cpu"}


def test_lstm_autoencoder_shape_architecture_and_device():
    model = LSTMAutoencoder(4, 32, 8, 1)
    values = torch.randn(5, 6, 4)
    output = model(values)
    assert output.shape == values.shape
    assert model.encoder.batch_first and model.decoder.batch_first
    assert model.encoder.hidden_size == model.decoder.hidden_size == 32
    assert model.to_latent.out_features == model.decoder.input_size == 8
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_lstm_rejects_wrong_feature_shape_and_nonfinite_values():
    model = LSTMAutoencoder()
    with pytest.raises(DataContractError, match="shape"):
        model(torch.zeros(2, 6, 3))
    values = torch.zeros(2, 6, 4)
    values[0, 0, 0] = float("nan")
    with pytest.raises(DataContractError, match="finite"):
        model(values)


def test_training_uses_full_windows_and_reload_is_identical():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(24, 6, 4)).astype(np.float32)
    validation = rng.normal(size=(12, 6, 4)).astype(np.float32)
    model, result = fit_autoencoder(train, validation, CONFIG)
    assert 1 <= result.best_epoch <= result.epochs_ran <= 2
    assert result.device == "cpu" and result.seed == 42
    assert result.optimizer == "Adam" and result.loss == "MSELoss_full_window"
    assert all(np.isfinite(row["train_loss"]) and np.isfinite(row["validation_loss"]) for row in result.history)
    expected = reconstruct(model, validation)
    persisted = BytesIO()
    torch.save(result.best_state, persisted)
    persisted.seek(0)
    reloaded = LSTMAutoencoder(4, 32, 8, 1)
    reloaded.load_state_dict(torch.load(persisted, map_location="cpu", weights_only=True))
    np.testing.assert_array_equal(reconstruct(reloaded, validation), expected)
