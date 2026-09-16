"""Regression tests exercise installed public imports, not source-path aliases."""
import subprocess
import sys

import numpy as np
import pytest
import torch
from dmi import estimators

NAMES = estimators.__all__


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(name, **kwargs):
    config = dict(max_n_steps=2, batch_size=32)
    if name in ("MINEEstimator", "DIMEEstimator"):
        config["early_stopping"] = False
    if name == "MINDEEstimator":
        config.update(mc_iter=1, use_ema=False)
    if name == "MMGEstimator":
        config.update(use_ema=False)
    config.update(kwargs)
    return getattr(estimators, name)(**config)


def data(n=17):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 1))
    y = np.column_stack([x[:, 0] + rng.normal(size=n), rng.normal(size=n)])
    return x, y


@pytest.mark.parametrize("name", NAMES)
def test_small_sample_unequal_dimensions_and_inference(name):
    x, y = data()
    model = make(name)
    assert model.fit(x, y) is model
    net = next(getattr(model, key) for key in ("critic", "model", "score") if getattr(model, key, None) is not None)
    before = {key: value.clone() for key, value in net.state_dict().items()}
    value = model.estimate(*data(8))
    assert isinstance(value, float)
    assert np.isfinite(value)
    assert all(torch.equal(before[key], value) for key, value in net.state_dict().items())
    if hasattr(model, "trainer"):
        assert model.trainer.global_step == 2
    with pytest.raises(ValueError, match="shape"):
        model.estimate(x, np.ones((len(x), 3)))


@pytest.mark.parametrize("name", NAMES)
def test_input_errors_and_unfitted(name):
    x, y = data()
    model = make(name)
    with pytest.raises(RuntimeError, match="fit"):
        model.estimate(x, y)
    with pytest.raises(ValueError, match="same number"):
        model.fit(x[:-1], y)
    with pytest.raises(ValueError, match="finite"):
        model.fit(x * np.nan, y)
    with pytest.raises(ValueError, match="shape"):
        model.fit(x[:, 0], y)
    with pytest.raises(ValueError, match="two paired"):
        model.fit(x[:1], y[:1])
    with pytest.raises(ValueError, match="both"):
        model.fit(x, y, X_val=x)
    with pytest.raises(ValueError, match="batch_size"):
        make(name, batch_size=1).fit(x, y)


def test_cpc_temperature_on_uninformative_scores():
    # Identical logits represent chance predictions at any temperature.
    for temperature in (0.1, 0.5, 1.0, 2.0):
        result = estimators.CPCEstimator._infonce_lower_bound(torch.ones(4, 4), temperature)
        assert result.item() == pytest.approx(0, abs=1e-6)


def test_mine_epoch_only_and_checkpoint(tmp_path):
    model = make("MINEEstimator", max_n_steps=None, max_epochs=1, create_checkpoint=True)
    model.trainer_config['checkpoint']['dirpath'] = str(tmp_path)
    model.fit(*data())
    assert list(tmp_path.rglob('*.ckpt'))


def test_minde_uses_training_scalers(monkeypatch):
    x, y = data()
    model = make("MINDEEstimator")
    model.fit(x, y, x + 10, y + 20)
    np.testing.assert_allclose(model.test_samples['x0'].numpy(), model.x_scaler_.transform(x + 10), rtol=1e-5)
    original_mean = model.x_scaler_.mean_.copy()
    captured = {}
    def capture(data):
        captured.update(data)
        return 0.0, 0.0
    monkeypatch.setattr(model, 'compute_mi', capture)
    model.estimate(x + 30, y + 40)
    np.testing.assert_array_equal(model.x_scaler_.mean_, original_mean)
    np.testing.assert_allclose(captured['x0'].numpy(), model.x_scaler_.transform(x + 30), rtol=1e-5)


def test_dime_singleton_tail_is_not_used():
    model = make("DIMEEstimator", batch_size=8, max_n_steps=3)
    model.fit(*data(17))
    assert np.isfinite(model.estimate(*data(8)))


def test_lightweight_public_import():
    result = subprocess.run([sys.executable, '-c',
        "import sys, dmi; assert 'torch' not in sys.modules; "
        "assert 'lightning' not in sys.modules; "
        "from dmi.estimators import CPCEstimator; assert 'lightning' not in sys.modules"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["CPCEstimator", "DoEEstimator", "NWJEstimator", "SMILEEstimator"])
def test_early_stop_returns_fitted_model(name):
    model = make(name, max_n_steps=5)
    assert model.fit(*data(), early_stopping=True,
                     early_stopping_patience=1, early_stopping_min_delta=1e9) is model
    assert np.isfinite(model.estimate(*data(8)))
