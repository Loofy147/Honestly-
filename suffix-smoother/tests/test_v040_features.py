import pytest
import numpy as np
import os
import json
from suffix_smoother import SuffixSmoother, SuffixConfig

def test_pruning_min_count():
    cfg = SuffixConfig(n_classes=3, max_suffix_length=2)
    model = SuffixSmoother(cfg)
    # Train with some patterns
    # (1, 2) -> 0 (twice)
    # (2, 2) -> 1 (once)
    model.train([((1, 2), 0), ((1, 2), 0), ((2, 2), 1)])

    initial_nodes = len(model._nodes)
    assert initial_nodes > 0

    # Prune nodes with count < 2
    res = model.prune(min_count=2)
    assert res["removed"] > 0
    assert len(model._nodes) < initial_nodes
    # Verify (2, 2) suffix is gone but (1, 2) remains
    assert (2, 2) not in model._nodes
    assert (1, 2) in model._nodes

def test_calibration_curve():
    probs = [0.1, 0.2, 0.7, 0.8, 0.9]
    labels = [0, 0, 1, 1, 1]
    curve = SuffixSmoother.calibration_curve(probs, labels, n_bins=5)
    assert len(curve) > 0
    for entry in curve:
        assert "bin" in entry
        assert "confidence" in entry
        assert "accuracy" in entry
        assert "count" in entry

def test_temperature_scaling():
    cfg = SuffixConfig(n_classes=3, temperature=1.0)
    model = SuffixSmoother(cfg)
    model.train([((1, 1), 0), ((1, 1), 0), ((1, 1), 1)])

    p1 = model.predict_proba_batch([(1, 1)])[0]

    model.cfg.temperature = 2.0
    p2 = model.predict_proba_batch([(1, 1)])[0]

    # Higher temperature should make distribution more uniform
    assert np.std(p2) < np.std(p1)

    model.cfg.temperature = 0.5
    p3 = model.predict_proba_batch([(1, 1)])[0]
    # Lower temperature should make distribution sharper
    assert np.std(p3) > np.std(p1)

def test_fit_temperature():
    cfg = SuffixConfig(n_classes=2)
    model = SuffixSmoother(cfg)
    # Create a miscalibrated situation
    model.train([((1,), 0)] * 10 + [((1,), 1)] * 5)

    val_data = [((1,), 0)] * 10 + [((1,), 1)] * 5
    best_t = model.fit_temperature(val_data)
    assert best_t > 0
    assert model.cfg.temperature == best_t

def test_json_serialization(tmp_path):
    path = tmp_path / "model.json"
    cfg = SuffixConfig(n_classes=3, smoothing_method="witten-bell")
    model = SuffixSmoother(cfg)
    model.train([((1, 2), 0), ((3, 4), 1)])

    model.to_json(str(path))
    assert os.path.exists(path)

    # Load back
    model2 = SuffixSmoother.from_json(str(path))
    assert model2.n_classes == model.n_classes
    assert model2.training_samples == model.training_samples
    assert model2.cfg.smoothing_method == model.cfg.smoothing_method

    # Test prediction parity
    p1 = model.predict_proba_batch([((1, 2),)])[0]
    p2 = model2.predict_proba_batch([((1, 2),)])[0]
    np.testing.assert_allclose(p1, p2)
