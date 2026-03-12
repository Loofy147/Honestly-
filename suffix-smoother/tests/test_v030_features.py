import pytest
import numpy as np
from suffix_smoother import SuffixSmoother, SuffixConfig

def test_merge():
    config = SuffixConfig(n_classes=2, max_suffix_length=2)
    m1 = SuffixSmoother(config); m1.train([((1, 2), 0)])
    m2 = SuffixSmoother(config); m2.train([((3, 4), 1)])
    merged = SuffixSmoother.merge(m1, m2)
    assert merged.predict((1, 2))[0] == 0
    assert merged.predict((3, 4))[0] == 1
    assert merged.training_samples == 2

def test_predict_batch():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1, 2), 0), ((3, 4), 1)])
    seqs = [(1, 2), (3, 4)]
    batch = smoother.predict_batch(seqs)
    single = [smoother.predict(s) for s in seqs]
    for b, s in zip(batch, single):
        assert b[0] == s[0]
        assert abs(b[1] - s[1]) < 1e-10

def test_feature_importance():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1, 1), 0)] * 10 + [((2, 2), 1)] * 10)
    importance = smoother.feature_importance(top_n=5)
    assert len(importance) > 0
    assert "kl" in importance[0]

def test_compare():
    test_data = [((1, 2), 0), ((3, 4), 1)]
    m_wb = SuffixSmoother(SuffixConfig(n_classes=2, smoothing_method="witten-bell"))
    m_wb.train(test_data)
    m_kn = SuffixSmoother(SuffixConfig(n_classes=2, smoothing_method="kneser-ney"))
    m_kn.train(test_data)
    report = SuffixSmoother.compare([("WB", m_wb), ("KN", m_kn)], test_data)
    assert len(report) == 2

def test_prune():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1, 1), 0)] * 10 + [((2, 2), 0), ((2, 2), 1)])
    stats = smoother.prune(min_kl=0.1)
    assert stats["removed"] > 0

def test_conformal_aps():
    config = SuffixConfig(n_classes=5)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), i) for i in range(5)] * 10)
    smoother.calibrate([((1,), i) for i in range(5)], score_type="aps")
    pset = smoother.predict_set((1,), coverage=0.9)
    assert len(pset["labels"]) >= 1

def test_prune_to_budget():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((i,), 0) for i in range(10)])
    stats = smoother.prune_to_budget(max_nodes=5)
    assert smoother.n_nodes == 5

def test_update_calibration():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0)] * 5)
    smoother.update_calibration((1,), 0)
    assert smoother.is_calibrated

def test_score():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0), ((2,), 1)])
    acc = smoother.score([((1,), 0), ((2,), 1)])
    assert acc == 1.0

def test_top_k():
    config = SuffixConfig(n_classes=5)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0)] * 10 + [((1,), 1)] * 5 + [((1,), 2)] * 2)

    top3 = smoother.predict_top_k((1,), k=3)
    assert len(top3) == 3
    assert top3[0][0] == 0
    assert top3[1][0] == 1
    assert top3[2][0] == 2

def test_jm_optimization():
    config = SuffixConfig(smoothing_method="jelinek-mercer", n_classes=2)
    smoother = SuffixSmoother(config)
    data = [((1,), 0), ((2,), 1)]
    smoother.train(data)

    best_l = smoother.optimize_jm_lambda(data)
    assert 0.1 <= best_l <= 0.9
