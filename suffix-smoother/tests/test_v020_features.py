import pytest
import numpy as np
from suffix_smoother import SuffixSmoother, SuffixConfig

def test_jelinek_mercer():
    config = SuffixConfig(smoothing_method="jelinek-mercer", smoothing_lambda=0.5, n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0)] * 100)
    dist = smoother.predict_distribution((1,))
    assert abs(dist[0] - 0.75) < 1e-6

def test_witten_bell():
    config = SuffixConfig(smoothing_method="witten-bell", n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0)] * 10 + [((1,), 1)] * 10)
    dist = smoother.predict_distribution((1,))
    assert abs(dist[0] - 0.5) < 1e-6

    # After 80 more of label 0: N=100, T=2
    # P(0|1) = 90/102 + (2/102)*0.5 = 91/102 approx 0.8921568
    smoother.train([((1,), 0)] * 80)
    dist = smoother.predict_distribution((1,))
    assert abs(dist[0] - 91/102) < 1e-6

def test_kneser_ney_root_continuation():
    config = SuffixConfig(smoothing_method="kneser-ney", n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0), ((2,), 1), ((3,), 1)])
    dist = smoother.predict_distribution(())

    # Root is smoothed continuation distribution
    # alpha=0.01, contexts: {0: [(1,)], 1: [(2,), (3,)]}
    # count(0)=1, count(1)=2, total=3
    # P(0) = (1+0.01)/(3 + 2*0.01) = 0.375 approx 0.334437
    assert abs(dist[0] - 1.5/4.0) < 1e-6

def test_conformal_prediction():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0)] * 7 + [((1,), 1)] * 3)

    cal_data = [((1,), 1)] * 10 + [((1,), 0)] * 10
    result = smoother.calibrate(cal_data)
    assert smoother.is_calibrated

    pset_95 = smoother.predict_set((1,), coverage=0.95)
    assert len(pset_95["labels"]) == 2

def test_streaming_train_one():
    smoother = SuffixSmoother(SuffixConfig(n_classes=2))
    smoother.train_one((1, 2), 0)
    assert smoother.training_samples == 1
    assert (1, 2) in smoother._nodes
    assert (2,) in smoother._nodes
    dist = smoother.predict_distribution((1, 2))
    assert dist[0] > 0.5
