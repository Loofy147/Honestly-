import pytest
import numpy as np
from suffix_smoother import SuffixSmoother, SuffixConfig

def test_merge():
    config = SuffixConfig(n_classes=2, max_suffix_length=2)
    m1 = SuffixSmoother(config)
    m1.train([((1, 2), 0)])

    m2 = SuffixSmoother(config)
    m2.train([((3, 4), 1)])

    merged = SuffixSmoother.merge(m1, m2)

    # Check predictions from both
    assert merged.predict((1, 2))[0] == 0
    assert merged.predict((3, 4))[0] == 1
    assert merged.training_samples == 2
    assert merged.n_nodes >= 4 # (1,2), (2,), (3,4), (4,)

def test_predict_batch():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    data = [((1, 2), 0), ((3, 4), 1), ((1, 2), 0)]
    smoother.train(data)

    seqs = [(1, 2), (3, 4), (5, 6)]
    batch_results = smoother.predict_batch(seqs)

    single_results = [smoother.predict(s) for s in seqs]

    assert len(batch_results) == len(seqs)
    for b, s in zip(batch_results, single_results):
        assert b[0] == s[0]
        assert abs(b[1] - s[1]) < 1e-10

def test_feature_importance():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    # Suffix (1,1) always leads to 0, (2,2) always to 1
    smoother.train([((1, 1), 0)] * 10 + [((2, 2), 1)] * 10)

    importance = smoother.feature_importance(top_n=5)
    assert len(importance) > 0
    assert "suffix" in importance[0]
    assert "kl_divergence" in importance[0]

    # (1,1) or (2,2) should be at the top
    top_suffixes = [feat["suffix"] for feat in importance]
    assert (1, 1) in top_suffixes or (2, 2) in top_suffixes

def test_compare():
    test_data = [((1, 2), 0), ((3, 4), 1)]
    config_wb = SuffixConfig(smoothing_method="witten-bell", n_classes=2)
    config_kn = SuffixConfig(smoothing_method="kneser-ney", n_classes=2)

    m_wb = SuffixSmoother(config_wb)
    m_wb.train(test_data)

    m_kn = SuffixSmoother(config_kn)
    m_kn.train(test_data)

    report = SuffixSmoother.compare([("WB", m_wb), ("KN", m_kn)], test_data)

    assert len(report) == 2
    assert report[0]["name"] in ["WB", "KN"]
    assert "accuracy" in report[0]
    assert "ece" in report[0]

def test_memory_optimization_internal():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1, 2), 0)])

    node = smoother._nodes[(1, 2)]
    # Check that _T exists and is an int
    assert hasattr(node, "_T")
    assert isinstance(node._T, int)
    assert node._T == 1

    # Check that observed_labels (from v0.2.1) does NOT exist
    assert not hasattr(node, "observed_labels")

def test_kn_memory_freed():
    config = SuffixConfig(smoothing_method="kneser-ney", n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1, 2), 0), ((3, 2), 0)])

    # Force inference to trigger finalization
    smoother.predict((1, 2))

    # Internal context sets should be None after finalization
    assert smoother._kn_label_seen_contexts is None
    for node in smoother._nodes.values():
        assert node._seen_contexts is None

    # But counts should be preserved in continuation_counts
    node_2 = smoother._nodes[(2,)]
    assert node_2.continuation_counts[0] == 1 # label 0 seen in contexts (1,) and (3,)

def test_prune():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    # (1,1) is very discriminative (KL high)
    # (2,2) is uniform (KL=0)
    smoother.train([((1, 1), 0)] * 10 + [((2, 2), 0), ((2, 2), 1)])

    initial_nodes = smoother.n_nodes
    stats = smoother.prune(min_kl=0.1)

    assert stats["nodes_removed"] > 0
    assert (2, 2) not in smoother._nodes
    assert (1, 1) in smoother._nodes # Should stay as it's discriminative

def test_conformal_margin():
    config = SuffixConfig(n_classes=2)
    smoother = SuffixSmoother(config)
    smoother.train([((1,), 0)] * 8 + [((1,), 1)] * 2)

    cal_data = [((1,), 0), ((1,), 1)]
    # This is a small sample, but should work
    smoother.calibrate(cal_data, score_type="margin")
    assert smoother._conformal_score_type == "margin"

    pset = smoother.predict_set((1,), coverage=0.5)
    assert "labels" in pset
