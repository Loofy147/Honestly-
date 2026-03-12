import pytest
import numpy as np
import sys, os
from suffix_smoother import SuffixSmoother, SuffixConfig

def test_memory_scaling_adversarial():
    # Vocabulary size 1000, context length 5
    # Train on 10,000 random samples
    # Ensure memory (nodes) stays bounded
    config = SuffixConfig(n_classes=10, max_suffix_length=3)
    smoother = SuffixSmoother(config)

    rng = np.random.default_rng(42)
    for _ in range(100):
        data = []
        for _ in range(100):
            seq = tuple(rng.integers(0, 1000, size=5))
            label = rng.integers(0, 10)
            data.append((seq, label))
        smoother.train(data)

    # Each sample of length 5 adds at most 3 nodes.
    # 10,000 samples -> max 30,000 nodes.
    print(f"Nodes after 10k samples: {smoother.n_nodes}")
    assert smoother.n_nodes <= 30001

    # Verify no context sets are leaking in KN
    kn_config = SuffixConfig(smoothing_method="kneser-ney", n_classes=10)
    kn_smoother = SuffixSmoother(kn_config)
    kn_smoother.train(data)
    kn_smoother.predict((1,2,3)) # trigger finalization

    assert kn_smoother._kn_label_seen_contexts is None
    for node in kn_smoother._nodes.values():
        assert not hasattr(node, "_seen_contexts") or node._seen_contexts is None

if __name__ == "__main__":
    test_memory_scaling_adversarial()
