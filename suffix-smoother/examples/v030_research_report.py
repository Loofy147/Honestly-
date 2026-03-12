import numpy as np
import time
import sys, os
from suffix_smoother import SuffixSmoother, SuffixConfig

def generate_noise_data(n_samples, n_classes, seed=42):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_samples):
        seq = tuple(rng.integers(0, 100, size=rng.integers(1, 4)))
        label = rng.integers(0, n_classes)
        data.append((seq, label))
    return data

def research_merging_fidelity():
    print("\n--- [Research 1] Model Merging Fidelity (Deep Copy Verification) ---")
    config = SuffixConfig(n_classes=2)
    m1 = SuffixSmoother(config); m1.train([((1,), 0)])
    m2 = SuffixSmoother(config); m2.train([((2,), 1)])

    merged = SuffixSmoother.merge(m1, m2)
    m1._nodes[(1,)].counts[0] = 999.0
    merged_val = merged.predict_distribution((1,))[0]
    if merged_val < 0.9:
        print("  ✓ Deep copy verified: Original model modification did not affect merged model.")
    else:
        print("  ❌ Merge failed deep copy check.")

def research_weighted_merging():
    print("\n--- [Research 2] Weighted Merging (Domain Adaptation) ---")
    config = SuffixConfig(n_classes=2)
    general = SuffixSmoother(config); general.train([((1,), 0)] * 10)
    domain = SuffixSmoother(config); domain.train([((1,), 1)] * 10)
    merged = SuffixSmoother.merge_weighted(general, domain, w_a=1.0, w_b=2.0)
    prob_1 = merged.predict_distribution((1,))[1]
    print(f"  Domain Weight: 2.0 | Prob(Label 1): {prob_1:.4f} (expected ~0.66)")
    if prob_1 > 0.6:
        print("  ✓ Weighted merge verified.")
    else:
        print("  ❌ Weighted merge failed.")

def research_pruning_tradeoffs():
    print("\n--- [Research 3] Pruning Tradeoffs (Accuracy vs Frequency vs Info) ---")
    n_classes = 10
    train_data = []
    patterns = { (1,1): 0, (2,2): 1 }
    for _ in range(500):
        for p, l in patterns.items(): train_data.append((p, l))
    rng = np.random.default_rng(42)
    for i in range(1000):
        train_data.append(((i+10,), rng.integers(0, n_classes)))

    smoother = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    smoother.train(train_data)
    initial_nodes = smoother.n_nodes
    test_data = [((1,1), 0), ((2,2), 1)]
    base_acc = sum(smoother.predict(s)[0] == l for s, l in test_data) / 2
    stats = smoother.prune(min_samples=10)
    pruned_acc = sum(smoother.predict(s)[0] == l for s, l in test_data) / 2
    print(f"  Initial Nodes: {initial_nodes}")
    print(f"  Nodes Removed (min_samples=10): {stats['nodes_removed']}")
    print(f"  Pattern Accuracy maintained: {pruned_acc == base_acc} ({pruned_acc:.1%})")

def research_batch_latency():
    print("\n--- [Research 4] Latency Scaling (Vectorized vs Loop) ---")
    n_classes = 32
    smoother = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    train_data = generate_noise_data(5000, n_classes)
    smoother.train(train_data)
    test_seqs = [d[0] for d in generate_noise_data(1000, n_classes, seed=99)]
    t0 = time.time()
    for s in test_seqs: smoother.predict(s)
    loop_time = time.time() - t0
    t0 = time.time()
    smoother.predict_batch(test_seqs)
    vec_time = time.time() - t0
    print(f"  Loop Inference: {loop_time*1000:.1f} ms")
    print(f"  Vectorized Batch: {vec_time*1000:.1f} ms")
    print(f"  Speedup: {loop_time/vec_time:.1f}x")

def research_model_summary():
    print("\n--- [Research 5] Model Structure Insight ---")
    config = SuffixConfig(n_classes=5)
    smoother = SuffixSmoother(config)
    smoother.train([((1,2), 0), ((3,4), 1), ((5,6), 2)])
    summary = smoother.model_summary()
    print(f"  Smoothing: {summary['smoothing']}")
    print(f"  Total Nodes: {summary['total_nodes']}")
    print(f"  Mean KL: {summary.get('mean_kl', 'N/A')}")

if __name__ == "__main__":
    research_merging_fidelity()
    research_weighted_merging()
    research_pruning_tradeoffs()
    research_batch_latency()
    research_model_summary()
