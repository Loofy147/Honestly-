"""
v0.3.0 Grand Integration Demo: Distributed Learning & Online Adaptation
========================================================================
1. Sharded Training: Train 4 independent models on disjoint data shards.
2. Weighted Fusion: Merge shards with weights (e.g. recent data is more important).
3. Budget Optimization: Prune the fused model to fit a strict memory budget.
4. Active Learning: Process a stream of new data, updating the model online
   only when the system is uncertain (high Shannon entropy).
5. Conformal Reliability: Continuously monitor and update coverage guarantees.
"""

import numpy as np
import time
from suffix_smoother import SuffixSmoother, SuffixConfig

def generate_data(n, n_classes, seed):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        seq = tuple(rng.integers(0, 50, size=rng.integers(2, 5)))
        label = (sum(seq) % n_classes)
        data.append((seq, label))
    return data

def run_grand_demo():
    print("=== [v0.3.0] Grand Integration Demo ===")
    n_classes = 10
    cfg = SuffixConfig(n_classes=n_classes)

    # 1. Sharded Training
    print("\n[Step 1] Sharded Training (4 shards)...")
    shards = []
    for i in range(4):
        m = SuffixSmoother(cfg)
        m.train(generate_data(2000, n_classes, seed=i))
        shards.append(m)
        print(f"  Shard {i} trained. Nodes: {m.n_nodes}")

    # 2. Weighted Fusion
    print("\n[Step 2] Weighted Fusion (Merging all shards)...")
    # Merge shard 0 and 1
    m01 = SuffixSmoother.merge_weighted(shards[0], shards[1], 1.0, 1.0)
    # Merge shard 2 and 3 with higher weight for shard 3 (e.g. "newer" data)
    m23 = SuffixSmoother.merge_weighted(shards[2], shards[3], 1.0, 2.0)
    # Final merge
    fused = SuffixSmoother.merge(m01, m23)
    print(f"  Fused model nodes: {fused.n_nodes}")

    # 3. Budget Optimization
    print("\n[Step 3] Budget Optimization (Pruning to 5000 nodes)...")
    summary_before = fused.model_summary()
    fused.prune_to_budget(max_nodes=5000)
    summary_after = fused.model_summary()
    print(f"  Nodes: {summary_before['total_nodes']} -> {summary_after['total_nodes']}")
    print(f"  Mean KL Divergence: {summary_before['mean_kl']:.4f} -> {summary_after['mean_kl']:.4f}")

    # 4. Calibration
    print("\n[Step 4] Conformal Calibration...")
    cal_data = generate_data(500, n_classes, seed=99)
    fused.calibrate(cal_data)
    print(f"  Calibrated. Quantile (90%): {fused._conformal_quantile(0.1):.4f}")

    # 5. Online Active Learning Loop
    print("\n[Step 5] Online Active Learning (Streaming data)...")
    stream_data = generate_data(1000, n_classes, seed=101)
    updates = 0
    t0 = time.time()

    # Batch predict for speed, then loop for incremental logic
    batch_uncs = fused.uncertainty_batch([d[0] for d in stream_data])

    for i, (seq, true_label) in enumerate(stream_data):
        unc = batch_uncs[i]

        # If model is uncertain (entropy > threshold), ask for "human" label and update
        if unc > 1.5:
            fused.train_one(seq, true_label)
            fused.update_calibration(seq, true_label)
            updates += 1

    elapsed = time.time() - t0
    print(f"  Processed 1000 stream events in {elapsed*1000:.1f}ms")
    print(f"  Model updated online {updates} times based on uncertainty feedback.")
    print(f"  Final Node count: {fused.n_nodes}")

    # 6. Result Validation
    test_data = generate_data(1000, n_classes, seed=202)
    acc = fused.score(test_data)
    print(f"\n[Step 6] Final Accuracy: {acc:.1%}")
    print(f"  Summary: {fused.model_summary()}")

if __name__ == "__main__":
    run_grand_demo()
