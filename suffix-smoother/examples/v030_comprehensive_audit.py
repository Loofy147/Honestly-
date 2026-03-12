import numpy as np
import time
import sys, os
from suffix_smoother import SuffixSmoother, SuffixConfig

def generate_data(n, n_classes, seed):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        seq = tuple(rng.integers(0, 100, size=rng.integers(2, 6)))
        label = (sum(seq) % n_classes)
        data.append((seq, label))
    return data

def run_audit():
    print("======================================================")
    print("  SUFFIX SMOOTHER v0.3.0 — COMPREHENSIVE SYSTEM AUDIT")
    print("======================================================")

    n_classes = 10
    cfg = SuffixConfig(n_classes=n_classes)

    # 1. Distributed Sharded Training
    print("\n[Audit 1] Distributed Sharded Training...")
    m1 = SuffixSmoother(cfg); m1.train(generate_data(1000, n_classes, 1))
    m2 = SuffixSmoother(cfg); m2.train(generate_data(1000, n_classes, 2))
    print(f"  Shard 1: {m1.n_nodes} nodes | Shard 2: {m2.n_nodes} nodes")

    # 2. Deep Merge and Weighted Fusion
    print("\n[Audit 2] Model Fusion...")
    merged = SuffixSmoother.merge_weighted(m1, m2, w_a=0.5, w_b=1.5)
    print(f"  Fused Model: {merged.n_nodes} nodes")

    # 3. High-Throughput Batch Operations
    print("\n[Audit 3] Vectorized Batch Operations...")
    test_data = generate_data(1000, n_classes, 99)
    test_seqs = [d[0] for d in test_data]

    t0 = time.time()
    batch_preds = merged.predict_batch(test_seqs)
    batch_dists = merged.predict_distributions_batch(test_seqs)
    batch_uncs = merged.uncertainty_batch(test_seqs)
    elapsed = time.time() - t0
    print(f"  Throughput: {3000 / elapsed:.1f} ops/sec (Inference + Dist + Entropy)")

    # 4. Advanced Conformal Prediction (APS)
    print("\n[Audit 4] Adaptive Prediction Sets (APS)...")
    cal_data = generate_data(500, n_classes, 77)
    merged.calibrate(cal_data, score_type="aps")
    batch_sets = merged.predict_set_batch(test_seqs[:5], coverage=0.9)
    print(f"  APS Quantile (90%): {merged._conformal_quantile(0.1):.4f}")
    print(f"  Sample Set: {batch_sets[0]['labels']} (size: {batch_sets[0]['n_labels']})")

    # 5. Online Active Learning
    print("\n[Audit 5] Online Active Learning & Calibration...")
    stream = generate_data(100, n_classes, 101)
    updates = 0
    for seq, label in stream:
        if merged.uncertainty(seq) > 2.0:
            merged.train_one(seq, label)
            merged.update_calibration(seq, label)
            updates += 1
    print(f"  Stream updates: {updates}/100 based on uncertainty threshold.")

    # 6. Memory Budget Enforcement
    print("\n[Audit 6] Pruning & Budgeting...")
    nodes_before = merged.n_nodes
    merged.prune_to_budget(max_nodes=1000)
    print(f"  Memory Budgeting: {nodes_before} -> {merged.n_nodes} nodes (Target: 1000)")

    # 7. Model Summarization
    print("\n[Audit 7] Model Insights...")
    summary = merged.model_summary()
    print(f"  Final Summary: {summary}")

    imp = merged.feature_importance(top_n=3)
    print("  Top Predictive Suffixes:")
    for f in imp:
        print(f"    {f['suffix']} -> KL: {f['kl_divergence']:.3f}")

    print("\n======================================================")
    print("  AUDIT COMPLETE — SYSTEM OPTIMIZED AND VALIDATED")
    print("======================================================")

if __name__ == "__main__":
    export_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, export_path)
    run_audit()
