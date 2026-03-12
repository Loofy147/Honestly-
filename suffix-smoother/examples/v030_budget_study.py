import numpy as np
import time
import sys, os
from suffix_smoother import SuffixSmoother, SuffixConfig

def generate_complex_data(n_samples, n_classes, seed=42):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_samples):
        # Long context (len 5)
        seq = tuple(rng.integers(0, 50, size=5))
        # Label depends mostly on the last 2 elements
        label = (seq[-1] + seq[-2]) % n_classes
        data.append((seq, label))
    return data

def study_budget_pruning():
    print("--- [Budget Study 1] Accuracy vs Node Budget ---")
    n_classes = 10
    train_data = generate_complex_data(10000, n_classes)
    test_data = generate_complex_data(2000, n_classes, seed=99)

    # Train full model
    full_model = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    full_model.train(train_data)
    initial_nodes = full_model.n_nodes
    base_acc = full_model.score(test_data)

    print(f"  Full Model: {initial_nodes} nodes | Accuracy: {base_acc:.4f}")

    budgets = [initial_nodes, initial_nodes//2, initial_nodes//4, initial_nodes//10, 100, 10]
    print(f"\n  {'Budget':<10} | {'Nodes':<10} | {'Accuracy':<10} | {'Δ Acc':<10}")
    print(f"  {'-'*50}")

    for b in budgets:
        # Clone full model via merge with empty (simplified clone)
        # Actually I didn't implement clone, but merge(model, empty) works
        empty = SuffixSmoother(SuffixConfig(n_classes=n_classes))
        m = SuffixSmoother.merge(full_model, empty)

        m.prune_to_budget(max_nodes=b)
        acc = m.score(test_data)
        print(f"  {b:<10} | {m.n_nodes:<10} | {acc:<10.4f} | {acc - base_acc:>+10.4f}")

def study_incremental_calibration():
    print("\n--- [Budget Study 2] Incremental Calibration Speed ---")
    n_classes = 10
    train_data = generate_complex_data(5000, n_classes)
    cal_data = generate_complex_data(2000, n_classes, seed=88)

    smoother = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    smoother.train(train_data)

    # Batch Calibration
    t0 = time.time()
    smoother.calibrate(cal_data)
    batch_time = time.time() - t0
    print(f"  Batch Calibration (2000 samples): {batch_time*1000:.1f} ms")

    # Incremental Calibration
    smoother_inc = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    smoother_inc.train(train_data)

    t0 = time.time()
    for seq, label in cal_data:
        smoother_inc.update_calibration(seq, label)
    inc_time = time.time() - t0
    print(f"  Incremental Calibration (2000 total calls): {inc_time*1000:.1f} ms")

    # Verify results are identical
    q_batch = smoother._conformal_quantile(0.1)
    q_inc = smoother_inc._conformal_quantile(0.1)
    print(f"  Quantile Equality: {abs(q_batch - q_inc) < 1e-10} (Q={q_batch:.4f})")

if __name__ == "__main__":
    study_budget_pruning()
    study_incremental_calibration()
