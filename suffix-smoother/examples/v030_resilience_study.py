import numpy as np
import time
import sys, os
from suffix_smoother import SuffixSmoother, SuffixConfig

def generate_domain_data(n, n_classes, shift=0, seed=42):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        seq = tuple(rng.integers(0, 50, size=3))
        # Logic shifts based on the 'shift' parameter
        label = (seq[0] + seq[1] + shift) % n_classes
        data.append((seq, label))
    return data

def run_resilience_study():
    print("=== [v0.3.0] Production Resilience & Adaptation Study ===")
    n_classes = 5
    cfg = SuffixConfig(n_classes=n_classes, max_nodes=500)
    model = SuffixSmoother(cfg)

    # 1. INITIAL TRAINING (Domain A)
    print("\n[Step 1] Initial Training (Domain A)...")
    train_a = generate_domain_data(1000, n_classes, shift=0, seed=1)
    model.train(train_a)
    model.calibrate(generate_domain_data(200, n_classes, shift=0, seed=2))
    print(f"  Model ready. Nodes: {model.n_nodes} | Coverage (Stable): {model.coverage_report(train_a[:200])['actual_coverage']:.1%}")

    # 2. DRIFT DETECTION (Enter Domain B)
    print("\n[Step 2] Domain Shift Detected!")
    drift_data = generate_domain_data(200, n_classes, shift=2, seed=3)
    drift_report = model.detect_calibration_drift(drift_data)
    print(f"  Drift Analysis: {drift_report['status']}")
    print(f"  Actual Coverage dropped to: {drift_report['actual_coverage']:.1%}")

    # 3. ACTIVE ADAPTATION
    print("\n[Step 3] Online Adaptation via Active Learning...")
    t0 = time.time()
    for seq, label in drift_data:
        # Update model ONLY if it's uncertain or fails coverage
        if model.uncertainty(seq) > 1.0:
            model.train_one(seq, label)
            model.update_calibration(seq, label)

    print(f"  Adaptation complete in {(time.time()-t0)*1000:.1f}ms")

    # 4. VERIFY RECOVERY
    post_adapt_report = model.coverage_report(generate_domain_data(200, n_classes, shift=2, seed=4))
    print(f"  Post-Adaptation Coverage: {post_adapt_report['actual_coverage']:.1%}")
    print(f"  Is Reliability Restored? {post_adapt_report['is_valid']}")

    # 5. BUDGET ENFORCEMENT
    print("\n[Step 4] Maintaining Memory Budget...")
    print(f"  Nodes before pruning: {model.n_nodes}")
    model.prune_to_budget(max_nodes=300)
    print(f"  Nodes after budget pruning: {model.n_nodes}")

    final_acc = model.score(generate_domain_data(500, n_classes, shift=2, seed=5))
    print(f"  Final Adapted Accuracy: {final_acc:.1%}")

if __name__ == "__main__":
    run_resilience_study()
