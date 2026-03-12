import numpy as np
import time

from suffix_smoother import SuffixSmoother, SuffixConfig

def generate_complex_data(n_samples, n_classes, n_patterns=20, seed=42):
    rng = np.random.default_rng(seed)
    patterns = []
    for i in range(n_patterns):
        pat_len = rng.integers(2, 6)
        pat = tuple(rng.integers(0, 50, size=pat_len))
        label = i % n_classes
        patterns.append((pat, label))

    data = []
    for _ in range(n_samples):
        if rng.random() > 0.7:
            # Noise
            seq = tuple(rng.integers(0, 100, size=rng.integers(1, 6)))
            label = rng.integers(0, n_classes)
        else:
            pat, label = patterns[rng.integers(0, n_patterns)]
            prefix = tuple(rng.integers(0, 50, size=rng.integers(0, 2)))
            data.append((prefix + pat, label))
    return data

def study_batch_performance():
    print("\n--- [Study 1] Batch Prediction Throughput ---")
    n_classes = 16
    train_data = generate_complex_data(10000, n_classes)
    smoother = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    smoother.train(train_data)

    test_seqs = [d[0] for d in generate_complex_data(5000, n_classes, seed=99)]

    batch_sizes = [1, 10, 100, 1000, 5000]
    for bs in batch_sizes:
        # Warmup
        smoother.predict_batch(test_seqs[:min(bs, 100)])

        t0 = time.time()
        n_iters = max(1, 5000 // bs)
        for _ in range(n_iters):
            smoother.predict_batch(test_seqs[:bs])
        elapsed = time.time() - t0
        throughput = (bs * n_iters) / elapsed
        print(f"  Batch Size: {bs:>5} | Throughput: {throughput:>10.1f} queries/sec")

def study_pruning():
    print("\n--- [Study 2] Pruning Efficiency (Accuracy vs Memory) ---")
    n_classes = 16
    train_data = generate_complex_data(20000, n_classes)
    test_data = generate_complex_data(2000, n_classes, seed=88)

    smoother = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    smoother.train(train_data)
    initial_nodes = smoother.n_nodes

    test_seqs = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]

    # Baseline Accuracy
    preds = smoother.predict_batch(test_seqs)
    base_acc = sum(p[0] == l for p, l in zip(preds, test_labels)) / len(test_labels)

    thresholds = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    print(f"  {'Thresh':<8} | {'Nodes':<10} | {'% Rem':<8} | {'Accuracy':<10} | {'Δ Acc':<8}")
    print(f"  {'-'*55}")

    for th in thresholds:
        # Reload/Retrain model to avoid cumulative pruning for study
        s = SuffixSmoother(SuffixConfig(n_classes=n_classes))
        s.train(train_data)

        pruned = s.prune(min_kl=th)
        preds = s.predict_batch(test_seqs)
        acc = sum(p[0] == l for p, l in zip(preds, test_labels)) / len(test_labels)

        rem_pct = 100 * pruned['nodes_removed'] / initial_nodes
        print(f"  {th:<8.2f} | {s.n_nodes:<10} | {rem_pct:>7.1f}% | {acc:>10.4f} | {acc-base_acc:>+8.4f}")

def study_conformal_types():
    print("\n--- [Study 3] Conformal Prediction Efficiency ---")
    n_classes = 16
    train_data = generate_complex_data(10000, n_classes)
    val_data = generate_complex_data(2000, n_classes, seed=77)
    test_data = generate_complex_data(2000, n_classes, seed=66)

    smoother = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    smoother.train(train_data)

    test_seqs = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]

    for score_type in ["lac", "margin"]:
        smoother.calibrate(val_data, score_type=score_type)

        set_sizes = []
        covered = 0
        for seq, label in test_data:
            pset = smoother.predict_set(seq, coverage=0.9)
            set_sizes.append(len(pset['labels']))
            covered += (label in pset['labels'])

        avg_size = np.mean(set_sizes)
        actual_cov = covered / len(test_data)
        print(f"  Score Type: {score_type:<8} | Avg Set Size: {avg_size:.2f} | Actual Coverage: {actual_cov:.1%}")

def study_merging():
    print("\n--- [Study 4] Knowledge Merging (Distributed Learning) ---")
    n_classes = 5
    # Two disjoint sets of patterns
    data_a = generate_complex_data(5000, n_classes, n_patterns=10, seed=1)
    data_b = generate_complex_data(5000, n_classes, n_patterns=10, seed=2)
    test_data = generate_complex_data(2000, n_classes, n_patterns=20, seed=3) # Contains patterns from both

    test_seqs = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]

    m_a = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    m_a.train(data_a)

    m_b = SuffixSmoother(SuffixConfig(n_classes=n_classes))
    m_b.train(data_b)

    merged = SuffixSmoother.merge(m_a, m_b)

    def get_acc(model):
        preds = model.predict_batch(test_seqs)
        return sum(p[0] == l for p, l in zip(preds, test_labels)) / len(test_labels)

    print(f"  Model A Accuracy: {get_acc(m_a):.4f}")
    print(f"  Model B Accuracy: {get_acc(m_b):.4f}")
    print(f"  Merged Accuracy:  {get_acc(merged):.4f} (expected boost)")

if __name__ == "__main__":
    study_batch_performance()
    study_pruning()
    study_conformal_types()
    study_merging()
