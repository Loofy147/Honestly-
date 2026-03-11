import numpy as np
import time
from suffix_smoother import SuffixSmoother, SuffixConfig

def generate_synthetic_data(n_samples, n_classes, sparse=True, seed=42):
    rng = np.random.default_rng(seed)
    data = []
    patterns = {(1, 2, 3): 0, (4, 4, 4): 1, (5, 6): 2}
    for _ in range(n_samples):
        if sparse and rng.random() > 0.8:
            seq = tuple(rng.integers(0, 100, size=rng.integers(1, 6)))
            label = rng.integers(0, n_classes)
        else:
            p_keys = list(patterns.keys())
            pattern = p_keys[rng.integers(0, len(p_keys))]
            prefix = tuple(rng.integers(0, 10, size=rng.integers(0, 3)))
            seq = prefix + pattern
            label = patterns[pattern]
        data.append((seq, label))
    return data

def run_study(data_type="Sparse", label_smoothing=0.0):
    print(f"\n--- Study: {data_type} Data (LS={label_smoothing}) ---")
    n_classes = 10
    train_data = generate_synthetic_data(5000, n_classes, sparse=(data_type=="Sparse"), seed=42)
    test_data = generate_synthetic_data(1000, n_classes, sparse=(data_type=="Sparse"), seed=43)
    val_data = generate_synthetic_data(500, n_classes, sparse=(data_type=="Sparse"), seed=44)

    methods = ["jelinek-mercer", "witten-bell", "kneser-ney"]
    for method in methods:
        config = SuffixConfig(smoothing_method=method, n_classes=n_classes, label_smoothing=label_smoothing)
        smoother = SuffixSmoother(config)

        start_train = time.time()
        smoother.train(train_data)
        train_time = time.time() - start_train

        smoother.calibrate(val_data)

        correct = 0
        confs = []
        is_correct = []
        set_sizes = []
        covered = 0

        start_infer = time.time()
        for seq, label in test_data:
            pred_label, conf = smoother.predict(seq)
            correct += (pred_label == label)
            confs.append(conf)
            is_correct.append(pred_label == label)

            pset = smoother.predict_set(seq, coverage=0.9)
            set_sizes.append(len(pset["labels"]))
            covered += (label in pset["labels"])
        infer_time = (time.time() - start_infer) / len(test_data)

        ece = SuffixSmoother.expected_calibration_error(confs, is_correct)
        acc = correct / len(test_data)
        avg_set_size = np.mean(set_sizes)
        actual_coverage = covered / len(test_data)

        print(f"{method:<16} | Acc: {acc:.4f} | ECE: {ece:.4f} | SetSize: {avg_set_size:.2f} | Cov: {actual_coverage:.1%} | Inf: {infer_time*1000:.3f}ms")

if __name__ == "__main__":
    run_study("Dense", label_smoothing=0.0)
    run_study("Sparse", label_smoothing=0.0)
    run_study("Sparse", label_smoothing=0.1)
