import numpy as np
import time
from multiprocessing import Pool
from suffix_smoother import SuffixSmoother, SuffixConfig

def train_shard(args):
    data, cfg = args
    model = SuffixSmoother(cfg)
    model.train(data)
    return model

def generate_shard_data(n, n_classes, seed):
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        seq = tuple(rng.integers(0, 50, size=3))
        label = (seq[0] + seq[1]) % n_classes
        data.append((seq, label))
    return data

def run_distributed_study():
    print("--- [v0.3.0] Distributed Parallel Training Study ---")
    n_shards = 4
    samples_per_shard = 5000
    n_classes = 10
    cfg = SuffixConfig(n_classes=n_classes)

    # 1. Sequential Training (Baseline)
    print(f"Sequential Training ({n_shards * samples_per_shard} samples)...")
    t0 = time.time()
    seq_model = SuffixSmoother(cfg)
    for i in range(n_shards):
        seq_model.train(generate_shard_data(samples_per_shard, n_classes, i))
    seq_time = time.time() - t0
    print(f"  Completed in {seq_time:.3f}s. Nodes: {seq_model.n_nodes}")

    # 2. Parallel Training + v0.3.0 Merging
    print(f"Parallel Training on {n_shards} shards + merge_all()...")
    t0 = time.time()
    shard_args = [(generate_shard_data(samples_per_shard, n_classes, i), cfg) for i in range(n_shards)]

    with Pool(processes=n_shards) as pool:
        models = pool.map(train_shard, shard_args)

    merged_model = SuffixSmoother.merge_all(models)
    par_time = time.time() - t0
    print(f"  Completed in {par_time:.3f}s. Nodes: {merged_model.n_nodes}")

    # 3. Validation
    test_data = generate_shard_data(2000, n_classes, 99)
    acc_seq = seq_model.score(test_data)
    acc_merged = merged_model.score(test_data)

    print(f"\nResults:")
    print(f"  Sequential Accuracy: {acc_seq:.1%}")
    print(f"  Merged Accuracy:     {acc_merged:.1%}")
    print(f"  Accuracy Loss:       {acc_seq - acc_merged:.4f}")

    if abs(acc_seq - acc_merged) < 0.01:
        print("\n  ✓ Distributed training verified: Merged model matches sequential performance.")
    else:
        print("\n  ⚠ Significant difference in accuracy. Investigation needed.")

if __name__ == "__main__":
    run_distributed_study()
