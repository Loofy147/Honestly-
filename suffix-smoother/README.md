# suffix-smoother

A high-performance, production-ready sequence classifier using **recursive suffix smoothing**.

Zero neural networks. Zero GPUs. Zero model files. Handles OOV via progressive backoff.

---

## What's New in v0.4.0
- **Industrial Pruning**: `prune(min_count=k)` for massive RAM savings (up to 93% reduction).
- **Auto-Calibration**: `fit_temperature()` minimizes ECE/NLL for reliable confidence scores.
- **Diagnostics**: `calibration_curve()` for binned reliability analysis.
- **Portability**: `to_json()` and `from_json()` for cross-platform model persistence.

---

## Core Philosophy: Efficient AI
Suffix Smoother is designed for industrial environments where memory, latency, and explainability matter more than parameter count. By utilizing **recursive suffix backoff** instead of deep learning, it achieves competitive accuracy on sequence tasks with **linear training throughput** and **sub-microsecond inference** on standard CPUs.

---

## Documentation & Benchmarks

- **[Technical Documentation](DOCUMENTATION.md)**: Deep dive into pruning, calibration, and architecture.
- **[Benchmark Reports](BENCHMARKS.md)**: Performance results on 1M+ samples and real-world sentiment data.

---

## Install

```bash
pip install suffix-smoother
```

---

## Quick Start (v0.4.0)

```python
from suffix_smoother import SuffixSmoother, SuffixConfig

# 1. Train at Industrial Scale
model = SuffixSmoother(SuffixConfig(n_classes=10))
model.train(data) # list of (seq_tuple, label_int)

# 2. Optimize Reliability (v0.4.0)
model.fit_temperature(val_data)
curve = SuffixSmoother.calibration_curve(test_probs, test_labels)

# 3. Compress for Production (v0.4.0)
# Removes rare nodes (90% reduction on 1M+ samples)
model.prune(min_count=10)

# 4. Save Portably (v0.4.0)
model.to_json("production_model.json")
```

---

## Performance (Industrial Benchmark)

| Dataset | Scale | Throughput | RAM (Pruned) |
|---|---|---|---|
| Synthetic | 1.5M samples | 100k/sec | -89% nodes |
| Sentiment140 | 1.6M tweets | 85k/sec | 4MB (JSON) |

---

## License

MIT
