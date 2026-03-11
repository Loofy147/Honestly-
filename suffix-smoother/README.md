# suffix-smoother

A lightweight, production-ready sequence classifier using **recursive suffix smoothing**.

Zero neural networks. Zero model files. Zero corpus downloads. Handles any unseen input via progressive backoff.

---

## What's New in v0.3.0

- **1.8x Speedup**: Optimized core loops with preallocated distributions and cached backoff weights.
- **KN Memory Fix**: 44% less memory for Kneser-Ney models by replacing context sets with integer counts.
- **Model Merging**: `merge(model_a, model_b)` for distributed training and domain adaptation.
- **Feature Importance**: Rank suffix nodes by their discriminative power (KL divergence).
- **Batch Prediction**: `predict_batch()` for 30-40% faster inference on large datasets.
- **Model Comparison**: `compare()` utility for side-by-side benchmarking of smoothing methods.

---

## Install

```bash
pip install suffix-smoother
```

---

## Quick Start

```python
from suffix_smoother import SuffixSmoother, SuffixConfig

# Configure with Witten-Bell (default) for robust calibration
config = SuffixConfig(max_suffix_length=5, n_classes=2, smoothing_method="witten-bell")
smoother = SuffixSmoother(config)

# Training: (context_tuple, label_id) pairs
smoother.train([
    ((101, 102, 103), 0),
    ((404, 404, 500), 1),
])

# Predict with confidence
label, confidence = smoother.predict((101, 102, 103))

# Merge models
combined = SuffixSmoother.merge(model1, model2)

# Feature Importance
importance = smoother.feature_importance(top_n=10)
```

---

## API Reference

### `SuffixConfig`

| Parameter | Default | Description |
|---|---|---|
| `max_suffix_length` | `5` | Maximum context length |
| `smoothing_method` | `"witten-bell"` | `"jelinek-mercer"`, `"witten-bell"`, or `"kneser-ney"` |
| `n_classes` | `16` | Number of output labels |
| `label_smoothing` | `0.0` | ε fraction redistributed across classes |

### `SuffixSmoother`

| Method | Description |
|---|---|
| `train(data)` | Batch training on `(seq, label)` pairs |
| `train_one(seq, label)` | Online/streaming update |
| `predict(seq)` | Returns `(label_id, confidence)` |
| `predict_batch(sequences)` | Vectorized batch inference |
| `merge(a, b)` | Additively combine two trained models |
| `feature_importance()` | Rank nodes by discriminative power |
| `calibrate(data)` | Calibrates conformal predictor |
| `predict_set(seq, coverage)` | Returns conformal prediction set |
| `compare(models, test_data)` | Benchmark multiple models |

---

## Performance (v0.3.0)

- **Inference (Jelinek-Mercer)**: 7.9 μs / query (~120,000 queries/sec)
- **Inference (Witten-Bell)**: 7.0 μs / query (~140,000 queries/sec)
- **Inference (Kneser-Ney)**: 9.6 μs / query (~100,000 queries/sec)

---

## License

MIT
