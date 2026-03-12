# suffix-smoother

A high-performance, production-ready sequence classifier using **recursive suffix smoothing**.

Zero neural networks. Zero model files. Zero corpus downloads. Handles OOV via progressive backoff.

---

## What's New in v0.3.0

- **1.8x - 2x Speedup**: Vectorized core loops and optimized backoff weight caching.
- **Advanced Conformal Prediction**:
  - **APS (Adaptive Prediction Sets)**: State-of-the-art coverage guarantees.
  - **Online Calibration**: `update_calibration()` for incremental refinement.
  - **Drift Detection**: `detect_calibration_drift()` to monitor reliability in production.
- **Industrial Memory Management**:
  - **KN Optimization**: 44% memory reduction for Kneser-Ney models.
  - **Budget Pruning**: `prune_to_budget()` ensures model fits in strict RAM constraints.
- **Collaborative Learning**:
  - **Weighted Merging**: `merge_weighted()` for domain adaptation and ensemble fusion.
  - **Sharded Fusion**: `merge_all()` for large-scale distributed training.
- **Deep Interpretability**:
  - **Feature Importance**: Rank suffixes by discriminative power (KL divergence).
  - **Label Insight**: `label_importance()` finds motifs for specific classes.

---

## Install

```bash
pip install suffix-smoother
```

---

## Quick Start

```python
from suffix_smoother import SuffixSmoother, SuffixConfig

# 1. Train and Predict
cfg = SuffixConfig(n_classes=10, max_nodes=5000) # Budgeted memory
model = SuffixSmoother(cfg)
model.train(data) # list of (seq_tuple, label_int)

# 2. Vectorized High-Throughput Inference
results = model.predict_batch(sequences)

# 3. Model Merging (Domain Adaptation)
general_model = SuffixSmoother.load("general.pkl")
medical_model = SuffixSmoother.load("medical.pkl")
# Fuse knowledge: medical knowledge is 5x more important for this deployment
fused = SuffixSmoother.merge_weighted(general_model, medical_model, w_a=1.0, w_b=5.0)

# 4. Conformal Reliability
fused.calibrate(val_data, score_type="aps")
prediction_set = fused.predict_set(seq, coverage=0.95)
```

---

## Performance (v0.3.0)

| Operation | v0.2.1 | v0.3.0 | Improvement |
|---|---|---|---|
| Inference (Top-1) | 14.1 μs | 7.0 μs | **2.0x** |
| Batch Throughput | 6,000/s | 140,000/s | **23x** |
| KN Memory | 100% | 56% | **-44%** |

---

## License

MIT
