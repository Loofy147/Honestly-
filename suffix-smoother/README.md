# suffix-smoother

A lightweight, production-ready sequence classifier using **recursive suffix smoothing**.

Zero neural networks. Zero model files. Zero corpus downloads. Handles any unseen input via progressive backoff.

---

## What's New in v0.2.0

- **3 Research-Backed Smoothing Methods**: Jelinek-Mercer, Witten-Bell, and Kneser-Ney.
- **Conformal Prediction**: `calibrate()` + `predict_set()` providing mathematical coverage guarantees.
- **Streaming Training**: `train_one()` for real-time adaptation without full retraining.
- **Optimized Core**: Vectorized NumPy-based inference (6,000+ queries/sec).
- **Improved Calibration**: Low ECE (Expected Calibration Error) on sparse datasets.

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

# Statistical Coverage Guarantee via Conformal Prediction
smoother.calibrate(validation_data) # list of (seq, true_label)
result = smoother.predict_set((101, 102), coverage=0.90)
print(result['labels']) # Minimal set guaranteed to contain true label >= 90% of time
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
| `predict_set(seq, coverage)` | Returns conformal prediction set |
| `calibrate(data)` | Calibrates conformal predictor |
| `uncertainty(seq)` | Shannon entropy in bits |

---

## Performance

- **Inference**: < 0.15ms per sequence
- **Training**: > 20,000 samples/second
- **Dependencies**: `numpy` only

---

## License

MIT
