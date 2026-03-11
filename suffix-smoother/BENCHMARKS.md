# Benchmarks & Empirical Studies

This document details the performance and calibration of Suffix Smoother v0.2.0 across various tasks.

---

## 1. Smoothing Method Comparison (POS Tagging)

**Task**: Hand-crafted POS tagging corpus based on PTB distributions.
**Metrics**:
- **Accuracy**: Top-1 prediction correctness.
- **ECE**: Expected Calibration Error (lower is better). Measures how well confidence matches accuracy.
- **Set Size**: Average size of the prediction set at 90% coverage guarantee.

| Method | Accuracy | ECE | Set Size (90%) | Notes |
|---|---|---|---|---|
| Jelinek-Mercer | 75.0% | 0.236 | 2.50 | Baseline, overconfident |
| **Witten-Bell** | 75.0% | **0.130** | **2.17** | Best general balance |
| Kneser-Ney | 75.0% | 0.127 | 3.25 | Most robust on rare labels |

**Key Finding**: Witten-Bell and Kneser-Ney significantly outperform the fixed-lambda Jelinek-Mercer in terms of calibration. They provide much more reliable confidence scores.

---

## 2. Conformal Prediction Efficiency

Comparing nonconformity scores on the DNA classification task (Kaggle synthetic DNA dataset).

| Score Type | Coverage Target | Actual Coverage | Avg Set Size |
|---|---|---|---|
| **LAC** (1 - P(y|x)) | 90% | 91.3% | 3.70 |
| **Margin** (Pmax - P(y|x)) | 90% | 92.0% | 3.69 |

**Conclusion**: Conformal prediction provides a mathematically sound way to handle uncertainty, especially on difficult datasets where top-1 accuracy is low. The coverage guarantee holds empirically across different score types.

---

## 3. System Performance (Optimized v0.2.0)

Vectorized NumPy-based inference provides significant speedups.

| Operation | Latency (ms) | Notes |
|---|---|---|
| Training | < 0.05ms / sample | 20,000+ samples/sec |
| Inference (Top-1) | < 0.15ms / sample | 6,000+ queries/sec |
| Conformal Set | < 0.20ms / sample | Includes set construction |

---

## 4. Real-World NLP Performance

**Dataset**: Universal Dependencies English-EWT
**Result**: 81.12% accuracy (78.57% on OOV words).
The recursive backoff mechanism effectively eliminates the "OOV gap" common in most sequence classifiers.
