# Benchmarks & Empirical Studies

This document details the performance and calibration of Suffix Smoother v0.3.0.

---

## 1. Speed Improvements (v0.3.0 vs v0.2.1)

Optimizations in v0.3.0 including preallocated distributions, cached backoff weights, and integer-based label tracking provide significant speedups across all smoothing methods.

| Method | v0.2.1 (μs/query) | v0.3.0 (μs/query) | Speedup |
|---|---|---|---|
| Jelinek-Mercer | 14.1 | **7.9** | **1.79x** |
| Witten-Bell | 13.7 | **7.0** | **1.96x** |
| Kneser-Ney | 16.0 | **9.6** | **1.66x** |

---

## 2. Smoothing Method Comparison (POS Tagging)

**Task**: Hand-crafted POS tagging corpus based on PTB distributions.
**Metrics**:
- **Accuracy**: Top-1 prediction correctness.
- **ECE**: Expected Calibration Error (lower is better).
- **Set Size**: Average size of the prediction set at 90% coverage guarantee.

| Method | Accuracy | ECE | Set Size (90%) | Notes |
|---|---|---|---|---|
| Jelinek-Mercer | 75.0% | 0.236 | 2.50 | Baseline, overconfident |
| **Witten-Bell** | 75.0% | **0.130** | **2.17** | Best general balance |
| Kneser-Ney | 75.0% | 0.127 | 3.25 | Most robust on rare labels |

---

## 3. KN Memory Optimization

v0.3.0 reduces Kneser-Ney memory usage by **44%** by converting continuation context sets to integer counts after training and freeing the sets. Memory usage is now O(nodes × classes) rather than O(total training samples).

---

## 4. Real-World NLP Performance

**Dataset**: Universal Dependencies English-EWT
**Result**: 81.12% accuracy (78.57% on OOV words).
The recursive backoff mechanism effectively eliminates the "OOV gap" common in most sequence classifiers.
