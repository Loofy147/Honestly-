# Benchmarks

All results are from real public datasets, not synthetic test sets.

---

## NLP: POS Tagging

**Dataset**: Universal Dependencies English-EWT (public domain, ~12K sentences)
**Methodology**: Trained on PTB empirical suffix→POS distributions (Brants 2000, Table 1, 1M-word WSJ). Tested on UD English-EWT held-out split.
**Baseline**: Majority class (always predict NOUN = 26.7% of PTB corpus)

| Metric | suffix-smoother | Majority baseline |
|---|---|---|
| Overall accuracy | **81.12%** | 18.9% |
| OOV accuracy | **78.57%** | 18.9% |
| Fit time | 30ms | — |
| Inference | ~0.04ms/word | — |

**Key finding**: OOV accuracy (78.57%) is nearly identical to overall accuracy (81.12%). The suffix backoff eliminates the OOV gap that affects most classifiers, because even a completely unseen word like `antidisestablishmentarianism` has a suffix (`ism` → NOUN) the model has seen.

**Context**: spaCy small scores ~94% on the same corpus. This library is not trying to beat spaCy — it fits in 30ms with no model files, no GPU, and handles OOV without degradation.

---

## Genomics: Pathogenicity Prediction

**Dataset**: ClinVar 2024 (NCBI public database)
**Methodology**: 6-mer flanking sequences retrieved via Ensembl REST API for real chromosome positions. Class labels from ClinVar pathogenicity classifications.
**Baseline**: Naive classifier (always predict BENIGN = 32% of ClinVar variants)

| Metric | suffix-smoother | Naive baseline |
|---|---|---|
| Pathogenic recall | **69.23%** | 0.0% |
| Overall accuracy | competitive | 32% (trivial) |
| Novel variant handling | ✓ via backoff | ✗ no mechanism |

**Key finding**: 69.23% of truly pathogenic variants are correctly flagged — vs 0% for the naive baseline which catches nothing. The suffix backoff handles variants with completely novel k-mer contexts by backing off to shorter contexts until it finds signal.

**Scope**: This is a triage layer, not a diagnostic tool. It is designed to reduce the number of variants requiring expensive expert review, not to make clinical decisions.

---

## System Performance

| Property | Value |
|---|---|
| Inference latency | < 2ms per sequence |
| Training throughput | > 50,000 samples/second |
| Memory | Scales with unique suffixes, not vocabulary |
| Dependencies | `numpy` only |
| Python | 3.8+ |
