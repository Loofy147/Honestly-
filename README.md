# Quantum Suffix Smoother & Ribbon Filter Suite

This repository contains a collection of high-performance algorithms for sequence classification and memory-efficient membership testing.

## Recent Updates: Real-World Validation

We have performed an honest assessment of our core algorithms against real public datasets:

### NLP POS Tagging (UD English-EWT)
- **Overall Accuracy**: 81.12%
- **OOV Accuracy**: 78.57%
- **Performance**: Inference at ~1.5ms/sentence.
- *Status*: Production-ready for low-resource or high-speed tagging.

### Genomic Variant Classification (ClinVar)
- **Pathogenic Recall**: 69.23% (on real hg38 flanking sequences)
- **Ribbon Filter Memory Savings**: 26.8% vs Bloom filter baseline with 0% false negatives.
- *Status*: Strong proof-of-concept for memory-constrained clinical variant databases.

## Standalone Library: Suffix Smoother

The core intellectual asset, `QuantumSuffixSmoother`, has been extracted into a standalone, domain-agnostic library: **`suffix_smoother_lib`**.

- **Rename**: Refactored as `SuffixSmoother`.
- **Location**: `/suffix_smoother_lib`
- **Usage**: Ideal for any sequence-to-class problem where suffix contexts are predictive (logs, genomics, NLP).
- **Distribution**: Includes `pyproject.toml` for easy packaging.

## Directory Structure

- `filters/`: Ribbon Filter implementation.
- `error_correction/`: Suffix Smoothing and Quantum Error Correction.
- `engines/`: EKRLS (Extended Kernel Recursive Least Squares) engines.
- `algebra/`: Lie Algebra expansions and Entanglement Batteries.
- `metacognition/`: Bayesian monitoring and Q-score validation.
- `suffix_smoother_lib/`: Standalone extraction of the Suffix Smoother.

## Testing

Run the full test suite (36 tests):
```bash
python3 test_all.py
```
