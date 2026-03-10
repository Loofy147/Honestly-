# Suffix Smoother Library

A lightweight, high-performance sequence classifier using recursive suffix smoothing. This library is domain-agnostic and excels at problems where sequence suffixes are strong predictors of class labels.

## Features

- **Robust OOV Handling**: Gracefully falls back to shorter suffixes (n-grams) when an exact match isn't found.
- **Blazing Fast**: Constant time lookup O(k) where k is max suffix length.
- **Zero-Dependency Core**: Only requires \`numpy\`.
- **Interpretable**: Provides confidence and entropy-based uncertainty metrics.

## Installation

\`\`\`bash
pip install .
\`\`\`

## Quick Start

\`\`\`python
from suffix_smoother import SuffixSmoother, SuffixConfig

# Configure: max suffix 5, 2 classes (Normal, Anomaly)
config = SuffixConfig(max_suffix_length=5, n_classes=2)
smoother = SuffixSmoother(config)

# Training data: (context_tuple, label_id)
training = [
    ((101, 102, 103), 0),
    ((404, 404, 500), 1),
]
smoother.train(training)

# Predict
label, confidence = smoother.predict((101, 102, 103))
print(f"Label: {label}, Confidence: {confidence:.2f}")
\`\`\`

## Real-World Performance

See [BENCHMARKS.md](BENCHMARKS.md) for details on NLP (81% accuracy) and Genomics (69% recall) validation.
