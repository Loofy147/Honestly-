# Benchmarks

The Suffix Smoother has been validated on real-world public datasets to ensure it performs in production environments, not just on synthetic test sets.

## NLP: POS Tagging
- **Dataset**: Universal Dependencies (UD) English-EWT corpus.
- **Overall Accuracy**: 81.12%
- **OOV (Out-of-Vocabulary) Accuracy**: 78.57%
- **Notes**: While trailing state-of-the-art neural models, it handles unknown words significantly better than naive baselines and runs in constant time relative to vocabulary size.

## Genomics: Pathogenicity Prediction
- **Dataset**: ClinVar Conflicting variants.
- **Methodology**: Classification based on 6-mer flanking sequences retrieved via Ensembl REST API.
- **Pathogenic Recall**: 69.23%
- **Notes**: Demonstrates biological signal extraction from raw nucleotide context using recursive backoff.

## System Performance
- **Inference Latency**: < 2ms per sequence (Python implementation).
- **Training Speed**: > 50,000 samples per second.
- **Memory Footprint**: Sparse tree scales with number of unique suffixes observed, not total vocabulary.
