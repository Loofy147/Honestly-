"""
Quantum-Classical Isomorphism Bridge
=====================================
الجسر الرياضي بين الكم والعالم الكلاسيكي

KEY INSIGHT: The quantum simulation math IS domain-agnostic.
We built 5 mathematical engines. Each solves a STRUCTURAL problem
that appears identically across unrelated fields.

ISOMORPHISM TABLE:
══════════════════════════════════════════════════════════════════
 Quantum Concept          →  Classical Isomorph
══════════════════════════════════════════════════════════════════
 Quantum state Φ_n        →  Any hidden system state (price, health, climate)
 Entanglement ρ⊗σ         →  Correlated variables (assets, genes, neurons)
 RKHS kernel mapping      →  Non-linear feature space (any domain)
 EKRLS tracking           →  Real-time non-linear state estimation
 Entanglement battery      →  Resource reservoir (energy, capital, attention)
 Lie algebra expansion     →  Symmetry-constrained optimization
 Suffix smoothing P(t|w)  →  Sequence prediction under sparsity
 Viterbi decoding         →  Optimal path in any hidden Markov model
 Ribbon filter            →  Space-efficient membership in any large set
 Q-score Bayesian         →  Universal model quality metric
══════════════════════════════════════════════════════════════════

DOMAINS IMPLEMENTED:
  1. Finance         → EKRLS for volatility + Viterbi for regime detection
  2. Genomics        → Ribbon filter for SNP lookup + Suffix for mutation prediction
  3. Climate         → EKRLS for anomaly detection + Lie for conservation laws
  4. Drug Discovery  → Q-score for molecular model validation
  5. NLP/Cognition   → Suffix smoothing for language + Viterbi for POS tagging
"""

DOMAIN_REGISTRY = {
    "finance": {
        "engines": ["EKRLS", "Viterbi", "Ribbon", "Q-Score"],
        "quantum_analog": "price = state vector Φ; volatility = coherence; regime = QEC code",
        "value_proposition": "Non-linear vol estimation + market regime detection",
        "key_outputs": ["volatility_forecast", "regime_label", "signal_confidence"],
    },
    "genomics": {
        "engines": ["Ribbon", "Suffix", "Q-Score"],
        "quantum_analog": "SNP variant = entanglement pair; mutation pattern = error syndrome",
        "value_proposition": "27% memory reduction for genome-scale lookups; mutation prediction",
        "key_outputs": ["snp_present", "mutation_probability", "qec_correction"],
    },
    "climate": {
        "engines": ["EKRLS", "Lie", "Q-Score"],
        "quantum_analog": "climate variable = state Φ; conservation = energy balance",
        "value_proposition": "Anomaly detection with conservation law enforcement",
        "key_outputs": ["anomaly_score", "trend_forecast", "conservation_residual"],
    },
    "drug_discovery": {
        "engines": ["Suffix", "Q-Score", "Ribbon"],
        "quantum_analog": "molecular fingerprint = quantum state; drug-target = entanglement pair",
        "value_proposition": "Bayesian validation of molecular docking models",
        "key_outputs": ["binding_probability", "model_q_score", "candidate_ranking"],
    },
    "nlp": {
        "engines": ["Suffix", "Viterbi", "Q-Score"],
        "quantum_analog": "word sequence = quantum suffix tree; tag = QEC code",
        "value_proposition": "Universal POS tagging with recursive smoothing",
        "key_outputs": ["pos_tags", "confidence", "oov_handling"],
    },
}


def get_isomorphism(domain: str) -> dict:
    """Return the structural isomorphism for a domain."""
    return DOMAIN_REGISTRY.get(domain, {})


def list_domains() -> list[str]:
    return list(DOMAIN_REGISTRY.keys())
