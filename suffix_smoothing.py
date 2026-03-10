"""
Suffix Smoothing — تصحيح الأخطاء الكمومية
Quantum Suffix Trees for QEC code lookup.

P(t|w_n) = λ · P_ML(t|w_n) + (1-λ) · P(t|w_{n-1})

Recursive abstraction over quantum state "suffixes" reduces
uncertainty in error correction protocol lookup.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class SuffixConfig:
    """Configuration for quantum suffix smoothing."""
    max_suffix_length: int = 8    # Maximum suffix depth
    smoothing_lambda: float = 0.7  # λ weight for MLE vs backoff
    min_count: float = 1.0         # Minimum count threshold
    n_qec_codes: int = 16          # Number of QEC code classes


class QuantumSuffixNode:
    """
    Node in the Quantum Suffix Tree.
    Stores probability distribution over QEC codes at this suffix level.
    """

    def __init__(self, depth: int = 0):
        self.depth = depth
        self.counts: defaultdict[int, float] = defaultdict(float)
        self.children: dict[str, 'QuantumSuffixNode'] = {}
        self.total: float = 0.0

    def observe(self, code: int, weight: float = 1.0):
        """Record observation of QEC code at this node."""
        self.counts[code] += weight
        self.total += weight

    def mle_probability(self, code: int) -> float:
        """Maximum Likelihood Estimate P_ML(code | suffix)."""
        if self.total < 1e-12:
            return 0.0
        return self.counts[code] / self.total

    def uniform_probability(self, n_codes: int) -> float:
        """Uniform (uninformative) prior."""
        return 1.0 / n_codes


class QuantumSuffixSmoother:
    """
    Quantum Suffix Smoothing for QEC code probability estimation.

    Recursive formula:
        P(t | suffix_i) = λᵢ · P_ML(t | suffix_i) + (1-λᵢ) · P(t | suffix_{i-1})

    Base case (i=0): P(t | ∅) = 1/|T|  [uniform prior]

    Supports unknown quantum "words" (unseen error patterns) via
    progressive suffix abstraction — backing off to shorter contexts
    until a reliable estimate is found.
    """

    def __init__(self, config: Optional[SuffixConfig] = None):
        self.cfg = config or SuffixConfig()
        self.root = QuantumSuffixNode(depth=0)
        self.nodes: dict[tuple, QuantumSuffixNode] = {}
        self.n_codes = self.cfg.n_qec_codes
        self.training_samples: int = 0

        # Adaptive smoothing weights per suffix level
        self.lambdas: list[float] = [
            self.cfg.smoothing_lambda ** (i + 1)
            for i in range(self.cfg.max_suffix_length)
        ]

    def _get_or_create_node(self, suffix: tuple) -> QuantumSuffixNode:
        """Get or create a suffix tree node."""
        if suffix not in self.nodes:
            self.nodes[suffix] = QuantumSuffixNode(depth=len(suffix))
        return self.nodes[suffix]

    def train(self, sequences: list[tuple[tuple, int]]) -> dict:
        """
        Train on (quantum_state_sequence, qec_code) pairs.
        quantum_state_sequence: tuple of discretized state labels
        qec_code: integer class label for the correction applied
        """
        for state_seq, code in sequences:
            n = len(state_seq)
            # Update all suffix nodes
            for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
                suffix = state_seq[max(0, n - length):]
                node = self._get_or_create_node(suffix)
                node.observe(code)

            # Update root (empty suffix)
            self.root.observe(code)
            self.training_samples += 1

        return {
            "samples_trained": len(sequences),
            "total_nodes": len(self.nodes),
            "total_training_samples": self.training_samples,
        }

    def predict_probability(self, state_seq: tuple, code: int) -> float:
        """
        Compute P(code | state_seq) using recursive suffix smoothing.

        P(t | w_n) = λ · P_ML(t | w_n) + (1-λ) · P(t | w_{n-1})
        Base:        P(t | ∅)  = 1/|T|

        Returns probability in [0, 1].
        """
        n = len(state_seq)

        # Start from base case: uniform prior
        p_current = self.root.uniform_probability(self.n_codes)

        # Recursive smoothing from shortest suffix to longest
        for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
            suffix = state_seq[max(0, n - length):]
            node = self._get_or_create_node(suffix)

            lam = self.lambdas[length - 1]

            if node.total >= self.cfg.min_count:
                p_mle = node.mle_probability(code)
                # Smooth: blend MLE with lower-order estimate
                p_current = lam * p_mle + (1 - lam) * p_current
            # else: skip this level — not enough data, keep p_current

        return float(p_current)

    def predict_distribution(self, state_seq: tuple) -> dict[int, float]:
        """
        Return full probability distribution over all QEC codes.
        Returns dict {code: probability} normalized to sum=1.
        """
        probs = {
            code: self.predict_probability(state_seq, code)
            for code in range(self.n_codes)
        }
        total = sum(probs.values())
        if total > 1e-12:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    def best_correction(self, state_seq: tuple) -> tuple[int, float]:
        """
        Return the most probable QEC code and its probability.
        Returns: (best_code, confidence)
        """
        dist = self.predict_distribution(state_seq)
        best_code = max(dist, key=dist.get)
        confidence = dist[best_code]
        return best_code, confidence

    def uncertainty(self, state_seq: tuple) -> float:
        """
        Measure uncertainty as entropy of distribution over QEC codes.
        H = -Σ P(t) log₂ P(t)   [bits]
        Max entropy = log₂(n_codes) bits.
        """
        dist = self.predict_distribution(state_seq)
        probs = np.array(list(dist.values()))
        probs = probs[probs > 1e-12]
        return float(-np.sum(probs * np.log2(probs)))

    def max_uncertainty(self) -> float:
        """Maximum possible uncertainty (uniform distribution)."""
        return float(np.log2(self.n_codes))


class QuantumErrorCorrector:
    """
    Full QEC system using suffix smoothing for code selection.

    Workflow:
    1. Observe quantum error syndrome (state sequence)
    2. Look up best QEC code via suffix smoother
    3. Apply correction and update training data
    4. Feedback loop: successful corrections reinforce the suffix model
    """

    def __init__(self, config: Optional[SuffixConfig] = None):
        self.cfg = config or SuffixConfig()
        self.smoother = QuantumSuffixSmoother(config)
        self.corrections_applied: int = 0
        self.successful_corrections: int = 0
        self.uncertainty_history: list[float] = []

    def _discretize_state(self, phi: np.ndarray, n_bins: int = 8) -> tuple:
        """Convert continuous quantum state to discrete symbol sequence."""
        probs = np.abs(phi) ** 2
        probs = probs / (probs.sum() + 1e-12)
        bins = np.linspace(0, 1, n_bins + 1)
        symbols = tuple(int(np.digitize(p, bins) - 1) for p in probs)
        return symbols

    def initialize(self, n_training: int = 1000, seed: int = 42) -> dict:
        """
        Initialize with synthetic QEC training data.
        Simulates common quantum error patterns and their corrections.
        """
        rng = np.random.default_rng(seed)
        dim = 4
        sequences = []

        # Generate synthetic (error_syndrome → correction) pairs
        for _ in range(n_training):
            # Random quantum state with various error patterns
            phi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            phi /= np.linalg.norm(phi) + 1e-12

            state_seq = self._discretize_state(phi.real)

            # QEC code depends on dominant error type
            dominant_error = int(np.argmax(np.abs(phi.real)))
            error_magnitude = float(np.max(np.abs(phi.imag)))

            # Map to one of 16 stabilizer codes
            code = (dominant_error * 4 + int(error_magnitude * 4)) % self.cfg.n_qec_codes

            sequences.append((state_seq, code))

        result = self.smoother.train(sequences)
        return result

    def correct(self, phi: np.ndarray) -> dict:
        """
        Find and apply best QEC correction for quantum state phi.
        Returns correction report.
        """
        state_seq = self._discretize_state(phi.real)
        code, confidence = self.smoother.best_correction(state_seq)
        uncertainty = self.smoother.uncertainty(state_seq)

        self.uncertainty_history.append(uncertainty)
        self.corrections_applied += 1

        # Apply correction (flip phase based on code parity)
        corrected_phi = phi.copy()
        if code % 2 == 1:
            corrected_phi = corrected_phi * np.exp(1j * np.pi * code / self.cfg.n_qec_codes)

        # Assess correction quality
        norm_before = float(np.linalg.norm(phi))
        norm_after = float(np.linalg.norm(corrected_phi))
        quality = 1.0 - abs(norm_before - norm_after) / (norm_before + 1e-12)

        if quality > 0.95:
            self.successful_corrections += 1
            # Reinforce: update smoother with successful correction
            self.smoother.train([(state_seq, code)])

        return {
            "qec_code": code,
            "confidence": confidence,
            "uncertainty_bits": uncertainty,
            "max_uncertainty_bits": self.smoother.max_uncertainty(),
            "uncertainty_reduction_pct": 100 * (
                1 - uncertainty / self.smoother.max_uncertainty()
            ),
            "correction_quality": quality,
            "total_corrections": self.corrections_applied,
            "success_rate": self.successful_corrections / self.corrections_applied,
        }

    def viterbi_sequence(self, phi_sequence: list[np.ndarray]) -> list[int]:
        """
        Viterbi decoding over a sequence of quantum states.
        Finds globally optimal QEC code sequence using second-order Markov.

        V(code, t) = max_{code'} [P(code|code', φ_t) · V(code', t-1)]
        """
        T = len(phi_sequence)
        n_codes = self.cfg.n_qec_codes

        # Initialize
        V = np.full((n_codes, T), -np.inf)
        backtrack = np.zeros((n_codes, T), dtype=int)

        # t=0: use emission probability only
        phi0 = phi_sequence[0]
        seq0 = self._discretize_state(phi0.real)
        dist0 = self.smoother.predict_distribution(seq0)
        for c in range(n_codes):
            V[c, 0] = np.log(dist0.get(c, 1e-10) + 1e-10)

        # Forward pass
        for t in range(1, T):
            seq_t = self._discretize_state(phi_sequence[t].real)
            dist_t = self.smoother.predict_distribution(seq_t)

            for c in range(n_codes):
                p_emit = np.log(dist_t.get(c, 1e-10) + 1e-10)
                # Transition: penalize large code jumps
                scores = np.array([
                    V[c_prev, t-1] - 0.1 * abs(c - c_prev) + p_emit
                    for c_prev in range(n_codes)
                ])
                best_prev = int(np.argmax(scores))
                V[c, t] = scores[best_prev]
                backtrack[c, t] = best_prev

        # Backtrack
        path = []
        c = int(np.argmax(V[:, T-1]))
        for t in range(T-1, -1, -1):
            path.append(c)
            c = backtrack[c, t]
        path.reverse()
        return path

    def summary(self) -> dict:
        return {
            "total_corrections": self.corrections_applied,
            "success_rate": (
                self.successful_corrections / max(1, self.corrections_applied)
            ),
            "mean_uncertainty_bits": float(np.mean(self.uncertainty_history))
                if self.uncertainty_history else 0.0,
            "mean_uncertainty_reduction_pct": float(
                100 * (1 - np.mean(self.uncertainty_history) / self.smoother.max_uncertainty())
            ) if self.uncertainty_history else 0.0,
            "suffix_nodes": len(self.smoother.nodes),
        }
