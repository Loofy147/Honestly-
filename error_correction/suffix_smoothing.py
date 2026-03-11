import numpy as np
from typing import Optional, Any
from suffix_smoother import SuffixSmoother, SuffixConfig

class QuantumSuffixSmoother(SuffixSmoother):
    """
    Quantum-specialized wrapper for SuffixSmoother.
    Maintains compatibility with existing QEC code.
    """
    def __init__(self, config: Optional[Any] = None):
        # Handle legacy n_qec_codes if present
        if config and hasattr(config, 'n_qec_codes'):
            config.n_classes = config.n_qec_codes
        super().__init__(config)

    @property
    def nodes(self):
        # Compatibility with old code that accessed .nodes
        return self._nodes

    def best_correction(self, state_seq: tuple) -> tuple[int, float]:
        """Alias for predict() in QEC context."""
        return self.predict(state_seq)

    def predict_probability(self, state_seq: tuple, code: int) -> float:
        """Compatibility with old predict_probability API."""
        dist = self.predict_distribution(state_seq)
        return dist.get(code, 0.0)

class QuantumErrorCorrector:
    """
    Full QEC system using suffix smoothing for code selection.
    """

    def __init__(self, config: Optional[Any] = None):
        self.cfg = config or SuffixConfig(n_classes=16)
        if hasattr(self.cfg, 'n_qec_codes'):
            self.cfg.n_classes = self.cfg.n_qec_codes

        self.smoother = QuantumSuffixSmoother(self.cfg)
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
        rng = np.random.default_rng(seed)
        dim = 4
        sequences = []
        for _ in range(n_training):
            phi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            phi /= np.linalg.norm(phi) + 1e-12
            state_seq = self._discretize_state(phi.real)
            dominant_error = int(np.argmax(np.abs(phi.real)))
            error_magnitude = float(np.max(np.abs(phi.imag)))
            code = (dominant_error * 4 + int(error_magnitude * 4)) % self.cfg.n_classes
            sequences.append((state_seq, code))

        result = self.smoother.train(sequences)
        return result

    def correct(self, phi: np.ndarray) -> dict:
        state_seq = self._discretize_state(phi.real)
        code, confidence = self.smoother.best_correction(state_seq)
        uncertainty = self.smoother.uncertainty(state_seq)

        self.uncertainty_history.append(uncertainty)
        self.corrections_applied += 1

        corrected_phi = phi.copy()
        if code % 2 == 1:
            corrected_phi = corrected_phi * np.exp(1j * np.pi * code / self.cfg.n_classes)

        norm_before = float(np.linalg.norm(phi))
        norm_after = float(np.linalg.norm(corrected_phi))
        quality = 1.0 - abs(norm_before - norm_after) / (norm_before + 1e-12)

        if quality > 0.95:
            self.successful_corrections += 1
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
        T = len(phi_sequence)
        n_codes = self.cfg.n_classes
        V = np.full((n_codes, T), -np.inf)
        backtrack = np.zeros((n_codes, T), dtype=int)

        phi0 = phi_sequence[0]
        seq0 = self._discretize_state(phi0.real)
        dist0 = self.smoother.predict_distribution(seq0)
        for c in range(n_codes):
            V[c, 0] = np.log(dist0.get(c, 1e-10) + 1e-10)

        for t in range(1, T):
            seq_t = self._discretize_state(phi_sequence[t].real)
            dist_t = self.smoother.predict_distribution(seq_t)
            for c in range(n_codes):
                p_emit = np.log(dist_t.get(c, 1e-10) + 1e-10)
                scores = np.array([
                    V[c_prev, t-1] - 0.1 * abs(c - c_prev) + p_emit
                    for c_prev in range(n_codes)
                ])
                best_prev = int(np.argmax(scores))
                V[c, t] = scores[best_prev]
                backtrack[c, t] = best_prev

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
            "suffix_nodes": self.smoother.n_nodes,
        }
