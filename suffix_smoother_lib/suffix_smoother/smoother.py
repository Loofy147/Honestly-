import numpy as np
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class SuffixConfig:
    """Configuration for Suffix Smoother."""
    max_suffix_length: int = 5
    smoothing_lambda: float = 0.7
    n_classes: int = 16  # Number of output classes/labels
    min_count: int = 1    # Minimum observations to trust a suffix level

class SuffixNode:
    """A node in the suffix tree representing a specific sequence context."""
    def __init__(self, depth: int = 0):
        self.depth = depth
        self.counts: dict[int, int] = defaultdict(int)
        self.total: int = 0

    def observe(self, label: int):
        self.counts[label] += 1
        self.total += 1

    def mle_probability(self, label: int) -> float:
        """Maximum Likelihood Estimate: P(label | context)."""
        if self.total == 0:
            return 0.0
        return self.counts[label] / self.total

    def uniform_probability(self, n_classes: int) -> float:
        """Fallback: Uniform distribution."""
        return 1.0 / n_classes

class SuffixSmoother:
    """
    General-purpose sequence classifier using recursive suffix smoothing.

    Implements a modified Witten-Bell or Jelinek-Mercer style smoothing:
    P(label | sequence) = λ * P_MLE(label | longest_suffix) + (1-λ) * P(label | shorter_suffix)
    """

    def __init__(self, config: Optional[SuffixConfig] = None):
        self.cfg = config or SuffixConfig()
        self.root = SuffixNode(depth=0)
        self.nodes: dict[tuple, SuffixNode] = {}
        self.n_classes = self.cfg.n_classes
        self.training_samples: int = 0

        # Adaptive smoothing weights per suffix level
        self.lambdas: list[float] = [
            self.cfg.smoothing_lambda ** (i + 1)
            for i in range(self.cfg.max_suffix_length)
        ]

    def _get_or_create_node(self, suffix: tuple) -> SuffixNode:
        """Get or create a suffix tree node."""
        if suffix not in self.nodes:
            self.nodes[suffix] = SuffixNode(depth=len(suffix))
        return self.nodes[suffix]

    def train(self, sequences: list[tuple[tuple, int]]) -> dict:
        """
        Train on (context_sequence, label) pairs.
        context_sequence: tuple of discrete symbols
        label: integer class label
        """
        for state_seq, label in sequences:
            n = len(state_seq)
            # Update all suffix nodes
            for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
                suffix = state_seq[max(0, n - length):]
                node = self._get_or_create_node(suffix)
                node.observe(label)

            # Update root (empty suffix)
            self.root.observe(label)
            self.training_samples += 1

        return {
            "samples_trained": len(sequences),
            "total_nodes": len(self.nodes),
            "total_training_samples": self.training_samples,
        }

    def predict_probability(self, state_seq: tuple, label: int) -> float:
        """Compute P(label | state_seq) using recursive suffix smoothing."""
        n = len(state_seq)
        p_current = self.root.uniform_probability(self.n_classes)

        for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
            suffix = state_seq[max(0, n - length):]
            node = self._get_or_create_node(suffix)
            lam = self.lambdas[length - 1]

            if node.total >= self.cfg.min_count:
                p_mle = node.mle_probability(label)
                p_current = lam * p_mle + (1 - lam) * p_current

        return float(p_current)

    def predict_distribution(self, state_seq: tuple) -> dict[int, float]:
        """Return full probability distribution over all classes."""
        probs = {
            label: self.predict_probability(state_seq, label)
            for label in range(self.n_classes)
        }
        total = sum(probs.values())
        if total > 1e-12:
            probs = {k: v / total for k, v in probs.items()}
        return probs

    def predict(self, state_seq: tuple) -> tuple[int, float]:
        """Return the most probable label and its probability."""
        dist = self.predict_distribution(state_seq)
        best_label = max(dist, key=dist.get)
        confidence = dist[best_label]
        return best_label, confidence

    def uncertainty(self, state_seq: tuple) -> float:
        """Measure uncertainty as entropy of distribution over labels (bits)."""
        dist = self.predict_distribution(state_seq)
        probs = np.array(list(dist.values()))
        probs = probs[probs > 1e-12]
        return float(-np.sum(probs * np.log2(probs)))

    def max_uncertainty(self) -> float:
        """Maximum possible uncertainty (uniform distribution)."""
        return float(np.log2(self.n_classes))
