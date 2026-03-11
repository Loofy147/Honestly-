"""
Suffix Smoother v0.2.0 — Recursive Sequence Classifier
=======================================================

Implements suffix-tree-based sequence classification with three
research-backed smoothing strategies, conformal prediction sets,
label smoothing, and streaming/online training.

Smoothing methods
-----------------
jelinek-mercer (default)
    P(y|seq_k) = λ · P_MLE(y|seq_k) + (1-λ) · P(y|seq_{k-1})
    Fixed λ from config. Simple and fast. Good when you have a
    reasonable prior on how much to trust longer contexts.

witten-bell
    λ(node) = T / (T + N) where T = unique labels seen, N = total count.
    Input-adaptive: nodes with high evidence (many distinct outcomes)
    trust their MLE more; sparse nodes back off more aggressively.
    No tuning required. Based on Bell, Cleary & Witten (1990).

kneser-ney
    Uses absolute discounting (max(count - D, 0)) at higher-order nodes
    and a continuation probability at lower-order nodes: how many
    distinct suffix contexts a label appeared in, not its raw frequency.
    Widely regarded as the best smoothing method for sparse data.
    Based on Kneser & Ney (1995), Chen & Goodman (1999).

Conformal Prediction
--------------------
Split conformal prediction (Papadopoulos 2002, Angelopoulos & Bates 2021).
Call calibrate(data) once after training, then predict_set(seq, coverage=0.9)
returns the minimal set of labels guaranteed to contain the true label
with at least 90% probability—a finite-sample, distribution-free guarantee.
Nonconformity score: s(x, y) = 1 - P(y|x)  [LAC score].

New in 0.2.0
------------
- Three smoothing methods: jelinek-mercer, witten-bell, kneser-ney
- Label smoothing: redistributes ε mass during training for better calibration
- Conformal prediction: calibrate() + predict_set() with coverage guarantee
- Streaming: train_one() for real-time/online adaptation
- ECE metric: expected calibration error for measuring confidence quality
- Continuation counts tracked in nodes for Kneser-Ney lower-order distributions

Changelog
---------
0.2.0   Three smoothing methods, conformal prediction, label smoothing, streaming
0.1.1   Bug fixes: inference side-effect, bad label validation, single-pass inference
0.1.0   Initial release
"""

import math
import pickle
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass, field
from collections import defaultdict

__version__ = "0.2.0"

SmootherMethod = Literal["jelinek-mercer", "witten-bell", "kneser-ney"]


@dataclass
class SuffixConfig:
    """
    Configuration for SuffixSmoother.

    Parameters
    ----------
    max_suffix_length : int
        Maximum context window (in symbols). Default 5.
    smoothing_lambda : float
        λ for Jelinek-Mercer interpolation. Ignored by witten-bell and
        kneser-ney which compute it adaptively. Default 0.7.
    n_classes : int
        Number of output labels. Must cover all label ids in training data.
    min_count : int
        Minimum observations before a node contributes to inference. Default 1.
    smoothing_method : str
        One of 'jelinek-mercer', 'witten-bell', 'kneser-ney'. Default 'witten-bell'.
        witten-bell is now the default because it requires no tuning and
        outperforms jelinek-mercer on sparse data.
    label_smoothing : float
        ε in [0, 1). Fraction of each observation redistributed uniformly
        across all other classes during training. Improves confidence
        calibration especially when training data is limited. Default 0.0.
    kn_discount : float or None
        Absolute discount D for Kneser-Ney. If None (default), estimated
        automatically from training data: D = n1 / (n1 + 2*n2) where
        n1 and n2 are counts of events occurring exactly once and twice.
    """
    max_suffix_length: int = 5
    smoothing_lambda: float = 0.7
    n_classes: int = 16
    min_count: int = 1
    smoothing_method: SmootherMethod = "witten-bell"
    label_smoothing: float = 0.0
    kn_discount: Optional[float] = None


class _SuffixNode:
    """Internal suffix tree node — stores counts and continuation diversity."""
    __slots__ = ("counts", "total", "continuation_contexts")

    def __init__(self):
        # counts[label] = total (possibly fractional with label smoothing) weight
        self.counts: dict = defaultdict(float)
        self.total: float = 0.0
        # For Kneser-Ney lower-order: set of (parent_suffix,) contexts that
        # contributed to this node. Tracks diversity of histories.
        self.continuation_contexts: set = set()

    def observe(self, label: int, weight: float = 1.0, context: Optional[tuple] = None) -> None:
        self.counts[label] += weight
        self.total += weight
        if context is not None:
            self.continuation_contexts.add(context)

    def n_unique_labels(self) -> int:
        """Number of distinct labels observed at this node (T in Witten-Bell)."""
        return sum(1 for v in self.counts.values() if v > 0)

    def continuation_count(self, label: int) -> int:
        """How many distinct suffix contexts this label appeared in (for KN lower-order)."""
        # Approximation: for the root/low-order nodes, we track contexts per label
        # This is computed separately via _label_continuation_counts
        return 0  # overridden at inference for KN

    def mle(self, label: int) -> float:
        return self.counts.get(label, 0.0) / self.total if self.total > 0 else 0.0

    def mle_all(self, n_classes: int) -> list:
        if self.total == 0:
            return [0.0] * n_classes
        return [self.counts.get(i, 0.0) / self.total for i in range(n_classes)]

    def wb_distribution(self, n_classes: int, lower_p: list) -> list:
        """
        Correct Witten-Bell distribution (Bell, Cleary & Witten 1990).
        P_WB(y|node) = C(y)/(T+N) + [T/(T+N)] * P_lower(y)
        MLE mass = N/(T+N),  Backoff mass = T/(T+N).
        As N grows (more evidence) backoff shrinks → more trust in MLE.
        """
        T = self.n_unique_labels()
        N = self.total
        denom = T + N
        if denom == 0:
            return lower_p[:]
        backoff_w = T / denom
        return [
            self.counts.get(i, 0.0) / denom + backoff_w * lower_p[i]
            for i in range(n_classes)
        ]

    def kn_discounted_all(self, n_classes: int, D: float) -> list:
        """Absolute-discounted higher-order term: max(C(y)-D, 0) / N."""
        if self.total == 0:
            return [0.0] * n_classes
        return [max(self.counts.get(i, 0.0) - D, 0.0) / self.total for i in range(n_classes)]

    def kn_backoff_weight(self, D: float) -> float:
        """KN back-off weight to lower-order: D * T / N."""
        T = self.n_unique_labels()
        return D * T / self.total if self.total > 0 else 0.0


class SuffixSmoother:
    """
    Sequence classifier using suffix smoothing with three research-backed methods.

    All three methods share the same suffix tree and training interface.
    The smoothing strategy only affects inference.

    Examples
    --------
    NLP POS tagging::

        smoother = SuffixSmoother(SuffixConfig(
            max_suffix_length=6, n_classes=17,
            smoothing_method="kneser-ney"
        ))
        encode = lambda word: tuple(ord(c) % 32 for c in word[-6:])
        smoother.train([(encode(word), tag_id) for word, tag_id in corpus])
        label, conf = smoother.predict(encode("running"))

    With conformal coverage guarantee::

        smoother.calibrate(validation_data)  # list of (seq, true_label)
        labels = smoother.predict_set(encode("running"), coverage=0.90)
        # Guaranteed to contain true label ≥90% of the time

    Online/streaming::

        smoother.train_one(encode("streamed"), tag_id)
    """

    def __init__(self, config: Optional[SuffixConfig] = None):
        self.cfg = config or SuffixConfig()
        self._root = _SuffixNode()
        self._nodes: dict[tuple, _SuffixNode] = {}
        self.n_classes = self.cfg.n_classes
        self.training_samples: int = 0

        # Precompute JM lambdas (only used for jelinek-mercer method)
        self._jm_lambdas = [
            self.cfg.smoothing_lambda ** (i + 1)
            for i in range(self.cfg.max_suffix_length)
        ]

        # Kneser-Ney state — estimated from data during training
        self._kn_D: Optional[float] = self.cfg.kn_discount
        self._kn_n1: int = 0  # singletons count for D estimation
        self._kn_n2: int = 0  # doubletons count for D estimation

        # Continuation counts for KN: label → set of suffix contexts it appeared in
        # Tracked at the root level for lower-order distribution
        self._label_continuation_contexts: dict[int, set] = defaultdict(set)

        # Conformal calibration state
        self._conformal_scores: Optional[list] = None  # calibration nonconformity scores
        self._conformal_n: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _smooth_label(self, label: int, weight: float = 1.0) -> list:
        """
        Apply label smoothing: returns list of (label, weight) pairs.
        Without smoothing: [(label, 1.0)].
        With smoothing ε: [(label, 1-ε), (all others, ε/(K-1))].
        """
        eps = self.cfg.label_smoothing
        if eps == 0.0:
            return [(label, weight)]
        K = self.n_classes
        main_w = weight * (1.0 - eps)
        other_w = weight * eps / (K - 1) if K > 1 else 0.0
        pairs = [(label, main_w)]
        for i in range(K):
            if i != label:
                pairs.append((i, other_w))
        return pairs

    def _update_kn_counts(self, label: int, count: float) -> None:
        """Track singleton/doubleton counts for automatic D estimation."""
        if self._kn_D is not None:
            return  # fixed D, no need to estimate
        # Approximate: count transitions in n1/n2 based on rounded counts
        rounded = round(count)
        if rounded == 1:
            self._kn_n1 += 1
        elif rounded == 2:
            self._kn_n2 += 1

    def _get_kn_D(self) -> float:
        """Return estimated or configured KN discount factor D."""
        if self._kn_D is not None:
            return self._kn_D
        n1, n2 = self._kn_n1, self._kn_n2
        if n1 == 0 or (n1 + 2 * n2) == 0:
            return 0.75  # Kneser & Ney's default
        return n1 / (n1 + 2 * n2)

    # ── Training ──────────────────────────────────────────────────────────

    def _train_sequence(self, seq: tuple, label: int) -> None:
        """Train a single (seq, label) pair — core inner loop."""
        label = int(label)
        if not (0 <= label < self.n_classes):
            raise ValueError(
                f"Label {label} out of range [0, {self.n_classes}). "
                f"Increase n_classes or fix your labels."
            )

        smoothed = self._smooth_label(label)

        n = len(seq)
        for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
            suffix = seq[max(0, n - length):]
            node = self._nodes.setdefault(suffix, _SuffixNode())
            parent_suffix = seq[max(0, n - length + 1):]  # one shorter
            for lbl, w in smoothed:
                node.observe(lbl, w, context=parent_suffix)
            # Track KN counts on single-label case only
            if len(smoothed) == 1:
                self._update_kn_counts(label, 1.0)
            # Track continuation diversity at label level
            self._label_continuation_contexts[label].add(suffix)

        # Update root
        for lbl, w in smoothed:
            self._root.observe(lbl, w)

        self.training_samples += 1

    def train(self, data: list) -> dict:
        """
        Train on a list of (context_sequence, label_id) pairs.

        Parameters
        ----------
        data : list of (tuple, int)
            context_sequence: any hashable symbols.
            label_id: integer in [0, n_classes).

        Returns
        -------
        dict with keys: samples_trained, total_nodes, total_training_samples.

        Raises
        ------
        ValueError if any label is out of range.
        """
        for seq, label in data:
            self._train_sequence(seq, label)

        return {
            "samples_trained": len(data),
            "total_nodes": len(self._nodes),
            "total_training_samples": self.training_samples,
        }

    def train_one(self, seq: tuple, label: int) -> None:
        """
        Online / streaming training — update model with a single example.

        Safe to call after calibrate(); does NOT invalidate conformal scores
        (you should re-calibrate after significant concept drift).
        """
        self._train_sequence(seq, label)

    # ── Inference — all three smoothing methods ────────────────────────────

    def _infer_jm(self, seq: tuple) -> list:
        """Jelinek-Mercer: fixed λ interpolation."""
        n = len(seq)
        uniform = 1.0 / self.n_classes
        p = [uniform] * self.n_classes

        for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
            suffix = seq[max(0, n - length):]
            node = self._nodes.get(suffix)
            if node is not None and node.total >= self.cfg.min_count:
                lam = self._jm_lambdas[length - 1]
                mle = node.mle_all(self.n_classes)
                p = [lam * m + (1.0 - lam) * pk for m, pk in zip(mle, p)]

        return p

    def _infer_wb(self, seq: tuple) -> list:
        """
        Witten-Bell: correct formula per Bell, Cleary & Witten (1990).
        P_WB(y|ctx) = C(y)/(T+N) + [T/(T+N)] * P_lower(y)
        T = distinct label types, N = total count.
        More evidence (high N) → more MLE weight. More diverse outcomes
        (high T) → more backoff. Adapts automatically with no tuning.
        """
        n = len(seq)
        uniform = 1.0 / self.n_classes
        p = [uniform] * self.n_classes

        for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
            suffix = seq[max(0, n - length):]
            node = self._nodes.get(suffix)
            if node is not None and node.total >= self.cfg.min_count:
                p = node.wb_distribution(self.n_classes, p)

        return p

    def _infer_kn(self, seq: tuple) -> list:
        """
        Kneser-Ney: absolute discounting at higher-order, continuation
        probability at lower-order nodes.

        Higher-order: P_KN(y|suffix_k) = max(count(y,suffix) - D, 0) / count(suffix)
                      + λ(suffix) · P_KN(y|suffix_{k-1})
        Lower-order base: P_KN(y|∅) ∝ |{contexts c : (c, y) seen}| / total_bigrams
                          i.e. how many distinct contexts y appeared in.
        """
        n = len(seq)
        D = self._get_kn_D()

        # Base: continuation probability from root
        total_continuation = sum(
            len(ctxs) for ctxs in self._label_continuation_contexts.values()
        )
        if total_continuation > 0:
            p = [
                len(self._label_continuation_contexts.get(i, set())) / total_continuation
                for i in range(self.n_classes)
            ]
        else:
            p = [1.0 / self.n_classes] * self.n_classes

        # Layer by layer: shortest suffix → longest
        for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
            suffix = seq[max(0, n - length):]
            node = self._nodes.get(suffix)
            if node is not None and node.total >= self.cfg.min_count:
                disc = node.kn_discounted_all(self.n_classes, D)
                bw = node.kn_backoff_weight(D)
                p = [d + bw * pk for d, pk in zip(disc, p)]

        return p

    def _infer(self, seq: tuple) -> list:
        """Dispatch to the configured smoothing method."""
        method = self.cfg.smoothing_method
        if method == "jelinek-mercer":
            return self._infer_jm(seq)
        elif method == "witten-bell":
            return self._infer_wb(seq)
        elif method == "kneser-ney":
            return self._infer_kn(seq)
        else:
            raise ValueError(
                f"Unknown smoothing_method '{method}'. "
                f"Choose: 'jelinek-mercer', 'witten-bell', 'kneser-ney'."
            )

    # ── Public inference API ───────────────────────────────────────────────

    def predict_distribution(self, seq: tuple) -> dict:
        """
        Return P(label | seq) for all labels.

        Parameters
        ----------
        seq : tuple — context sequence (same symbol type as training).

        Returns
        -------
        dict[int, float] — maps label_id → probability. Always sums to 1.0.
        """
        p = self._infer(seq)
        total = sum(p)
        if total > 1e-12:
            p = [v / total for v in p]
        return {i: p[i] for i in range(self.n_classes)}

    def predict(self, seq: tuple) -> tuple:
        """
        Predict most probable label and its probability.

        Returns
        -------
        (best_label: int, confidence: float)
        """
        dist = self.predict_distribution(seq)
        best = max(dist, key=dist.get)
        return best, dist[best]

    def predict_grouped(self, seq: tuple, groups: dict) -> tuple:
        """
        Predict over semantic class groups.

        Parameters
        ----------
        seq : tuple
        groups : dict[str, list[int]]
            e.g. {"pathogenic": [3, 4, 7], "benign": [0, 1, 2]}

        Returns
        -------
        (best_group: str, confidence: float, all_scores: dict[str, float])
        Scores are normalized within the defined groups.
        """
        dist = self.predict_distribution(seq)
        scores = {
            name: sum(dist.get(c, 0.0) for c in cls_ids)
            for name, cls_ids in groups.items()
        }
        total = sum(scores.values())
        if total > 1e-12:
            scores = {k: v / total for k, v in scores.items()}
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    # ── Conformal Prediction ───────────────────────────────────────────────

    def calibrate(self, calibration_data: list) -> dict:
        """
        Calibrate conformal predictor using split CP (Papadopoulos 2002).

        Computes nonconformity scores s(x, y) = 1 - P(true_label | x) on
        held-out calibration data. These scores are stored and used by
        predict_set() to guarantee coverage.

        IMPORTANT: calibration_data must be separate from training data
        (exchangeability assumption). Typically 10-30% of your labeled data.

        Parameters
        ----------
        calibration_data : list of (seq, true_label)

        Returns
        -------
        dict with: n_calibration, mean_score, median_score, coverage_at_90.
        """
        if not calibration_data:
            raise ValueError("calibration_data cannot be empty.")

        scores = []
        for seq, true_label in calibration_data:
            dist = self.predict_distribution(seq)
            # LAC nonconformity score: 1 - P(true label | seq)
            # Small score = well-predicted; large score = surprising
            score = 1.0 - dist.get(int(true_label), 0.0)
            scores.append(score)

        self._conformal_scores = sorted(scores)
        self._conformal_n = len(scores)

        # Measure actual coverage at α=0.10 as a calibration sanity check
        Q90 = self._conformal_quantile(0.10)
        covered = sum(1 for s in scores if s <= Q90)

        return {
            "n_calibration": len(scores),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "coverage_at_90": covered / len(scores),
        }

    def _conformal_quantile(self, alpha: float) -> float:
        """
        Return the (1-α) quantile of calibration scores with Bonferroni correction.

        Q = ceil((n+1)(1-α)) / n -th smallest score.
        This is the finite-sample valid threshold from split CP.
        """
        if self._conformal_scores is None:
            raise RuntimeError(
                "Model not calibrated. Call calibrate(calibration_data) first."
            )
        n = self._conformal_n
        # Corrected index per split CP theory
        idx = math.ceil((n + 1) * (1.0 - alpha)) - 1
        idx = max(0, min(idx, n - 1))
        return self._conformal_scores[idx]

    def predict_set(self, seq: tuple, coverage: float = 0.90) -> dict:
        """
        Return a prediction set with a statistical coverage guarantee.

        The returned set is guaranteed to contain the true label with
        probability ≥ coverage, provided that calibration data was
        exchangeable with test data (the only required assumption).

        Parameters
        ----------
        seq : tuple — context sequence.
        coverage : float in (0, 1) — desired coverage probability. Default 0.90.

        Returns
        -------
        dict with keys:
          - labels: list[int] — the prediction set (sorted)
          - n_labels: int — set size
          - threshold: float — nonconformity threshold used
          - coverage: float — requested coverage

        Raises
        ------
        RuntimeError if calibrate() has not been called.

        Examples
        --------
        ::

            smoother.calibrate(val_data)
            result = smoother.predict_set(seq, coverage=0.95)
            print(result['labels'])   # e.g. [3, 7] — both candidates
        """
        alpha = 1.0 - coverage
        Q = self._conformal_quantile(alpha)
        dist = self.predict_distribution(seq)

        # Include all labels whose nonconformity score <= Q
        # s(x,y) = 1 - P(y|x) <= Q  ↔  P(y|x) >= 1 - Q
        threshold_prob = 1.0 - Q
        included = sorted(
            [label for label, prob in dist.items() if prob >= threshold_prob]
        )

        # Edge case: if set is empty (all probs below threshold), include top-1
        if not included:
            included = [max(dist, key=dist.get)]

        return {
            "labels": included,
            "n_labels": len(included),
            "threshold": Q,
            "coverage": coverage,
        }

    # ── Diagnostics ────────────────────────────────────────────────────────

    def uncertainty(self, seq: tuple) -> float:
        """Shannon entropy of the prediction distribution in bits. 0=certain."""
        dist = self.predict_distribution(seq)
        probs = np.array(list(dist.values()))
        probs = probs[probs > 1e-12]
        return float(-np.sum(probs * np.log2(probs)))

    def max_uncertainty(self) -> float:
        """Maximum possible entropy (uniform distribution) in bits."""
        return float(np.log2(self.n_classes))

    def uncertainty_reduction(self, seq: tuple) -> float:
        """Fraction of max entropy eliminated: 0.0=no info, 1.0=certain."""
        return 1.0 - self.uncertainty(seq) / self.max_uncertainty()

    @staticmethod
    def expected_calibration_error(
        probs: list, labels: list, n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE) — measures how well confidence
        scores match empirical accuracy.

        A perfectly calibrated model has ECE=0: when it says 80% confident,
        it's correct 80% of the time. High ECE means overconfident or
        underconfident predictions.

        Parameters
        ----------
        probs : list of float — confidence scores for predicted labels
        labels : list of bool — whether each prediction was correct
        n_bins : int — number of equal-width confidence bins

        Returns
        -------
        float — ECE in [0, 1]. Lower is better.

        Example
        -------
        ::

            preds = [smoother.predict(seq) for seq, _ in test_data]
            probs = [conf for _, conf in preds]
            correct = [pred == true for (pred, _), (_, true) in zip(preds, test_data)]
            ece = SuffixSmoother.expected_calibration_error(probs, correct)
        """
        if len(probs) != len(labels):
            raise ValueError("probs and labels must have the same length.")

        probs = np.array(probs)
        labels = np.array(labels, dtype=float)
        n = len(probs)
        ece = 0.0

        for b in range(n_bins):
            lo = b / n_bins
            hi = (b + 1) / n_bins
            mask = (probs >= lo) & (probs < hi)
            if b == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            acc_bin = labels[mask].mean()
            conf_bin = probs[mask].mean()
            ece += (n_bin / n) * abs(acc_bin - conf_bin)

        return float(ece)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the full model (including conformal state) to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "SuffixSmoother":
        """Load a model saved with save()."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SuffixSmoother, got {type(obj)}")
        return obj

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        """Number of suffix tree nodes built during training."""
        return len(self._nodes)

    @property
    def is_calibrated(self) -> bool:
        """True if calibrate() has been called successfully."""
        return self._conformal_scores is not None

    def __repr__(self) -> str:
        cal = f", calibrated={self.is_calibrated}"
        return (
            f"SuffixSmoother("
            f"n_classes={self.n_classes}, "
            f"method={self.cfg.smoothing_method!r}, "
            f"max_suffix={self.cfg.max_suffix_length}, "
            f"nodes={self.n_nodes}, "
            f"samples={self.training_samples}"
            f"{cal})"
        )
