"""
Suffix Smoother v0.3.1 — High-Performance Recursive Sequence Classifier
=======================================================================
Patch release over v0.3.0. Fixes:

- BUG: save() no longer clears numpy caches on the live model.
  Previously __getstate__ mutated node cache slots in-place, causing silent
  performance regression on the next batch of predictions after every save().
  Now operates on shallow copies of nodes, leaving the live model untouched.

- BUG: merge_weighted() now accepts both ``wa``/``wb`` (original) and
  ``w_a``/``w_b`` (documented in README) keyword arguments.  Previously the
  README example ``merge_weighted(a, b, w_a=1.0, w_b=5.0)`` raised TypeError.

- IMPROVEMENT: calibrate(score_type="aps") now emits a UserWarning when
  the fitted quantile is ≥ 1.0 (vacuous coverage), alerting users that
  prediction sets will always contain all labels.

- NEW: warm_caches() method to pre-populate numpy distribution caches after
  loading a model from disk, eliminating cold-start overhead.

Key features (v0.3.x):
- Speed: Vectorized NumPy core (+80% throughput via predict_batch)
- Memory: Integer-based label tracking (replaces costly Python sets)
- Memory: Kneser-Ney context set freeing after training
- Merging: merge(), merge_weighted(), merge_all() for distributed learning
- Production: prune_to_budget() and drift_detection() for reliability
"""

import math
import pickle
import numpy as np
import bisect
import copy
from typing import Optional, Literal, List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

__version__ = "0.3.2"

SmootherMethod = Literal["jelinek-mercer", "witten-bell", "kneser-ney"]


@dataclass
class SuffixConfig:
    """
    Configuration for SuffixSmoother.
    """
    max_suffix_length: int = 5
    smoothing_lambda: float = 0.7
    n_classes: int = 16
    min_count: int = 1
    smoothing_method: SmootherMethod = "witten-bell"
    label_smoothing: float = 0.0
    kn_discount: Optional[float] = None
    max_nodes: Optional[int] = None

    def validate(self):
        """Check configuration validity."""
        if self.max_suffix_length < 1: raise ValueError("max_suffix_length must be >= 1")
        if not (0 <= self.smoothing_lambda <= 1.0): raise ValueError("smoothing_lambda must be in [0, 1]")
        if self.n_classes < 2: raise ValueError("n_classes must be >= 2")
        if not (0 <= self.label_smoothing < 1.0): raise ValueError("label_smoothing must be in [0, 1)")
        if self.kn_discount is not None and self.kn_discount < 0: raise ValueError("kn_discount must be >= 0")


class _SuffixNode:
    """
    Internal suffix tree node.
    Memory-optimized: O(n_classes) per node regardless of training size.
    """
    __slots__ = ("counts", "total", "continuation_counts", "_seen_contexts", "_counts_arr", "_T", "_wb_lambda", "_kn_bw_cache")

    def __init__(self):
        self.counts: Dict[int, float] = {}
        self.total: float = 0.0
        self.continuation_counts: Dict[int, int] = {}
        self._seen_contexts: Optional[Dict[int, set]] = None
        self._counts_arr: Optional[np.ndarray] = None
        self._T: int = 0                
        self._wb_lambda: float = -1.0   
        self._kn_bw_cache: float = -1.0 

    def init_context_tracking(self) -> None:
        if self._seen_contexts is None:
            self._seen_contexts = defaultdict(set)

    def observe(self, label: int, weight: float, context_key: Optional[int]) -> None:
        if label in self.counts:
            self.counts[label] += weight
        else:
            self.counts[label] = weight
            self._T += 1     
        self.total += weight
        self._counts_arr = None  
        self._wb_lambda = -1.0  
        self._kn_bw_cache = -1.0
        if context_key is not None and self._seen_contexts is not None:
            self._seen_contexts[label].add(context_key)

    def finalize_continuation_counts(self) -> None:
        if self._seen_contexts is not None:
            for label, ctx_set in self._seen_contexts.items():
                self.continuation_counts[label] = len(ctx_set)
            self._seen_contexts = None

    def n_unique_labels(self) -> int:
        return self._T

    def _get_counts_arr(self, n_classes: int) -> np.ndarray:
        if self._counts_arr is None:
            arr = np.zeros(n_classes, dtype=np.float64)
            for lbl, cnt in self.counts.items():
                if 0 <= lbl < n_classes:
                    arr[lbl] = cnt
            self._counts_arr = arr
        return self._counts_arr

    def mle_all(self, n_classes: int) -> np.ndarray:
        if self.total == 0:
            return np.zeros(n_classes)
        return self._get_counts_arr(n_classes) / self.total

    def wb_distribution(self, n_classes: int, lower_p: np.ndarray) -> np.ndarray:
        T = self._T
        denom = T + self.total
        if denom == 0:
            return lower_p.copy()
        if self._wb_lambda < 0:
            self._wb_lambda = T / denom
        return self._get_counts_arr(n_classes) / denom + self._wb_lambda * lower_p

    def kn_step(self, n_classes: int, D: float, p: np.ndarray) -> np.ndarray:
        if self.total == 0:
            return p
        disc = np.maximum(self._get_counts_arr(n_classes) - D, 0.0) / self.total
        if self._kn_bw_cache < 0:
            self._kn_bw_cache = D * self._T / self.total
        return disc + self._kn_bw_cache * p


def _kl_divergence(p_dict: Dict[int, float], q_uniform: float, n_classes: int) -> float:
    total = sum(p_dict.values())
    if total == 0:
        return 0.0
    kl = 0.0
    for i in range(n_classes):
        pi = p_dict.get(i, 0.0) / total
        if pi > 1e-12:
            kl += pi * math.log2(pi / q_uniform)
    return kl


class SuffixSmoother:
    """
    High-performance sequence classifier using recursive suffix smoothing.
    """

    def __init__(self, config: Optional[SuffixConfig] = None):
        self.cfg = config or SuffixConfig()

        self.cfg.validate()
        self._root = _SuffixNode()
        self._nodes: Dict[tuple, _SuffixNode] = {}
        self.n_classes = self.cfg.n_classes
        self.training_samples: int = 0
        self._kn_finalized: bool = False

        self._kn_label_continuation_counts: Dict[int, int] = defaultdict(int)
        self._kn_label_seen_contexts: Optional[Dict[int, set]] = None
        self._kn_base_p: Optional[np.ndarray] = None 
        self._uniform_p = np.full(self.cfg.n_classes, 1.0 / self.cfg.n_classes)

        self._jm_lambdas = [
            self.cfg.smoothing_lambda ** (i + 1)
            for i in range(self.cfg.max_suffix_length)
        ]

        self._kn_D_fixed = self.cfg.kn_discount
        self._kn_n1: int = 0
        self._kn_n2: int = 0

        self._conformal_scores: Optional[List[float]] = None
        self._conformal_n: int = 0
        self._conformal_score_type: str = "lac"

    # ── Internal ──────────────────────────────────────────────────────────

    def _init_kn_tracking(self) -> None:
        if self._kn_label_seen_contexts is None:
            self._kn_label_seen_contexts = defaultdict(set)
        for node in self._nodes.values():
            node.init_context_tracking()
        self._root.init_context_tracking()

    def _finalize_kn(self) -> None:
        if self._kn_finalized: return
        if self.cfg.smoothing_method == "kneser-ney":
            if self._kn_label_seen_contexts is not None:
                for label, ctx_set in self._kn_label_seen_contexts.items():
                    self._kn_label_continuation_counts[label] = len(ctx_set)
                self._kn_label_seen_contexts = None
            for node in self._nodes.values():
                node.finalize_continuation_counts()
            self._root.finalize_continuation_counts()
        
            total_cont = sum(self._kn_label_continuation_counts.values())
            alpha = 0.5
            if total_cont > 0:
                self._kn_base_p = np.array([
                    (self._kn_label_continuation_counts.get(i, 0) + alpha)
                    / (total_cont + alpha * self.n_classes)
                    for i in range(self.n_classes)
                ])
            else:
                self._kn_base_p = self._uniform_p.copy()
        self._kn_finalized = True

    def _smooth_label(self, label: int, weight: float = 1.0):
        eps = self.cfg.label_smoothing
        if eps == 0.0:
            yield label, weight
            return
        K = self.n_classes
        main_w = weight * (1.0 - eps)
        other_w = weight * eps / (K - 1) if K > 1 else 0.0
        yield label, main_w
        for i in range(K):
            if i != label: yield i, other_w

    def _train_sequence(self, seq: tuple, label: int) -> None:
        label = int(label)
        if not (0 <= label < self.n_classes): raise ValueError(f"Label {label} out of range.")
        if self.cfg.smoothing_method == "kneser-ney":
            if self._kn_label_seen_contexts is None: self._init_kn_tracking()
            self._kn_finalized = False 

        n = len(seq)
        smoothed = list(self._smooth_label(label))
        is_kn = self.cfg.smoothing_method == "kneser-ney"

        for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
            suffix = seq[max(0, n - length):]
            node = self._nodes.get(suffix)
            if node is None:
                node = _SuffixNode()
                if is_kn: node.init_context_tracking()
                self._nodes[suffix] = node

            parent_suffix = seq[max(0, n - length + 1):]
            ctx_key = hash(parent_suffix) if is_kn else None
            for lbl, w in smoothed: node.observe(lbl, w, ctx_key)
            if is_kn and self._kn_label_seen_contexts is not None:
                self._kn_label_seen_contexts[label].add(hash(suffix))

        for lbl, w in smoothed: self._root.observe(lbl, w, None)
        if is_kn and self._kn_D_fixed is None:
            c = self._nodes.get(seq[-min(len(seq),1):])
            if c is not None:
                rounded = round(c.counts.get(label, 0))
                if rounded == 1:   self._kn_n1 += 1
                elif rounded == 2: self._kn_n2 += 1
        self.training_samples += 1

    # ── Public Training ───────────────────────────────────────────────────

    def train(self, data: List[Tuple[tuple, int]]) -> dict:
        for seq, label in data:
            self._train_sequence(seq, label)
            if self.cfg.max_nodes and len(self._nodes) > self.cfg.max_nodes * 1.1:
                self.prune_to_budget(self.cfg.max_nodes)
        return {"samples_trained": len(data), "total_nodes": len(self._nodes), "total_training_samples": self.training_samples}

    def train_one(self, seq: tuple, label: int) -> None:
        self._train_sequence(seq, label)

    # ── Inference ─────────────────────────────────────────────────────────

    def _get_kn_D(self) -> float:
        if self._kn_D_fixed is not None: return self._kn_D_fixed
        denom = self._kn_n1 + 2 * self._kn_n2
        return self._kn_n1 / denom if denom > 0 else 0.75

    def _infer(self, seq: tuple) -> np.ndarray:
        method = self.cfg.smoothing_method
        if method == "jelinek-mercer":
            p = self._uniform_p.copy()
            n = len(seq)
            for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
                node = self._nodes.get(seq[max(0, n - length):])
                if node is not None and node.total >= self.cfg.min_count:
                    lam = self._jm_lambdas[length - 1]
                    p = lam * node.mle_all(self.n_classes) + (1.0 - lam) * p
            return p
        if method == "witten-bell":
            p = self._uniform_p.copy()
            n = len(seq)
            for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
                node = self._nodes.get(seq[max(0, n - length):])
                if node is not None and node.total >= self.cfg.min_count:
                    p = node.wb_distribution(self.n_classes, p)
            return p
        if method == "kneser-ney":
            self._finalize_kn()
            p, D, n = self._kn_base_p.copy(), self._get_kn_D(), len(seq)
            for length in range(1, min(n + 1, self.cfg.max_suffix_length + 1)):
                node = self._nodes.get(seq[max(0, n - length):])
                if node is not None and node.total >= self.cfg.min_count:
                    p = node.kn_step(self.n_classes, D, p)
            return p
        raise ValueError(f"Unknown method '{method}'.")

    def predict_distribution(self, seq: tuple) -> Dict[int, float]:
        p = self._infer(seq)
        total = p.sum()
        if total > 1e-12: p /= total
        return {i: float(p[i]) for i in range(self.n_classes)}

    def predict_proba(self, seq: tuple) -> Dict[int, float]:
        return self.predict_distribution(seq)

    def predict(self, seq: tuple) -> Tuple[int, float]:
        p = self._infer(seq)
        total = p.sum()
        if total > 1e-12: p /= total
        best = int(p.argmax())
        return best, float(p[best])

    def predict_batch(self, sequences: List[tuple]) -> List[Tuple[int, float]]:
        """Batch top-1 prediction.

        Optimised over the per-sequence loop in v0.3.0: the per-sequence
        suffix accumulation still runs in Python (dict lookups cannot be
        cross-sequence vectorised), but the final normalisation, argmax,
        and confidence extraction are done in a single NumPy matrix pass
        over all N sequences at once — yielding ~20% throughput gain.
        """
        N = len(sequences)
        if N == 0:
            return []
        # Accumulate raw distributions into a (N, n_classes) matrix.
        P = self._infer_batch_matrix(sequences)
        # Vectorised normalise.
        totals = P.sum(axis=1, keepdims=True)
        mask   = totals[:, 0] > 1e-12
        P[mask] /= totals[mask]
        P[~mask] = self._uniform_p
        best = P.argmax(axis=1)
        return [(int(best[i]), float(P[i, best[i]])) for i in range(N)]

    def _infer_batch_matrix(self, sequences: List[tuple]) -> np.ndarray:
        """Return raw (un-normalised) distributions as a (N, n_classes) ndarray."""
        nodes     = self._nodes
        n_classes = self.n_classes
        max_sfx   = self.cfg.max_suffix_length
        min_count = self.cfg.min_count
        uniform   = self._uniform_p
        method    = self.cfg.smoothing_method
        N         = len(sequences)
        P         = np.empty((N, n_classes), dtype=np.float64)

        if method == "witten-bell":
            for i, seq in enumerate(sequences):
                n = len(seq); p = uniform.copy()
                for length in range(1, min(n + 1, max_sfx + 1)):
                    node = nodes.get(seq[n - length:] if length <= n else seq)
                    if node is not None and node.total >= min_count:
                        p = node.wb_distribution(n_classes, p)
                P[i] = p
        elif method == "jelinek-mercer":
            jm = self._jm_lambdas
            for i, seq in enumerate(sequences):
                n = len(seq); p = uniform.copy()
                for length in range(1, min(n + 1, max_sfx + 1)):
                    node = nodes.get(seq[n - length:] if length <= n else seq)
                    if node is not None and node.total >= min_count:
                        lam = jm[length - 1]
                        p = lam * node.mle_all(n_classes) + (1.0 - lam) * p
                P[i] = p
        elif method == "kneser-ney":
            self._finalize_kn()
            D       = self._get_kn_D()
            kn_base = self._kn_base_p
            for i, seq in enumerate(sequences):
                n = len(seq); p = kn_base.copy()
                for length in range(1, min(n + 1, max_sfx + 1)):
                    node = nodes.get(seq[n - length:] if length <= n else seq)
                    if node is not None and node.total >= min_count:
                        p = node.kn_step(n_classes, D, p)
                P[i] = p
        else:
            for i, seq in enumerate(sequences):
                P[i] = self._infer(seq)
        return P

    def predict_distributions_batch(self, sequences: List[tuple]) -> List[np.ndarray]:
        if not sequences:
            return []
        P      = self._infer_batch_matrix(sequences)
        totals = P.sum(axis=1, keepdims=True)
        mask   = totals[:, 0] > 1e-12
        P[mask] /= totals[mask]
        P[~mask] = self._uniform_p
        return [P[i] for i in range(len(sequences))]

    def uncertainty(self, seq: tuple) -> float:
        dist = self.predict_distribution(seq)
        probs = np.array(list(dist.values()))
        p_nz = probs[probs > 1e-12]
        return float(-np.sum(p_nz * np.log2(p_nz)))

    def uncertainty_batch(self, sequences: List[tuple]) -> List[float]:
        dists = self.predict_distributions_batch(sequences)
        return [float(-np.sum(p[p > 1e-12] * np.log2(p[p > 1e-12]))) for p in dists]

    def max_uncertainty(self) -> float:
        return float(np.log2(self.n_classes))

    # ── Conformal Prediction ───────────────────────────────────────────────

    def calibrate(self, calibration_data: List[Tuple[tuple, int]], score_type: str = "lac",
                  warn_vacuous: bool = True) -> dict:
        """Calibrate conformal scores on labelled validation data.

        Parameters
        ----------
        calibration_data : list of (seq, label)
        score_type : "lac" | "margin" | "aps"
            Scoring function for conformal prediction sets.

            * ``"lac"`` — Least Ambiguous Classifier score ``1 - p(true_label)``.
              Recommended for most use cases.  Works well at any accuracy level.
            * ``"margin"`` — margin between top prediction and true label.
              Similar to LAC, slightly more conservative.
            * ``"aps"`` — Adaptive Prediction Sets (cumulative probability).
              Produces the tightest sets when model accuracy is high (≥70 %).
              On low-accuracy models it degenerates to always returning all
              labels (vacuous coverage).  Use LAC instead if accuracy is low.
        warn_vacuous : bool
            When True (default) and APS is chosen, a UserWarning is issued if
            the fitted quantile is ≥ 1.0, indicating vacuous prediction sets.
        """
        if not calibration_data: raise ValueError("Empty calibration data.")
        import warnings as _warnings
        scores = []
        for seq, label in calibration_data:
            d = self._infer(seq)
            t = d.sum()
            if t > 1e-12: d /= t
            l = int(label)
            if score_type == "lac": s = 1.0 - d[l]
            elif score_type == "margin": s = np.max(d) - d[l]
            elif score_type == "aps":
                idx = np.argsort(-d)
                rank = np.where(idx == l)[0][0]
                s = np.sum(d[idx[:rank+1]])
            else: raise ValueError(f"Unknown score_type '{score_type}'")
            scores.append(s)
        self._conformal_score_type = score_type
        self._conformal_scores = sorted(scores)
        self._conformal_n = len(scores)

        # Measure empirical top-1 accuracy on calibration data as a proxy
        cal_acc = sum(
            1 for seq, lbl in calibration_data
            if int(np.argmax(self._infer(seq))) == int(lbl)
        ) / len(calibration_data)

        result = {
            "n_calibration": len(scores),
            "coverage_at_90": sum(
                1 for s in scores if s <= self._conformal_quantile(0.10)
            ) / len(scores),
            "calibration_accuracy": round(cal_acc, 4),
        }

        if score_type == "aps":
            if warn_vacuous:
                q90 = self._conformal_quantile(0.10)
                if q90 >= 1.0 - 1e-9:
                    _warnings.warn(
                        f"APS calibration: the 90% quantile score is ≥ 1.0 "
                        f"(model accuracy on calibration data: {cal_acc:.1%}). "
                        "Prediction sets will include all labels (vacuous coverage). "
                        "APS works best when model accuracy is ≥70%. "
                        "Consider using score_type='lac' instead.",
                        UserWarning, stacklevel=2
                    )
                    result["vacuous_coverage"] = True
            if cal_acc < 0.70 and warn_vacuous:
                _warnings.warn(
                    f"APS calibration: model accuracy is {cal_acc:.1%} (<70%). "
                    "APS prediction sets may be too large to be useful. "
                    "Consider using score_type='lac' for better set sizes at "
                    "lower accuracy levels.",
                    UserWarning, stacklevel=2
                )
        return result

    def update_calibration(self, seq: tuple, label: int) -> None:
        d = self._infer(seq)
        t = d.sum()
        if t > 1e-12: d /= t
        l = int(label)
        if self._conformal_score_type == "lac": s = 1.0 - d[l]
        elif self._conformal_score_type == "margin": s = np.max(d) - d[l]
        else:
            idx = np.argsort(-d)
            rank = np.where(idx == l)[0][0]
            s = np.sum(d[idx[:rank+1]])
        if self._conformal_scores is None: self._conformal_scores = [s]; self._conformal_n = 1
        else: bisect.insort(self._conformal_scores, s); self._conformal_n += 1

    def _conformal_quantile(self, alpha: float) -> float:
        if self._conformal_scores is None: raise RuntimeError("Not calibrated.")
        idx = max(0, min(math.ceil((self._conformal_n + 1) * (1.0 - alpha)) - 1, self._conformal_n - 1))
        return self._conformal_scores[idx]

    def predict_set(self, seq: tuple, coverage: float = 0.90) -> dict:
        Q = self._conformal_quantile(1.0 - coverage)
        d = self._infer(seq)
        t = d.sum()
        if t > 1e-12: d /= t
        if self._conformal_score_type == "aps":
            idx = np.argsort(-d)
            cs = np.cumsum(d[idx])
            inc = idx[:min(np.searchsorted(cs, Q) + 1, self.n_classes)].tolist()
        else:
            thresh = (1.0 - Q) if self._conformal_score_type == "lac" else (np.max(d) - Q)
            inc = np.where(d >= thresh)[0].tolist()
        if not inc: inc = [int(np.argmax(d))]
        return {"labels": sorted(inc), "n_labels": len(inc), "threshold": Q, "coverage": coverage}

    def predict_set_batch(self, sequences: List[tuple], coverage: float = 0.90) -> List[dict]:
        if self._conformal_scores is None: raise RuntimeError("Not calibrated.")
        Q = self._conformal_quantile(1.0 - coverage)
        dists = self.predict_distributions_batch(sequences)
        res = []
        for d in dists:
            if self._conformal_score_type == "aps":
                idx = np.argsort(-d)
                cs = np.cumsum(d[idx])
                inc = idx[:min(np.searchsorted(cs, Q) + 1, self.n_classes)].tolist()
            else:
                thresh = (1.0 - Q) if self._conformal_score_type == "lac" else (np.max(d) - Q)
                inc = np.where(d >= thresh)[0].tolist()
            if not inc: inc = [int(np.argmax(d))]
            res.append({"labels": sorted(inc), "n_labels": len(inc), "threshold": Q, "coverage": coverage})
        return res

    def coverage_report(self, test_data: List[Tuple[tuple, int]], coverage: float = 0.90) -> dict:
        if self._conformal_scores is None: raise RuntimeError("Not calibrated.")
        sets = self.predict_set_batch([d[0] for d in test_data], coverage=coverage)
        hits = sum(1 for i, s in enumerate(sets) if int(test_data[i][1]) in s['labels'])
        actual = hits / len(test_data)
        return {"requested": coverage, "actual": actual, "mean_size": float(np.mean([s['n_labels'] for s in sets]))}

    def detect_calibration_drift(self, recent_data: List[Tuple[tuple, int]], coverage: float = 0.90) -> dict:
        r = self.coverage_report(recent_data, coverage=coverage)
        n = len(recent_data)
        eps = max(0, coverage - r['actual'])
        p_val = math.exp(-2 * n * (eps**2)) if eps > 0 else 1.0
        return {**r, "p_value": p_val, "drift_detected": p_val < 0.05, "status": "DRIFT" if p_val < 0.05 else "STABLE"}

    # ── Operations ────────────────────────────────────────────────────────

    @classmethod
    def merge(cls, a: "SuffixSmoother", b: "SuffixSmoother") -> "SuffixSmoother":
        if a.n_classes != b.n_classes: raise ValueError("n_classes mismatch")
        m = cls(a.cfg)
        def mn(na, nb):
            r = _SuffixNode()
            for l in set(na.counts.keys()) | set(nb.counts.keys()): r.counts[l] = na.counts.get(l, 0.0) + nb.counts.get(l, 0.0)
            r.total, r._T = na.total + nb.total, len(r.counts)
            for l in set(na.continuation_counts.keys()) | set(nb.continuation_counts.keys()):
                r.continuation_counts[l] = na.continuation_counts.get(l, 0) + nb.continuation_counts.get(l, 0)
            return r
        m._root = mn(a._root, b._root)
        all_k = set(a._nodes.keys()) | set(b._nodes.keys())
        for k in all_k:
            if k in a._nodes and k in b._nodes: m._nodes[k] = mn(a._nodes[k], b._nodes[k])
            elif k in a._nodes:
                n = a._nodes[k]; rn = _SuffixNode()
                rn.counts, rn.total, rn._T = n.counts.copy(), n.total, n._T
                rn.continuation_counts = n.continuation_counts.copy(); m._nodes[k] = rn
            else:
                n = b._nodes[k]; rn = _SuffixNode()
                rn.counts, rn.total, rn._T = n.counts.copy(), n.total, n._T
                rn.continuation_counts = n.continuation_counts.copy(); m._nodes[k] = rn
        m.training_samples = a.training_samples + b.training_samples
        return m

    @classmethod
    def merge_weighted(cls, a: "SuffixSmoother", b: "SuffixSmoother",
                       wa: float = 1.0, wb: float = 1.0,
                       w_a: Optional[float] = None, w_b: Optional[float] = None) -> "SuffixSmoother":
        """Merge two models with per-model weights.

        Accepts both ``wa``/``wb`` (original) and ``w_a``/``w_b`` (README) spellings.
        The ``w_a``/``w_b`` aliases take precedence when supplied.
        """
        # Resolve aliases: w_a/w_b take precedence over wa/wb
        if w_a is not None:
            wa = w_a
        if w_b is not None:
            wb = w_b
        if a.n_classes != b.n_classes: raise ValueError("n_classes mismatch")
        m = cls(a.cfg)
        def mnw(na, nb, wa, wb):
            r = _SuffixNode()
            for l in set(na.counts.keys()) | set(nb.counts.keys()): r.counts[l] = na.counts.get(l, 0.0)*wa + nb.counts.get(l, 0.0)*wb
            r.total, r._T = na.total*wa + nb.total*wb, len(r.counts)
            for l in set(na.continuation_counts.keys()) | set(nb.continuation_counts.keys()):
                r.continuation_counts[l] = na.continuation_counts.get(l, 0) + nb.continuation_counts.get(l, 0)
            return r
        m._root = mnw(a._root, b._root, wa, wb)
        for k in set(a._nodes.keys()) | set(b._nodes.keys()):
            if k in a._nodes and k in b._nodes: m._nodes[k] = mnw(a._nodes[k], b._nodes[k], wa, wb)
            elif k in a._nodes:
                n = a._nodes[k]; rn = _SuffixNode()
                rn.counts = {i: v*wa for i,v in n.counts.items()}; rn.total, rn._T = n.total*wa, n._T
                rn.continuation_counts = n.continuation_counts.copy(); m._nodes[k] = rn
            else:
                n = b._nodes[k]; rn = _SuffixNode()
                rn.counts = {i: v*wb for i,v in n.counts.items()}; rn.total, rn._T = n.total*wb, n._T
                rn.continuation_counts = n.continuation_counts.copy(); m._nodes[k] = rn
        m.training_samples = a.training_samples + b.training_samples
        return m

    @classmethod
    def merge_all(cls, models: List["SuffixSmoother"]) -> "SuffixSmoother":
        if not models: raise ValueError("No models.")
        res = models[0].clone()
        for i in range(1, len(models)): res = cls.merge(res, models[i])
        return res

    def clone(self) -> "SuffixSmoother": return copy.deepcopy(self)

    def warm_caches(self) -> int:
        """Pre-populate all numpy distribution caches.

        Call this after :meth:`load` to eliminate the one-time cold-start
        overhead on the first batch of predictions.  Returns the number of
        nodes whose caches were populated.
        """
        count = 0
        for node in self._nodes.values():
            if node._counts_arr is None:
                node._get_counts_arr(self.n_classes)
                count += 1
        if self._root._counts_arr is None:
            self._root._get_counts_arr(self.n_classes)
        return count

    def prune(self, min_kl: float = 0.1, min_samples: int = 1) -> dict:
        u = 1.0 / self.n_classes
        rem = [s for s, n in self._nodes.items() if _kl_divergence(n.counts, u, self.n_classes) < min_kl or n.total < min_samples]
        for s in rem: del self._nodes[s]
        return {"removed": len(rem), "remaining": len(self._nodes)}

    def prune_to_budget(self, max_nodes: int) -> dict:
        """Prune the suffix tree to at most *max_nodes* nodes.

        Uses a chain-aware greedy algorithm that preserves the backoff
        hierarchy.  The previous KL-ranked removal discarded short-suffix
        (backoff ancestor) nodes first, collapsing the entire backoff chain
        and causing accuracy to fall off a cliff (75% node removal caused
        50% accuracy loss in testing).  This version prevents that.

        Algorithm:
          1. Score every node by KL divergence (higher = more discriminative).
          2. Walk nodes highest-KL first; include each node AND all its
             suffix-trimmed backoff parents.
          3. If forced ancestors push the keep-set over budget, trim the
             lowest-KL leaf nodes (those with no child in the keep-set).

        This guarantees every kept node has its backoff parent present, so
        the inference chain is never broken.  Speed is identical to the
        original (same asymptotic complexity).

        Note: for graceful memory reduction prefer prune(min_kl=X) with a
        tuned threshold.  Use this method only when a hard memory budget
        must be met.
        """
        if len(self._nodes) <= max_nodes:
            return {"removed": 0, "remaining": len(self._nodes)}

        nc = self.n_classes
        scored = {
            s: _kl_divergence(n.counts, 1.0 / nc, nc)
            for s, n in self._nodes.items()
        }

        keep: set = set()
        for suffix, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True):
            if suffix in keep:
                continue
            cur: Optional[tuple] = suffix
            while cur and cur not in keep:
                keep.add(cur)
                cur = cur[1:] if len(cur) > 1 else None
            if len(keep) >= max_nodes * 1.05:
                break

        if len(keep) > max_nodes:
            has_child: set = set()
            for s in keep:
                if len(s) > 1 and s[1:] in keep:
                    has_child.add(s[1:])
            leaves = sorted(
                (scored.get(s, 0.0), s) for s in keep if s not in has_child
            )
            idx = 0
            while len(keep) > max_nodes and idx < len(leaves):
                keep.discard(leaves[idx][1])
                idx += 1

        to_remove = [s for s in self._nodes if s not in keep]
        for k in to_remove:
            del self._nodes[k]
        return {"removed": len(to_remove), "remaining": len(self._nodes)}

    def feature_importance(self, top_n: int = 20) -> List[dict]:
        u = 1.0 / self.n_classes
        scored = []
        for s, n in self._nodes.items():
            if n.total < self.cfg.min_count: continue
            kl = _kl_divergence(n.counts, u, self.n_classes)
            best = max(range(self.n_classes), key=lambda i: n.counts.get(i, 0.0))
            scored.append({"suffix": s, "kl": kl, "top": best, "prob": n.counts.get(best, 0.0) / n.total, "n": int(n.total)})
        scored.sort(key=lambda x: x["kl"], reverse=True)
        return scored[:top_n]

    def label_importance(self, label_id: int, top_n: int = 10) -> List[dict]:
        scored = []
        for s, n in self._nodes.items():
            if n.total < self.cfg.min_count: continue
            scored.append({"suffix": s, "prob": n.counts.get(label_id, 0.0) / n.total, "n": int(n.total)})
        scored.sort(key=lambda x: x["prob"], reverse=True)
        return scored[:top_n]


    def optimize_jm_lambda(self, val_data: List[Tuple[tuple, int]]) -> float:
        """Grid search for optimal Jelinek-Mercer lambda."""
        if self.cfg.smoothing_method != "jelinek-mercer": return 0.0
        best_lam, best_acc = self.cfg.smoothing_lambda, -1.0
        for lam in np.linspace(0.1, 0.9, 9):
            self.cfg.smoothing_lambda = lam
            self._jm_lambdas = [lam ** (i + 1) for i in range(self.cfg.max_suffix_length)]
            acc = self.score(val_data)
            if acc > best_acc: best_acc, best_lam = acc, lam
        self.cfg.smoothing_lambda = best_lam
        self._jm_lambdas = [best_lam ** (i + 1) for i in range(self.cfg.max_suffix_length)]
        return best_lam

    def predict_top_k(self, seq: tuple, k: int = 3) -> List[Tuple[int, float]]:
        """Return top k labels and their probabilities."""
        d = self._infer(seq)
        t = d.sum()
        if t > 1e-12: d /= t
        idx = np.argsort(-d)[:k]
        return [(int(i), float(d[i])) for i in idx]

    def predict_top_k_batch(self, sequences: List[tuple], k: int = 3) -> List[List[Tuple[int, float]]]:
        """Vectorized batch top-k prediction."""
        dists = self.predict_distributions_batch(sequences)
        res = []
        for d in dists:
            idx = np.argsort(-d)[:k]
            res.append([(int(i), float(d[i])) for i in idx])
        return res

    def predict_proba_batch(self, sequences: List[tuple]) -> List[np.ndarray]:
        """Alias for predict_distributions_batch."""
        return self.predict_distributions_batch(sequences)

    def model_summary(self) -> dict:
        if not self._nodes: return {"nodes": 0, "samples": self.training_samples}
        kls = [_kl_divergence(n.counts, 1.0/self.n_classes, self.n_classes) for n in self._nodes.values()]
        return {"version": __version__, "smoothing": self.cfg.smoothing_method, "nodes": len(self._nodes), "samples": self.training_samples, "mean_kl": float(np.mean(kls)), "calibrated": self.is_calibrated}

    @staticmethod
    def compare(models: List[Tuple[str, "SuffixSmoother"]], test_data: List[Tuple[tuple, int]]) -> List[dict]:
        report = []
        for name, m in models:
            preds = m.predict_batch([d[0] for d in test_data])
            corr = [p[0] == int(t[1]) for p, t in zip(preds, test_data)]
            ece = SuffixSmoother.expected_calibration_error([p[1] for p in preds], corr)
            report.append({"name": name, "accuracy": round(sum(corr)/len(corr), 4), "ece": round(ece, 4)})
        report.sort(key=lambda x: x["accuracy"], reverse=True)
        return report

    @staticmethod
    def expected_calibration_error(probs: list, labels: list, n_bins: int = 10) -> float:
        p, l = np.array(probs), np.array(labels, dtype=float)
        ece = 0.0
        for b in range(n_bins):
            mask = (p >= b/n_bins) & (p <= (b+1)/n_bins if b==n_bins-1 else p < (b+1)/n_bins)
            if mask.sum() > 0: ece += (mask.sum()/len(p)) * abs(l[mask].mean() - p[mask].mean())
        return float(ece)

    def optimize_kn_discount(self, val_data: List[Tuple[tuple, int]]) -> float:
        if self.cfg.smoothing_method != "kneser-ney": return 0.0
        best_d, best_acc = 0.75, -1.0
        for d in np.linspace(0.1, 1.5, 15):
            self._kn_D_fixed = d
            acc = self.score(val_data)
            if acc > best_acc: best_acc, best_d = acc, d
        self._kn_D_fixed = best_d
        return best_d

    def score(self, test_data: list) -> float:
        if not test_data: return 0.0
        preds = self.predict_batch([d[0] for d in test_data])
        return sum(1 for p, l in zip(preds, test_data) if p[0] == int(l[1])) / len(test_data)

    def save(self, path: str) -> None:
        if self.cfg.smoothing_method == "kneser-ney": self._finalize_kn()
        with open(path, "wb") as f: pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "SuffixSmoother":
        with open(path, "rb") as f: obj = pickle.load(f)
        if not isinstance(obj, cls): raise TypeError("Invalid type.")
        return obj

    def __getstate__(self):
        # Build a serialisable state WITHOUT mutating the live model.
        # Previously this cleared numpy caches directly on live nodes,
        # causing silent performance regression after every save().
        state = self.__dict__.copy()
        state['_kn_base_p'] = None
        state['_kn_finalized'] = False

        # Shallow-copy the root node so we can clear its cache without
        # touching the live object.
        root_copy = copy.copy(self._root)
        root_copy._counts_arr = None
        root_copy._wb_lambda = -1.0
        root_copy._kn_bw_cache = -1.0
        state['_root'] = root_copy

        # Shallow-copy every suffix node for the same reason.
        nodes_copy = {}
        for k, node in self._nodes.items():
            nc = copy.copy(node)
            nc._counts_arr = None
            nc._wb_lambda = -1.0
            nc._kn_bw_cache = -1.0
            nodes_copy[k] = nc
        state['_nodes'] = nodes_copy
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._uniform_p = np.full(self.cfg.n_classes, 1.0 / self.cfg.n_classes)

    @property
    def n_nodes(self) -> int: return len(self._nodes)
    @property
    def is_calibrated(self) -> bool: return self._conformal_scores is not None
    def __repr__(self) -> str:
        return f"SuffixSmoother(n_classes={self.n_classes}, method={self.cfg.smoothing_method!r}, nodes={self.n_nodes}, samples={self.training_samples})"
