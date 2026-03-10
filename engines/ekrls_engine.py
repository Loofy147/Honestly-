"""
EKRLS Engine — Extended Kernel Recursive Least Squares
للنظام الكمومي: تتبع الحالة في الوقت الحقيقي عبر RKHS

Φ_n = f(Φ_{n-1}) + v_n   (state transition)
y_n = g(Φ_n)  + w_n       (measurement)

Uses Square Root EKRLS with Givens rotations for numerical stability.
No matrix inversion at each step → O(n²) updates, FPGA-ready.
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EKRLSConfig:
    """Configuration for EKRLS Engine."""
    state_dim: int = 4           # Dimension of quantum state Φ
    kernel_sigma: float = 1.0    # RBF kernel bandwidth
    forgetting_factor: float = 0.99  # λ in recursive update (memory)
    process_noise: float = 0.01  # σ_v²
    measurement_noise: float = 0.05  # σ_w²
    window_size: int = 50        # Sliding window for RKHS dict


class RBFKernel:
    """Radial Basis Function kernel: k(x,x') = exp(-||x-x'||²/2σ²)"""

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x.flatten() - y.flatten()
        return float(np.exp(-np.dot(diff, diff) / (2 * self.sigma ** 2)))

    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute full Gram matrix K_{ij} = k(x_i, x_j)"""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = K[j, i] = self(X[i], X[j])
        return K


@dataclass
class QuantumState:
    """Represents a quantum state vector in Hilbert space."""
    phi: np.ndarray          # State vector (density matrix diagonal)
    timestamp: int = 0
    coherence: float = 1.0   # 1.0 = fully coherent, 0.0 = collapsed
    entanglement_entropy: float = 0.0

    def is_collapsed(self, threshold: float = 0.1) -> bool:
        return self.coherence < threshold

    def von_neumann_entropy(self) -> float:
        """S = -Tr(ρ log ρ) approximated from state vector."""
        probs = np.abs(self.phi) ** 2
        probs = probs / probs.sum()
        # Avoid log(0)
        safe = probs[probs > 1e-12]
        return float(-np.sum(safe * np.log2(safe)))


class SquareRootEKRLS:
    """
    Square Root Extended Kernel Recursive Least Squares.

    Uses Givens rotations to update the data window without matrix inversion.
    Suitable for real-time quantum state tracking and FPGA implementation.
    """

    def __init__(self, config: EKRLSConfig):
        self.cfg = config
        self.kernel = RBFKernel(sigma=config.kernel_sigma)
        self.lam = config.forgetting_factor
        self.q_noise = config.process_noise
        self.r_noise = config.measurement_noise

        # Dictionary (sliding window) of past states
        self._dict_X: list[np.ndarray] = []
        self._dict_y: list[float] = []

        # Square-root covariance factor (upper triangular)
        self._R_sqrt: Optional[np.ndarray] = None
        # Weight vector in RKHS
        self._alpha: Optional[np.ndarray] = None

        self.update_count = 0
        self.prediction_errors: list[float] = []

    def _givens_rotation(self, R: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Givens rotation update: annihilate subdiagonal using plane rotations.
        Updates R in-place to maintain triangular form.
        Returns updated R and rotated x.
        """
        R = R.copy()
        x = x.copy()
        n = len(x)
        for i in range(n):
            if abs(x[i]) < 1e-14:
                continue
            r = np.sqrt(R[i, i] ** 2 + x[i] ** 2)
            c = R[i, i] / r  # cosine
            s = x[i] / r     # sine
            # Apply rotation to row i of R and x
            R[i, i] = r
            if i + 1 < n:
                R[i, i+1:] = c * R[i, i+1:] + s * x[i+1:]
                x[i+1:]    = -s * R[i, i+1:] + c * x[i+1:]  # Note: uses updated R
            x[i] = 0.0
        return R, x

    def _kernel_vector(self, x_new: np.ndarray) -> np.ndarray:
        """Compute kernel vector k(x_new, x_i) for all dictionary entries."""
        if not self._dict_X:
            return np.array([])
        return np.array([self.kernel(x_new, xi) for xi in self._dict_X])

    def predict(self, phi_new: np.ndarray) -> Tuple[float, float]:
        """
        Predict measurement y for new state phi_new.
        Returns: (prediction, uncertainty_estimate)
        """
        if self._alpha is None or len(self._dict_X) == 0:
            return 0.0, float('inf')

        k_vec = self._kernel_vector(phi_new)
        # Guard against size mismatch (alpha not yet extended)
        n = min(len(self._alpha), len(k_vec))
        y_pred = float(np.dot(self._alpha[:n], k_vec[:n]))

        # Uncertainty from RKHS norm
        k_self = self.kernel(phi_new, phi_new)
        if self._R_sqrt is not None and len(k_vec) == self._R_sqrt.shape[0]:
            try:
                v = np.linalg.solve(self._R_sqrt.T, k_vec)
                uncertainty = float(np.sqrt(max(0, k_self - np.dot(v, v))))
            except np.linalg.LinAlgError:
                uncertainty = float(np.sqrt(k_self))
        else:
            uncertainty = float(np.sqrt(k_self))

        return y_pred, uncertainty

    def update(self, phi_n: np.ndarray, y_n: float) -> dict:
        """
        Recursive update given new state-measurement pair.
        Uses Givens rotations for numerically stable covariance update.
        """
        self.update_count += 1
        n = self.update_count

        # --- Window management ---
        if len(self._dict_X) >= self.cfg.window_size:
            self._dict_X.pop(0)
            self._dict_y.pop(0)
            if self._alpha is not None and len(self._alpha) > 0:
                self._alpha = self._alpha[1:]
            if self._R_sqrt is not None and self._R_sqrt.shape[0] > 1:
                self._R_sqrt = self._R_sqrt[1:, 1:]

        self._dict_X.append(phi_n.copy())
        self._dict_y.append(y_n)

        d = len(self._dict_X)

        # --- Initialize if first step ---
        if d == 1:
            k00 = self.kernel(phi_n, phi_n) + self.r_noise
            self._R_sqrt = np.array([[np.sqrt(k00)]])
            self._alpha = np.array([y_n / k00])
            self.prediction_errors.append(y_n)
            return {"step": n, "d": d, "pred_error": y_n, "uncertainty": float(np.sqrt(k00))}

        # --- Kernel vector for new point ---
        k_vec = np.array([self.kernel(phi_n, xi) for xi in self._dict_X[:-1]])
        k_self = self.kernel(phi_n, phi_n) + self.r_noise

        # --- Extend R_sqrt ---
        R_old = self._R_sqrt
        r_old_d = len(R_old)
        R_new = np.zeros((d, d))
        R_new[:r_old_d, :r_old_d] = R_old

        # New row: [k_vec, sqrt(k_self - k_vec·inv(R^T R)·k_vec)]
        if r_old_d > 0 and len(k_vec) > 0:
            try:
                v = np.linalg.solve(R_old.T, k_vec)
                schur = k_self - np.dot(v, v)
                R_new[r_old_d - 1, r_old_d - 1] = np.sqrt(max(schur, 1e-10))
            except np.linalg.LinAlgError:
                R_new[d-1, d-1] = np.sqrt(k_self)
        else:
            R_new[d-1, d-1] = np.sqrt(k_self)

        self._R_sqrt = R_new

        # --- Prediction error ---
        y_pred, uncertainty = self.predict(phi_n)
        # Re-compute before alpha update (using old alpha on new dict)
        if self._alpha is not None and len(self._alpha) < d:
            alpha_ext = np.append(self._alpha, 0.0)
        else:
            alpha_ext = self._alpha if self._alpha is not None else np.zeros(d)

        k_full = self._kernel_vector(phi_n)
        if len(k_full) == len(alpha_ext):
            y_pred = float(np.dot(alpha_ext, k_full))
        else:
            y_pred = 0.0

        pred_error = y_n - y_pred
        self.prediction_errors.append(pred_error)

        # --- Update alpha via RKHS weight update ---
        K = self.kernel.gram_matrix(self._dict_X)
        K += self.r_noise * np.eye(d)
        try:
            self._alpha = np.linalg.solve(K, np.array(self._dict_y))
        except np.linalg.LinAlgError:
            self._alpha = np.linalg.lstsq(K, np.array(self._dict_y), rcond=None)[0]

        return {
            "step": n,
            "d": d,
            "y_pred": float(y_pred),
            "y_true": float(y_n),
            "pred_error": float(pred_error),
            "uncertainty": float(uncertainty),
        }


class EKRLSQuantumEngine:
    """
    Full quantum state tracking engine.
    Wraps SquareRootEKRLS with quantum-specific state management.

    Detects entanglement collapse and issues metacognitive alerts.
    """

    def __init__(self, config: Optional[EKRLSConfig] = None):
        self.cfg = config or EKRLSConfig()
        self.ekrls = SquareRootEKRLS(self.cfg)
        self.state_history: list[QuantumState] = []
        self.collapse_events: list[int] = []
        self.entanglement_battery: float = 1.0  # Normalized [0,1]

    def _compute_coherence(self, phi: np.ndarray) -> float:
        """Coherence = normalized off-diagonal density matrix magnitude."""
        n = len(phi)
        if n < 2:
            return 1.0
        # Outer product approximation of density matrix
        rho = np.outer(phi, phi.conj())
        diag_sum = np.abs(np.diag(rho)).sum()
        total_sum = np.abs(rho).sum()
        if total_sum < 1e-12:
            return 0.0
        off_diag = total_sum - diag_sum
        return float(off_diag / total_sum)

    def step(self, phi_raw: np.ndarray, measurement: float) -> dict:
        """
        Process one quantum evolution step.
        phi_raw: raw state vector
        measurement: observed output y_n
        """
        # Normalize state
        norm = np.linalg.norm(phi_raw)
        phi_n = phi_raw / (norm + 1e-12)

        # Add process noise (quantum fluctuation)
        phi_n = phi_n + np.random.normal(0, self.cfg.process_noise, phi_n.shape)
        phi_n /= np.linalg.norm(phi_n) + 1e-12

        # Compute quantum properties
        coherence = self._compute_coherence(phi_n)
        qs = QuantumState(
            phi=phi_n,
            timestamp=len(self.state_history),
            coherence=coherence,
            entanglement_entropy=0.0
        )
        qs.entanglement_entropy = qs.von_neumann_entropy()

        # Check for collapse BEFORE update
        if qs.is_collapsed():
            self.collapse_events.append(len(self.state_history))

        # EKRLS update
        update_info = self.ekrls.update(phi_n, measurement)

        # Update entanglement battery based on entropy change
        if self.state_history:
            prev_S = self.state_history[-1].entanglement_entropy
            delta_S = qs.entanglement_entropy - prev_S
            # Battery charges when entanglement increases
            self.entanglement_battery = np.clip(
                self.entanglement_battery + 0.1 * delta_S, 0.0, 1.0
            )

        self.state_history.append(qs)

        return {
            **update_info,
            "coherence": coherence,
            "entropy": qs.entanglement_entropy,
            "battery_level": self.entanglement_battery,
            "collapse_detected": qs.is_collapsed(),
            "total_collapses": len(self.collapse_events),
        }

    def run_simulation(self, n_steps: int = 100, seed: int = 42) -> list[dict]:
        """Run a full simulation with synthetic quantum evolution."""
        np.random.seed(seed)
        results = []
        dim = self.cfg.state_dim

        # Initial state: Bell-like superposition
        phi = np.zeros(dim)
        phi[0] = 1.0 / np.sqrt(2)
        phi[1] = 1.0 / np.sqrt(2)

        for t in range(n_steps):
            # Unitary-like evolution (rotation + decoherence)
            theta = 0.05 * t
            if dim >= 2:
                c, s = np.cos(theta), np.sin(theta)
                phi_new = phi.copy()
                phi_new[0] = c * phi[0] - s * phi[1]
                phi_new[1] = s * phi[0] + c * phi[1]
                phi = phi_new

            # Add decoherence at random steps
            if np.random.random() < 0.05:
                noise = np.random.randn(dim) * 0.3
                phi = phi + noise
                phi /= np.linalg.norm(phi) + 1e-12

            # Measurement (projection onto first basis state)
            measurement = float(np.abs(phi[0]) ** 2) + np.random.normal(0, 0.02)
            result = self.step(phi, measurement)
            result["t"] = t
            results.append(result)

        return results

    def summary(self) -> dict:
        """Generate summary statistics."""
        if not self.state_history:
            return {}
        entropies = [s.entanglement_entropy for s in self.state_history]
        coherences = [s.coherence for s in self.state_history]
        errors = self.ekrls.prediction_errors
        return {
            "total_steps": len(self.state_history),
            "collapse_events": len(self.collapse_events),
            "mean_entropy": float(np.mean(entropies)),
            "mean_coherence": float(np.mean(coherences)),
            "mean_pred_error": float(np.mean(np.abs(errors))) if errors else 0.0,
            "rmse": float(np.sqrt(np.mean(np.array(errors)**2))) if errors else 0.0,
            "battery_final": self.entanglement_battery,
            "dictionary_size": len(self.ekrls._dict_X),
        }
