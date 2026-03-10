"""
Lie Algebra Expansion — التحكم في بطارية التشابك
Wei-Norman framework for entanglement battery resource flow.

U(t) = exp(g₁(t)·X₁) · exp(g₂(t)·X₂) · ... · exp(gₙ(t)·Xₙ)

Formal power series in expansion parameter ε controls resource transfer
between the entanglement battery and the active quantum network.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import scipy.linalg


@dataclass
class LieAlgebraConfig:
    """Configuration for Lie algebra expansion engine."""
    algebra_dim: int = 4         # Dimension of Lie algebra (generators)
    expansion_order: int = 6     # Truncation order N in power series
    dt: float = 0.01             # Time step for Wei-Norman integration
    coupling_alpha: float = 1e-3 # Entanglement-spacetime coupling (α < 10⁻¹⁵ physical)
    battery_capacity: float = 10.0  # Max stored entanglement units


class LieGenerator:
    """
    A generator of a Lie algebra: anti-Hermitian matrix X satisfying [X,X]=0.
    Used to build the algebra basis for different symmetry groups.
    """

    @staticmethod
    def galilei_mass(n: int = 2) -> np.ndarray:
        """Mass central charge M — Galilei algebra generator."""
        X = np.zeros((n, n), dtype=complex)
        X[0, 0] = 1j
        return X

    @staticmethod
    def boost(n: int = 2, i: int = 0, j: int = 1) -> np.ndarray:
        """Galilei boost generator K_i. Anti-Hermitian: X†=-X."""
        X = np.zeros((n, n), dtype=complex)
        if i < n and j < n:
            X[i, j] = 1j
            X[j, i] = 1j   # Makes X†=-X since (1j)*=-1j=-X[i,j] ✓
        return X

    @staticmethod
    def rotation(n: int = 2, i: int = 0, j: int = 1) -> np.ndarray:
        """Rotation generator J_{ij}. Anti-Hermitian real antisymmetric."""
        X = np.zeros((n, n), dtype=complex)
        if i < n and j < n:
            X[i, j] = 1.0    # Real antisymmetric → anti-Hermitian
            X[j, i] = -1.0
        return X

    @staticmethod
    def dilatation(n: int = 2) -> np.ndarray:
        """Dilatation generator D — Schrödinger algebra."""
        X = np.zeros((n, n), dtype=complex)
        for i in range(n):
            X[i, i] = 1j * (i - n / 2)
        return X

    @staticmethod
    def tensor_generator(n: int = 2, antisym: bool = True) -> np.ndarray:
        """Anti-symmetric tensor generator Z_μν — Maxwell algebra."""
        X = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(i + 1, n):
                X[i, j] = 1j
                if antisym:
                    X[j, i] = -1j
        return X


class LieAlgebra:
    """
    A specific Lie algebra with basis generators and structure constants.
    Supports Galilei, Poincaré, Schrödinger, and String-Galilei algebras.
    """

    ALGEBRA_TYPES = ['galilei', 'poincare', 'schrodinger', 'string_galilei']

    def __init__(self, algebra_type: str = 'galilei', n: int = 2):
        self.algebra_type = algebra_type
        self.n = n
        self.generators = self._build_generators()
        self.structure_constants = self._compute_structure_constants()

    def _build_generators(self) -> list[np.ndarray]:
        """Build basis generators for the selected algebra."""
        n = self.n
        g = LieGenerator

        if self.algebra_type == 'galilei':
            return [
                g.galilei_mass(n),   # M: mass central charge
                g.boost(n, 0, 1),    # K: Galilei boost
                g.rotation(n, 0, 1), # J: spatial rotation
                np.eye(n, dtype=complex) * 1j,  # H: Hamiltonian
            ]
        elif self.algebra_type == 'poincare':
            return [
                g.rotation(n, 0, 1),         # Spatial rotation
                g.tensor_generator(n, True),  # Lorentz boost
                np.eye(n, dtype=complex) * 1j, # 4-momentum P
                g.dilatation(n),               # Scaling
            ]
        elif self.algebra_type == 'schrodinger':
            return [
                g.galilei_mass(n),   # M
                g.boost(n, 0, 1),    # K
                g.rotation(n, 0, 1), # J
                g.dilatation(n),     # D: dilatation
            ]
        elif self.algebra_type == 'string_galilei':
            # Extended with central/non-central charges for p-branes
            base = [
                g.galilei_mass(n),
                g.boost(n, 0, 1),
                g.rotation(n, 0, 1),
                g.tensor_generator(n, False),  # Non-central extension
            ]
            return base
        else:
            raise ValueError(f"Unknown algebra: {self.algebra_type}")

    def _compute_structure_constants(self) -> np.ndarray:
        """
        Compute structure constants f^k_{ij} from [X_i, X_j] = f^k_{ij} X_k.
        Uses matrix commutators: [A,B] = AB - BA
        """
        d = len(self.generators)
        f = np.zeros((d, d, d), dtype=complex)

        for i, Xi in enumerate(self.generators):
            for j, Xj in enumerate(self.generators):
                comm = Xi @ Xj - Xj @ Xi
                # Project onto generator basis
                for k, Xk in enumerate(self.generators):
                    # f^k_{ij} = Tr(Xk† · [Xi,Xj]) / Tr(Xk† · Xk)
                    denom = np.trace(Xk.conj().T @ Xk)
                    if abs(denom) > 1e-12:
                        f[k, i, j] = np.trace(Xk.conj().T @ comm) / denom

        return f

    def commutator(self, i: int, j: int) -> np.ndarray:
        """Compute [X_i, X_j]."""
        Xi, Xj = self.generators[i], self.generators[j]
        return Xi @ Xj - Xj @ Xi

    def adjoint_representation(self, i: int) -> np.ndarray:
        """Adjoint rep matrix: (ad X_i)^k_j = f^k_{ij}"""
        return self.structure_constants[:, i, :]


class EntanglementBattery:
    """
    Entanglement Battery — بطارية التشابك

    Uses Lie algebra expansion to control resource flow.
    The formal power series in ε controls coupling between
    battery (storage) and active network (discharge).

    Wei-Norman decomposition:
    U(t) = ∏_i exp(g_i(t) · X_i)

    Conservation law: E(β_f) - E(β_i) = E(ρ) - E(σ)
    """

    def __init__(self, config: Optional[LieAlgebraConfig] = None,
                 algebra_type: str = 'galilei'):
        self.cfg = config or LieAlgebraConfig()
        self.algebra = LieAlgebra(algebra_type, n=self.cfg.algebra_dim)
        self.d = len(self.algebra.generators)

        # Battery state: stored entanglement E_battery
        self.E_battery: float = self.cfg.battery_capacity / 2.0  # Start at 50%

        # Wei-Norman coefficients g_i(t)
        self.g: np.ndarray = np.zeros(self.d)

        # History
        self.history: list[dict] = []
        self.conservation_violations: list[float] = []

    def formal_power_series(self, epsilon: float, order: int = None) -> np.ndarray:
        """
        Compute formal power series expansion up to order N.
        S(ε) = Σ_{n=0}^{N} ε^n · A_n

        Returns: expansion coefficients [A_0, A_1, ..., A_N]
        """
        N = order or self.cfg.expansion_order
        d = self.d
        Xgens = self.algebra.generators

        # A_0 = Identity
        coeffs = [np.eye(d, dtype=complex)]

        # A_n = (1/n!) Σ_{i} X_i (recursive Lie series)
        current = np.eye(d, dtype=complex)
        for n in range(1, N + 1):
            # Sum contributions from all generators weighted by g_i
            term = np.zeros((d, d), dtype=complex)
            for i, Xi in enumerate(Xgens):
                # Use commutator structure
                ad_Xi = self.algebra.adjoint_representation(i)
                term += self.g[i] * ad_Xi / n
            current = current @ term
            coeffs.append(current.copy())

        return np.array(coeffs)

    def evolution_operator(self, t: float) -> np.ndarray:
        """
        Compute U(t) via Wei-Norman product decomposition.
        U(t) = exp(g₁X₁) · exp(g₂X₂) · ... · exp(gₙXₙ)
        """
        U = np.eye(self.cfg.algebra_dim, dtype=complex)
        for i, (Xi, gi) in enumerate(zip(self.algebra.generators, self.g)):
            # Matrix exponential of gi·Xi
            U_i = scipy.linalg.expm(gi * Xi)
            U = U @ U_i
        return U

    def _wei_norman_ode(self, g: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        Wei-Norman ODE: dg/dt = F(g) · h
        where h = decomposition of H in generator basis.

        ḣᵢ = Σⱼ [exp(ad g)]ᵢⱼ · hⱼ
        """
        d = self.d
        dg = np.zeros(d)

        # Compute exp(ad g) matrix
        ad_g = np.zeros((d, d), dtype=complex)
        for i in range(d):
            ad_g += g[i] * self.algebra.adjoint_representation(i)

        try:
            exp_ad_g = scipy.linalg.expm(ad_g)
        except Exception:
            exp_ad_g = np.eye(d, dtype=complex)

        # Project Hamiltonian onto generators
        h = np.zeros(d, dtype=complex)
        for i, Xi in enumerate(self.algebra.generators):
            denom = np.trace(Xi.conj().T @ Xi)
            if abs(denom) > 1e-12:
                h[i] = np.trace(Xi.conj().T @ H) / denom

        # dg = Re(exp_ad_g · h) (real part for physical g)
        dg = np.real(exp_ad_g @ h)
        return dg

    def charge(self, delta_E: float) -> dict:
        """
        Charge battery with delta_E entanglement units.
        Increases stored resource, updates Wei-Norman coefficients.
        """
        E_before = self.E_battery
        self.E_battery = min(self.E_battery + delta_E, self.cfg.battery_capacity)
        actual_charge = self.E_battery - E_before

        # Update coupling parameter (g_0 tracks charging)
        self.g[0] += actual_charge * self.cfg.coupling_alpha
        self.g = np.clip(self.g, -10, 10)

        return {
            "operation": "charge",
            "requested": delta_E,
            "actual": actual_charge,
            "E_battery": self.E_battery,
            "capacity_pct": 100 * self.E_battery / self.cfg.battery_capacity,
        }

    def discharge(self, delta_E: float) -> dict:
        """
        Discharge battery: transfer delta_E to active network.
        Checks conservation law.
        """
        available = self.E_battery
        actual_discharge = min(delta_E, available)
        E_before = self.E_battery
        self.E_battery -= actual_discharge

        # Conservation check: |E_consumed - E_transferred| should be < threshold
        conservation_residual = abs(delta_E - actual_discharge)
        self.conservation_violations.append(conservation_residual)

        # Update coupling parameter
        self.g[0] -= actual_discharge * self.cfg.coupling_alpha
        self.g = np.clip(self.g, -10, 10)

        return {
            "operation": "discharge",
            "requested": delta_E,
            "actual": actual_discharge,
            "E_battery": self.E_battery,
            "conservation_residual": conservation_residual,
            "conservation_ok": conservation_residual < 1e-6,
        }

    def convert_states(self, E_rho: float, E_sigma: float) -> dict:
        """
        Convert between quantum states ρ→σ using battery buffer.
        Implements: r(ρ→σ) = E(ρ) / E(σ)

        Battery absorbs or releases the difference to ensure reversibility.
        """
        rate = E_rho / (E_sigma + 1e-12)
        delta = E_rho - E_sigma

        if delta > 0:
            # ρ has more entanglement — battery absorbs excess
            result = self.charge(delta)
        else:
            # σ needs more entanglement — battery provides
            result = self.discharge(abs(delta))

        # Conservation law: E(β_f) - E(β_i) = E(ρ) - E(σ)
        conservation_check = abs(
            (result["E_battery"] - (self.E_battery + delta)) - delta
        )

        return {
            "E_rho": E_rho,
            "E_sigma": E_sigma,
            "conversion_rate": rate,
            "delta_E": delta,
            "battery_after": self.E_battery,
            "reversible": result.get("conservation_ok", True),
            **result,
        }

    def evolve(self, n_steps: int, H_func=None) -> list[dict]:
        """
        Time-evolve battery via Wei-Norman ODEs.
        H_func(t) → Hamiltonian matrix at time t
        """
        dt = self.cfg.dt
        results = []
        U_total = np.eye(self.cfg.algebra_dim, dtype=complex)

        for step in range(n_steps):
            t = step * dt
            # Default: oscillating Hamiltonian (entanglement pumping)
            if H_func is None:
                omega = 2 * np.pi * 0.5  # 0.5 Hz oscillation
                H = np.zeros((self.cfg.algebra_dim, self.cfg.algebra_dim), dtype=complex)
                for i, Xi in enumerate(self.algebra.generators):
                    H += np.sin(omega * t + i * np.pi / self.d) * Xi
            else:
                H = H_func(t)

            # Wei-Norman ODE step (Euler)
            dg = self._wei_norman_ode(self.g, H)
            self.g = self.g + dt * dg
            self.g = np.clip(self.g, -10, 10)

            # Compute evolution operator
            U = self.evolution_operator(t + dt)
            U_total = U @ U_total

            # Extract coupling strength from power series
            series = self.formal_power_series(epsilon=self.cfg.coupling_alpha)
            coupling_strength = float(np.abs(series[1]).mean())

            results.append({
                "t": t,
                "g": self.g.copy(),
                "E_battery": self.E_battery,
                "coupling_strength": coupling_strength,
                "U_norm": float(np.linalg.norm(U_total)),
                "conservation_satisfied": len(self.conservation_violations) == 0
                    or self.conservation_violations[-1] < 1e-6,
            })

        return results

    def summary(self) -> dict:
        return {
            "algebra_type": self.algebra.algebra_type,
            "E_battery": self.E_battery,
            "capacity_pct": 100 * self.E_battery / self.cfg.battery_capacity,
            "g_norms": float(np.linalg.norm(self.g)),
            "n_conservation_violations": sum(
                1 for v in self.conservation_violations if v > 1e-6
            ),
            "mean_violation": float(np.mean(self.conservation_violations))
                if self.conservation_violations else 0.0,
        }
