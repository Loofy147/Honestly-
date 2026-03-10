"""
Finance Domain Adapter
========================
تطبيق المحركات الكمومية على أسواق المال

Quantum → Finance Isomorphism:
  Φ_n  (quantum state)    →  [price, volume, bid-ask, momentum]  (market state)
  coherence               →  market stability (low vol)
  entanglement collapse   →  volatility spike / regime break
  EKRLS in RKHS           →  non-linear volatility surface estimation
  Viterbi over QEC codes  →  hidden market regime sequence (bull/bear/range/crash)
  Entanglement battery     →  liquidity reserve (absorbs/releases volatility)
  Ribbon filter           →  fast lookup of historical pattern signatures

Key advantage over classical models:
  - GARCH is LINEAR in variance → misses regime-dependent non-linearity
  - EKRLS maps into RKHS → captures volatility clustering, skew, fat tails
  - Battery model: liquidity acts as buffer; its depletion = flash crash precursor
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engines.ekrls_engine import EKRLSQuantumEngine, EKRLSConfig
from error_correction.suffix_smoothing import QuantumErrorCorrector, SuffixConfig
from algebra.lie_expansion import EntanglementBattery, LieAlgebraConfig
from metacognition.metacognitive_layer import MetacognitiveLayer, MetacognitiveConfig


REGIME_LABELS = {
    0: "BULL_QUIET",   1: "BULL_VOLATILE",
    2: "RANGE_LOW",    3: "RANGE_HIGH",
    4: "BEAR_SLOW",    5: "BEAR_PANIC",
    6: "CRASH",        7: "RECOVERY",
}


def generate_market_data(n: int = 200, seed: int = 42) -> dict:
    """
    Generate synthetic OHLCV market data with regime changes.
    Returns dict of price, volume, returns, realized_vol arrays.
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros(n)
    prices[0] = 100.0
    regimes = np.zeros(n, dtype=int)

    # Regime schedule: bull→range→bear→recovery
    for i in range(1, n):
        if i < 50:   # Bull
            mu, sigma, regime = 0.001, 0.012, 0
        elif i < 90:  # Range
            mu, sigma, regime = 0.0, 0.008, 2
        elif i < 130: # Bear
            mu, sigma, regime = -0.002, 0.018, 4
        elif i < 150: # Crash
            mu, sigma, regime = -0.015, 0.045, 6
        else:         # Recovery
            mu, sigma, regime = 0.003, 0.020, 7

        ret = rng.normal(mu, sigma)
        prices[i] = prices[i-1] * (1 + ret)
        regimes[i] = regime

    returns = np.diff(np.log(prices))
    vol = np.array([
        float(np.std(returns[max(0,i-20):i+1])) if i > 0 else 0.01
        for i in range(len(returns))
    ])
    volume = rng.lognormal(mean=14.0, sigma=0.3, size=n)

    return {
        "prices": prices,
        "returns": returns,
        "realized_vol": vol,
        "volume": volume,
        "true_regimes": regimes,
        "n": n,
    }


def encode_market_state(prices: np.ndarray, returns: np.ndarray,
                         vol: np.ndarray, volume: np.ndarray,
                         i: int, lookback: int = 5) -> np.ndarray:
    """
    Encode market observation into 4D quantum-style state vector Φ.

    Φ = [normalized_price_momentum, vol_z_score, volume_z_score, trend_strength]
    """
    if i < lookback:
        return np.zeros(4)

    # Price momentum (normalized)
    momentum = float(np.sum(returns[max(0,i-lookback):i])) / (lookback * 0.01 + 1e-8)
    momentum = np.clip(momentum, -5, 5) / 5.0

    # Volatility z-score
    vol_mean = float(np.mean(vol[max(0,i-lookback):i]))
    vol_std = float(np.std(vol[max(0,i-lookback):i])) + 1e-8
    vol_z = np.clip((vol[i-1] - vol_mean) / vol_std, -3, 3) / 3.0

    # Volume z-score
    vol_m = float(np.mean(volume[max(0,i-lookback):i]))
    vol_s = float(np.std(volume[max(0,i-lookback):i])) + 1e-8
    vol_z2 = np.clip((volume[i] - vol_m) / vol_s, -3, 3) / 3.0

    # Trend strength (R² of price over window)
    window_prices = prices[max(0,i-lookback):i]
    if len(window_prices) > 2:
        x = np.arange(len(window_prices), dtype=float)
        x -= x.mean()
        y = window_prices - window_prices.mean()
        r2 = float((np.dot(x,y)**2) / (np.dot(x,x)*np.dot(y,y) + 1e-12))
    else:
        r2 = 0.0

    return np.array([momentum, vol_z, vol_z2, r2])


class FinancialQuantumAnalyzer:
    """
    Quantum-inspired financial analysis engine.

    Uses EKRLS for non-linear volatility estimation and
    Viterbi + Suffix Smoother for market regime detection.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ekrls = EKRLSQuantumEngine(EKRLSConfig(
            state_dim=4,
            kernel_sigma=0.5,    # Tighter kernel: captures local vol clustering
            forgetting_factor=0.97,
            process_noise=0.005,
            measurement_noise=0.02,
            window_size=30,
        ))
        self.regime_detector = QuantumErrorCorrector(SuffixConfig(
            max_suffix_length=5,
            n_qec_codes=8,       # 8 market regimes
        ))
        self.liquidity_battery = EntanglementBattery(
            LieAlgebraConfig(battery_capacity=1.0, coupling_alpha=0.01),
            algebra_type='galilei',
        )
        self.metacog = MetacognitiveLayer(MetacognitiveConfig(
            coherence_collapse_threshold=0.15,
        ))
        self.results: list[dict] = []

    def _vol_from_coherence(self, coherence: float, base_vol: float) -> float:
        """
        Isomorphism: coherence ↔ market stability
        Low coherence → high volatility
        coherence=1.0 → vol = base_vol
        coherence=0.0 → vol = 5x base_vol
        """
        vol_multiplier = 1.0 + 4.0 * (1.0 - coherence)
        return float(base_vol * vol_multiplier)

    def analyze(self, market_data: dict) -> dict:
        """Run full analysis on market data."""
        prices = market_data["prices"]
        returns = market_data["returns"]
        vol = market_data["realized_vol"]
        volume = market_data["volume"]
        n = market_data["n"]

        # Initialize regime detector
        self.regime_detector.initialize(n_training=400, seed=self.seed)

        vol_forecasts = []
        regime_sequence = []
        liquidity_levels = []
        anomaly_scores = []
        signals = []

        for i in range(1, min(n-1, len(returns)+1)):
            # Encode market state as quantum state Φ
            phi = encode_market_state(prices, returns, vol, volume, i)
            # Measurement = realized return at this step
            measurement = float(returns[i-1]) if i-1 < len(returns) else 0.0

            # EKRLS step
            step_result = self.ekrls.step(phi, measurement)

            # Volatility forecast from coherence
            coherence = step_result["coherence"]
            base_vol = vol[i-1] if i > 0 and i-1 < len(vol) else 0.01
            vol_forecast = self._vol_from_coherence(coherence, base_vol)
            vol_forecasts.append(vol_forecast)

            # Market regime detection via QEC → Viterbi codes
            qec_result = self.regime_detector.correct(phi)
            regime_code = qec_result["qec_code"] % 8
            regime_label = REGIME_LABELS[regime_code]
            regime_sequence.append(regime_label)

            # Liquidity battery: charge on calm, discharge on stress
            if coherence > 0.7:
                self.liquidity_battery.charge(0.005)
            else:
                stress = 1.0 - coherence
                self.liquidity_battery.discharge(0.01 * stress)
            liquidity_levels.append(self.liquidity_battery.E_battery)

            # Anomaly score: deviation of predicted vol from realized
            pred_error = abs(step_result.get("pred_error", 0.0))
            anomaly_score = float(np.clip(pred_error / (base_vol + 1e-8), 0, 10))
            anomaly_scores.append(anomaly_score)

            # Trading signal from state
            battery = step_result["battery_level"]
            confidence = qec_result["confidence"]
            signal = self._generate_signal(
                momentum=float(phi[0]),
                vol_z=float(phi[1]),
                coherence=coherence,
                regime=regime_code,
                battery=battery,
                confidence=confidence,
            )
            signals.append(signal)

            # Metacognitive monitoring
            self.metacog.monitor_step(step_result)

            self.results.append({
                "i": i,
                "vol_forecast": vol_forecast,
                "regime": regime_label,
                "regime_code": regime_code,
                "liquidity": self.liquidity_battery.E_battery,
                "anomaly_score": anomaly_score,
                "signal": signal,
                "coherence": coherence,
            })

        return {
            "n_analyzed": len(self.results),
            "vol_forecasts": vol_forecasts,
            "regime_sequence": regime_sequence,
            "liquidity_levels": liquidity_levels,
            "anomaly_scores": anomaly_scores,
            "signals": signals,
        }

    def _generate_signal(self, momentum: float, vol_z: float,
                         coherence: float, regime: int,
                         battery: float, confidence: float) -> dict:
        """
        Generate trading signal from quantum-derived features.

        Rules:
        - High coherence + positive momentum + adequate liquidity → LONG
        - Low coherence + negative momentum + crash regime → SHORT/HEDGE
        - Very low coherence → FLAT (uncertainty too high)
        """
        if coherence < 0.2:
            action = "FLAT"
            strength = 0.0
        elif regime == 6:  # CRASH
            action = "SHORT"
            strength = -0.8 * confidence
        elif coherence > 0.6 and momentum > 0.2 and battery > 0.3:
            action = "LONG"
            strength = min(1.0, momentum * coherence * confidence)
        elif coherence > 0.4 and momentum < -0.3:
            action = "SHORT"
            strength = -min(1.0, abs(momentum) * confidence)
        else:
            action = "FLAT"
            strength = 0.0

        return {"action": action, "strength": round(strength, 4), "confidence": round(confidence, 4)}

    def performance_summary(self) -> dict:
        """Compute summary statistics of the analysis."""
        if not self.results:
            return {}

        regimes = [r["regime"] for r in self.results]
        unique_regimes = list(set(regimes))
        regime_counts = {r: regimes.count(r) for r in unique_regimes}

        signals = [r["signal"]["action"] for r in self.results]
        anomalies = [r for r in self.results if r["anomaly_score"] > 3.0]

        coherences = [r["coherence"] for r in self.results]
        vols = [r["vol_forecast"] for r in self.results]

        return {
            "periods_analyzed": len(self.results),
            "regime_distribution": regime_counts,
            "anomaly_events": len(anomalies),
            "mean_vol_forecast": round(float(np.mean(vols)), 5),
            "mean_coherence": round(float(np.mean(coherences)), 3),
            "signal_breakdown": {
                "LONG": signals.count("LONG"),
                "SHORT": signals.count("SHORT"),
                "FLAT": signals.count("FLAT"),
            },
            "ekrls_rmse": round(self.ekrls.summary().get("rmse", 0), 5),
            "battery_final": round(self.liquidity_battery.E_battery, 4),
            "collapse_events": self.ekrls.summary().get("collapse_events", 0),
        }
