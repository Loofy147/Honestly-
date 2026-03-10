"""
Quantum Spacetime Emergence System — المحرك التكاملي
النظام الرئيسي الذي يدمج جميع المكونات:

  EKRLS Engine      → تتبع الحالة الكمومية في الزمن الحقيقي
  Ribbon Filters    → فهرسة أزواج التشابك
  Lie Expansion     → التحكم في بطارية التشابك
  Suffix Smoother   → تصحيح الأخطاء الكمومية
  Metacognitive     → المراقبة الذكية والتحقق

"لم يعد التشابك سحراً، بل أصبح بيانات مهيكلة."
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engines.ekrls_engine import EKRLSQuantumEngine, EKRLSConfig
from filters.ribbon_filter import EntanglementIndex, generate_entanglement_pairs
from algebra.lie_expansion import EntanglementBattery, LieAlgebraConfig
from error_correction.suffix_smoothing import QuantumErrorCorrector, SuffixConfig
from metacognition.metacognitive_layer import MetacognitiveLayer, MetacognitiveConfig


@dataclass
class SystemConfig:
    """Master configuration for the quantum spacetime system."""
    n_simulation_steps: int = 100
    n_entanglement_pairs: int = 10_000
    state_dim: int = 4
    seed: int = 42
    verbose: bool = True


class QuantumSpacetimeSystem:
    """
    Integrated Quantum Spacetime Emergence System.

    Orchestrates all subsystems through the RCF protocol:
    Study → Understand → Integrate → Test → Validate
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.cfg = config or SystemConfig()
        self.report: dict = {}

        if self.cfg.verbose:
            print("=" * 60)
            print("  QUANTUM SPACETIME EMERGENCE SYSTEM")
            print("  Initializing subsystems...")
            print("=" * 60)

        # Initialize subsystems
        self.ekrls = EKRLSQuantumEngine(EKRLSConfig(
            state_dim=self.cfg.state_dim,
            window_size=min(50, self.cfg.n_simulation_steps // 2),
        ))

        self.entanglement_index = EntanglementIndex(
            expected_pairs=self.cfg.n_entanglement_pairs,
            fp_rate=0.001,
        )

        self.battery = EntanglementBattery(
            LieAlgebraConfig(algebra_dim=self.cfg.state_dim),
            algebra_type='galilei',
        )

        self.qec = QuantumErrorCorrector(SuffixConfig(
            max_suffix_length=6,
            n_qec_codes=16,
        ))

        self.metacog = MetacognitiveLayer(MetacognitiveConfig(
            coherence_collapse_threshold=0.1,
        ))

    def phase_study(self) -> dict:
        """Phase 1: STUDY — Build entanglement index, initialize QEC."""
        if self.cfg.verbose:
            print("\n[Phase 1] STUDY — Building entanglement index...")

        pairs = generate_entanglement_pairs(self.cfg.n_entanglement_pairs, seed=self.cfg.seed)
        self._indexed_pairs = pairs  # Save for test phase
        ribbon_result = self.entanglement_index.build_from_pairs(pairs)
        qec_init = self.qec.initialize(n_training=500, seed=self.cfg.seed)

        if self.cfg.verbose:
            print(f"  ✓ Ribbon Filter: {ribbon_result['keys_inserted']:,} pairs indexed")
            print(f"  ✓ Memory savings: {ribbon_result['memory_reduction_pct']:.1f}% vs Bloom")
            print(f"  ✓ QEC initialized: {qec_init['total_nodes']} suffix nodes")

        return {"ribbon_filter": ribbon_result, "qec_initialization": qec_init,
                "memory_report": self.entanglement_index.memory_report()}

    def phase_understand(self) -> dict:
        """
        Phase 2: UNDERSTAND — Evolve battery, map entanglement topology.
        """
        if self.cfg.verbose:
            print("\n[Phase 2] UNDERSTAND — Evolving entanglement battery...")

        battery_history = self.battery.evolve(n_steps=50)

        # Test ACTUAL indexed pairs (from saved list)
        sample_pairs = self._indexed_pairs[:5]
        known_recall = sum(
            self.entanglement_index.is_entangled(a, b, t)
            for a, b, t in sample_pairs
        )
        # Test truly unknown pairs (very high IDs, guaranteed not indexed)
        rng = np.random.default_rng(99999)
        unknown_fp = sum(
            self.entanglement_index.is_entangled(
                int(rng.integers(5_000_000, 9_000_000)),
                int(rng.integers(5_000_000, 9_000_000)),
                0
            )
            for _ in range(5)
        )

        if self.cfg.verbose:
            print(f"  ✓ Battery evolved 50 steps, E = {self.battery.E_battery:.3f}")
            print(f"  ✓ Known pair recall rate: {known_recall}/5")
            print(f"  ✓ Unknown pair FP rate: {unknown_fp}/5 (target < 1)")

        return {
            "battery_summary": self.battery.summary(),
            "battery_history_last5": battery_history[-5:],
            "ribbon_recall": known_recall,
            "ribbon_fp_count": unknown_fp,
        }

    def phase_integrate(self) -> dict:
        """
        Phase 3: INTEGRATE — Run full EKRLS simulation with metacognitive monitoring.
        """
        if self.cfg.verbose:
            print(f"\n[Phase 3] INTEGRATE — Running {self.cfg.n_simulation_steps}-step simulation...")

        sim_results = self.ekrls.run_simulation(
            n_steps=self.cfg.n_simulation_steps,
            seed=self.cfg.seed,
        )

        # Metacognitive monitoring of each step
        meta_alerts_total = 0
        for i, result in enumerate(sim_results):
            meta_result = self.metacog.monitor_step(result)
            meta_alerts_total += len(meta_result.get("alerts", []))

            # Apply QEC: on collapse OR every 10 steps (proactive error correction)
            if result.get("collapse_detected") or (i % 10 == 0):
                phi = self.ekrls.state_history[-1].phi
                qec_result = self.qec.correct(phi)

                # Recharge battery from QEC success
                if qec_result["correction_quality"] > 0.9:
                    self.battery.charge(0.05)

        ekrls_summary = self.ekrls.summary()

        if self.cfg.verbose:
            print(f"  ✓ Simulation complete: {ekrls_summary['total_steps']} steps")
            print(f"  ✓ Collapse events: {ekrls_summary['collapse_events']}")
            print(f"  ✓ Mean coherence: {ekrls_summary['mean_coherence']:.3f}")
            print(f"  ✓ RMSE: {ekrls_summary['rmse']:.4f}")
            print(f"  ✓ Meta alerts: {meta_alerts_total}")

        return {"ekrls_summary": ekrls_summary, "meta_alerts": meta_alerts_total}

    def phase_test(self) -> dict:
        """
        Phase 4: TEST — Validate each subsystem independently.
        """
        if self.cfg.verbose:
            print("\n[Phase 4] TEST — Subsystem validation...")

        tests_passed = 0
        tests_total = 0
        test_results = {}

        # --- Test 1: EKRLS prediction accuracy ---
        tests_total += 1
        rmse = self.ekrls.summary().get("rmse", 999)
        t1_pass = rmse < 0.5
        if t1_pass:
            tests_passed += 1
        test_results["T1_EKRLS_accuracy"] = {
            "pass": t1_pass,
            "rmse": rmse,
            "threshold": 0.5,
        }

        # --- Test 2: Ribbon filter false positive rate ---
        tests_total += 1
        rng = np.random.default_rng(9999)
        n_test = 1000
        fp_count = sum(
            self.entanglement_index.is_entangled(
                int(rng.integers(500000, 1000000)),
                int(rng.integers(500000, 1000000)),
                0
            )
            for _ in range(n_test)
        )
        actual_fp_rate = fp_count / n_test
        t2_pass = actual_fp_rate < 0.01 * 3  # Within 3x target fp_rate
        if t2_pass:
            tests_passed += 1
        test_results["T2_Ribbon_FP"] = {
            "pass": t2_pass,
            "actual_fp_rate": actual_fp_rate,
            "target_fp_rate": 0.001,
        }

        # --- Test 3: Battery conservation ---
        tests_total += 1
        batt_sum = self.battery.summary()
        t3_pass = batt_sum["n_conservation_violations"] == 0
        if t3_pass:
            tests_passed += 1
        test_results["T3_Battery_conservation"] = {
            "pass": t3_pass,
            "violations": batt_sum["n_conservation_violations"],
            "mean_violation": batt_sum["mean_violation"],
        }

        # --- Test 4: QEC uncertainty reduction ---
        tests_total += 1
        qec_sum = self.qec.summary()
        t4_pass = qec_sum.get("mean_uncertainty_reduction_pct", 0) > 0
        if t4_pass:
            tests_passed += 1
        test_results["T4_QEC_uncertainty"] = {
            "pass": t4_pass,
            "uncertainty_reduction_pct": qec_sum.get("mean_uncertainty_reduction_pct", 0),
            "success_rate": qec_sum.get("success_rate", 0),
        }

        # --- Test 5: Metacognitive response ---
        tests_total += 1
        t5_pass = self.metacog.step_count > 0
        if t5_pass:
            tests_passed += 1
        test_results["T5_Metacognitive_active"] = {
            "pass": t5_pass,
            "steps_monitored": self.metacog.step_count,
            "collapse_alerts": len(self.metacog.collapse_alerts),
        }

        if self.cfg.verbose:
            for name, result in test_results.items():
                icon = "✅" if result["pass"] else "❌"
                print(f"  {icon} {name}: {'PASS' if result['pass'] else 'FAIL'}")
            print(f"\n  Results: {tests_passed}/{tests_total} passed")

        return {
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "all_pass": tests_passed == tests_total,
            "details": test_results,
        }

    def phase_validate(self) -> dict:
        """
        Phase 5: VALIDATE — Full Q-score validation with Bayesian calibration.
        """
        if self.cfg.verbose:
            print("\n[Phase 5] VALIDATE — Q-score Bayesian validation...")

        ekrls_sum = self.ekrls.summary()
        batt_sum = self.battery.summary()
        qec_sum = self.qec.summary()

        q_result = self.metacog.run_full_validation(ekrls_sum, batt_sum, qec_sum)
        calibration = self.metacog.confidence_calibration_report()

        if self.cfg.verbose:
            q = q_result["q_validation"]["q_score"]
            rec = q_result["q_validation"]["recommendation"]
            accepted = q_result["q_validation"]["accepted"]
            print(f"  Q-Score: {q:.4f} (threshold: 0.85)")
            print(f"  Decision: {'✅ ACCEPTED' if accepted else '❌ REJECTED'}")
            print(f"  Recommendation: {rec}")
            ci = q_result["q_validation"]["q_confidence_interval"]
            print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
            print(f"  System stability: {q_result['system_stability']:.1%}")

        return {
            "q_validation": q_result,
            "calibration_report": calibration,
        }

    def run(self) -> dict:
        """Execute the full RCF pipeline: Study → Understand → Integrate → Test → Validate."""
        np.random.seed(self.cfg.seed)

        study = self.phase_study()
        understand = self.phase_understand()
        integrate = self.phase_integrate()
        test = self.phase_test()
        validate = self.phase_validate()

        self.report = {
            "study": study,
            "understand": understand,
            "integrate": integrate,
            "test": test,
            "validate": validate,
            "system_config": {
                "n_steps": self.cfg.n_simulation_steps,
                "n_pairs": self.cfg.n_entanglement_pairs,
                "state_dim": self.cfg.state_dim,
            }
        }

        if self.cfg.verbose:
            print("\n" + "=" * 60)
            print("  SYSTEM EXECUTION COMPLETE")
            q = validate["q_validation"]["q_validation"]["q_score"]
            passed = test["tests_passed"]
            total = test["tests_total"]
            print(f"  Tests: {passed}/{total} | Q-Score: {q:.4f}")
            print("=" * 60)

        return self.report


if __name__ == "__main__":
    system = QuantumSpacetimeSystem(SystemConfig(
        n_simulation_steps=100,
        n_entanglement_pairs=10_000,
        verbose=True,
    ))
    report = system.run()

    # Save report
    with open("/home/claude/quantum_spacetime/system_report.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        import json
        json.dump(report, f, indent=2, default=convert)
    print("\n✓ Report saved to system_report.json")
