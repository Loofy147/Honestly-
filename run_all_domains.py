"""
Cross-Domain Quantum-Inspired Computing — التشغيل الشامل
=========================================================
Runs all 5 domain adapters and produces a unified results report.

"لم يعد التشابك سحراً، بل أصبح بيانات مهيكلة."
  — والآن هذه البيانات المهيكلة تحل مشاكل في 5 مجالات مختلفة.
"""

import numpy as np
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cross_domain.finance import FinancialQuantumAnalyzer, generate_market_data
from cross_domain.domain_adapters import (
    GenomicsAdapter, ClimateAdapter, DrugDiscoveryAdapter, NLPAdapter
)


def run_all_domains(verbose: bool = True) -> dict:
    """Execute all 5 domain analyses and collect results."""
    np.random.seed(42)
    all_results = {}

    # ─────────────────────────────────────────────────
    # DOMAIN 1: FINANCE
    # ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "═"*60)
        print("  DOMAIN 1: FINANCIAL MARKETS")
        print("  Quantum-Inspired Volatility & Regime Detection")
        print("═"*60)

    market = generate_market_data(n=200, seed=42)
    analyzer = FinancialQuantumAnalyzer(seed=42)
    finance_results = analyzer.analyze(market)
    finance_summary = analyzer.performance_summary()

    if verbose:
        print(f"  ✓ Periods analyzed: {finance_summary['periods_analyzed']}")
        print(f"  ✓ EKRLS RMSE: {finance_summary['ekrls_rmse']}")
        print(f"  ✓ Anomaly events: {finance_summary['anomaly_events']}")
        print(f"  ✓ Regime distribution: {finance_summary['regime_distribution']}")
        print(f"  ✓ Signals: LONG={finance_summary['signal_breakdown']['LONG']} | "
              f"SHORT={finance_summary['signal_breakdown']['SHORT']} | "
              f"FLAT={finance_summary['signal_breakdown']['FLAT']}")
        print(f"  ✓ Liquidity battery: {finance_summary['battery_final']:.4f}")

        # Show last 5 signals
        last_signals = analyzer.results[-5:]
        print("\n  Last 5 market snapshots:")
        for r in last_signals:
            print(f"    t={r['i']:3d} | regime={r['regime']:<16} | "
                  f"signal={r['signal']['action']:<5} "
                  f"(strength={r['signal']['strength']:+.3f}) | "
                  f"vol_forecast={r['vol_forecast']:.4f}")

    all_results["finance"] = finance_summary

    # ─────────────────────────────────────────────────
    # DOMAIN 2: GENOMICS
    # ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "═"*60)
        print("  DOMAIN 2: GENOMICS")
        print("  Ribbon Filter for SNP Indexing + Variant Classification")
        print("═"*60)

    genomics = GenomicsAdapter(n_variants=50_000)
    genomics_build = genomics.build_variant_database(seed=42)

    if verbose:
        print(f"  ✓ Variants indexed: {genomics_build['variants_indexed']:,}")
        print(f"  ✓ Ribbon memory: {genomics_build['memory_kb']:.1f} KB")
        print(f"  ✓ Bloom equivalent: {genomics_build['bloom_equiv_kb']:.1f} KB")
        print(f"  ✓ Memory reduction: {genomics_build['memory_reduction_pct']:.1f}%")
        print(f"  ✓ Suffix tree nodes: {genomics_build['suffix_nodes']}")

    # Query test variants
    test_queries = [
        (1,  923456,  "A", "T", "ATGCA"),
        (17, 7674220, "G", "C", "TGCGA"),
        (7,  55249063,"C", "T", "CTGAT"),  # EGFR-like
    ]
    variant_reports = []
    for chrom, pos, ref, alt, ctx in test_queries:
        q = genomics.query_variant(chrom, pos, ref, alt, ctx)
        variant_reports.append(q)
        if verbose:
            print(f"\n  Variant chr{chrom}:{pos} {ref}>{alt}")
            print(f"    In database: {q['in_database']}")
            print(f"    Predicted class: {q['predicted_class']} "
                  f"(confidence={q['confidence']:.1%})")
            print(f"    Uncertainty reduction: {q['uncertainty_reduction_pct']:.1f}%")

    # Novel variant prediction
    novel = genomics.predict_novel_variant("GCGCG")
    if verbose:
        print(f"\n  Novel variant (CpG island context: GCGCG):")
        print(f"    → {novel['novel_variant_class']} "
              f"(confidence={novel['confidence']:.1%})")
        print(f"    Method: {novel['method']}")

    all_results["genomics"] = {
        "build": genomics_build,
        "test_queries": variant_reports,
        "novel_prediction": novel,
    }

    # ─────────────────────────────────────────────────
    # DOMAIN 3: CLIMATE
    # ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "═"*60)
        print("  DOMAIN 3: CLIMATE ANOMALY DETECTION")
        print("  EKRLS Tracking + Energy Battery + Tipping Point Detection")
        print("═"*60)

    climate = ClimateAdapter(seed=42)
    climate_series = climate.generate_climate_series(n=300, seed=42)
    climate_results = climate.analyze(climate_series)

    if verbose:
        print(f"  ✓ Steps analyzed: {climate_results['n_steps']}")
        print(f"  ✓ Anomalies detected: {climate_results['anomalies_detected']}")
        print(f"  ✓ Tipping point detected: {climate_results['tipping_point_detected']}")
        print(f"  ✓ Mean coherence (stability): {climate_results['mean_coherence']}")
        print(f"  ✓ Earth energy imbalance: {climate_results['energy_final']:.4f}")
        print(f"  ✓ Conservation violations: {climate_results['conservation_violations']}")
        print(f"  ✓ EKRLS tracking RMSE: {climate_results['ekrls_rmse']}")

        if climate_results["anomaly_events"]:
            print("\n  Top anomaly events:")
            for ev in climate_results["anomaly_events"][:4]:
                print(f"    step={ev['step']:3d} | type={ev['type']:<15} | "
                      f"coherence={ev['coherence']:.3f} | "
                      f"T_anomaly={ev['temperature_anomaly']:+.3f}")

    all_results["climate"] = climate_results

    # ─────────────────────────────────────────────────
    # DOMAIN 4: DRUG DISCOVERY
    # ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "═"*60)
        print("  DOMAIN 4: DRUG DISCOVERY")
        print("  Compound Indexing + Activity Prediction + Docking Validation")
        print("═"*60)

    drug = DrugDiscoveryAdapter(n_compounds=30_000)
    drug_build = drug.build_compound_database(seed=42)

    if verbose:
        print(f"  ✓ Compounds indexed: {drug_build['compounds_indexed']:,}")
        print(f"  ✓ Memory reduction: {drug_build['memory_reduction_pct']:.1f}%")
        print(f"  ✓ Binding energy stored: {drug_build['binding_energy_stored']:.3f}")

    # Predict activity for sample compounds
    compound_profiles = [
        {"name": "Drug-A (Lipinski-favorable)", "rings": 2, "hbd": 2, "hba": 4, "mw": 2, "logp": 2},
        {"name": "Drug-B (High mol. weight)",   "rings": 5, "hbd": 4, "hba": 7, "mw": 4, "logp": 4},
        {"name": "Drug-C (Lead-like)",           "rings": 1, "hbd": 1, "hba": 3, "mw": 1, "logp": 1},
    ]
    activity_reports = []
    for cp in compound_profiles:
        pred = drug.predict_activity(
            cp["rings"], cp["hbd"], cp["hba"], cp["mw"], cp["logp"]
        )
        activity_reports.append({**cp, **pred})
        if verbose:
            print(f"\n  {cp['name']}")
            print(f"    Predicted activity: {pred['predicted_activity']} "
                  f"(confidence={pred['confidence']:.1%})")
            print(f"    Lipinski OK: {pred['lipinski_favorable']}")
            print(f"    Top classes: {pred['top_4_classes'][:2]}")

    # Validate a docking model using Q-Score
    docking_scores = {
        'G': 0.88,   # Grounded in MM-GBSA
        'C': 0.80,   # Consistent across conformers
        'S': 0.85,   # Clear binding pose structure
        'A': 0.75,   # Tested on 3 target classes
        'Co': 0.90,  # Internally coherent
        'Ge': 0.70,  # Generates 2 SAR hypotheses
    }
    q_result = drug.validate_docking_model(docking_scores)
    if verbose:
        print(f"\n  Docking Model Q-Score: {q_result['q_score']:.4f} "
              f"→ {'ACCEPTED' if q_result['accepted'] else 'REJECTED'}")
        print(f"  Recommendation: {q_result['recommendation']}")

    all_results["drug_discovery"] = {
        "build": drug_build,
        "activity_predictions": activity_reports,
        "docking_validation": q_result,
    }

    # ─────────────────────────────────────────────────
    # DOMAIN 5: NLP
    # ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "═"*60)
        print("  DOMAIN 5: NATURAL LANGUAGE PROCESSING")
        print("  Suffix Smoothing POS Tagger + Viterbi Sequence Decoder")
        print("═"*60)

    nlp = NLPAdapter()
    corpus = nlp.generate_synthetic_corpus(n=2000, seed=42)
    nlp_train = nlp.train(corpus, seed=42)

    if verbose:
        print(f"  ✓ Training pairs: {nlp_train['training_pairs']:,}")
        print(f"  ✓ Suffix nodes: {nlp_train['suffix_nodes']}")

    # Tag individual words (including OOV)
    test_words = ["running", "beautiful", "quantum", "xkrtlmn",
                  "quickly", "establishment", "antidisestablishmentarianism"]
    word_reports = []
    for w in test_words:
        r = nlp.tag_word(w)
        word_reports.append(r)
        if verbose:
            print(f"\n  '{w}'")
            print(f"    Tag: {r['predicted_tag']} (confidence={r['confidence']:.1%})")
            print(f"    Uncertainty: {r['uncertainty_bits']:.2f} bits")
            print(f"    Top 3: {r['top_3']}")

    # Tag a full sentence with Viterbi
    sentence = ["The", "quantum", "entanglement", "phenomenon", "enables", "faster", "computation"]
    tagged_sentence = nlp.tag_sequence(sentence)
    if verbose:
        print("\n  Full sentence Viterbi tagging:")
        print("  " + " ".join(f"{r['word']}/{r['tag']}" for r in tagged_sentence))

    all_results["nlp"] = {
        "training": nlp_train,
        "word_tagging": word_reports,
        "sentence_tagging": tagged_sentence,
    }

    # ─────────────────────────────────────────────────
    # UNIFIED SUMMARY
    # ─────────────────────────────────────────────────
    if verbose:
        print("\n" + "═"*60)
        print("  CROSS-DOMAIN UNIFIED SUMMARY")
        print("═"*60)
        print(f"""
  ┌─────────────────────┬──────────────────────────────────────────┐
  │ Domain              │ Key Metric                               │
  ├─────────────────────┼──────────────────────────────────────────┤
  │ Finance             │ RMSE={finance_summary['ekrls_rmse']} | {finance_summary['anomaly_events']} anomalies | {finance_summary['signal_breakdown']['LONG']}L/{finance_summary['signal_breakdown']['SHORT']}S signals │
  │ Genomics            │ {genomics_build['memory_reduction_pct']:.1f}% memory saved | OOV variants handled  │
  │ Climate             │ {climate_results['anomalies_detected']} anomalies | tipping point={climate_results['tipping_point_detected']}          │
  │ Drug Discovery      │ Q={q_result['q_score']:.3f} docking | {drug_build['memory_reduction_pct']:.1f}% compound index savings │
  │ NLP                 │ {nlp_train['suffix_nodes']} suffix nodes | Viterbi-optimal tagging   │
  └─────────────────────┴──────────────────────────────────────────┘

  Core Engines Used Across All 5 Domains:
    EKRLS  → Finance (vol), Climate (anomaly)
    Ribbon → Genomics (SNP), Drug Discovery (compounds)
    Suffix → Genomics (variants), Drug (activity), NLP (POS)
    Lie    → Climate (energy conservation)
    Q-Score→ Drug Discovery (model validation)
    Viterbi→ Finance (regime), NLP (tagging)
        """)

    return all_results


if __name__ == "__main__":
    results = run_all_domains(verbose=True)

    # Save
    def json_clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    out_path = os.path.join(os.path.dirname(__file__), "cross_domain_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=json_clean)
    print(f"\n✓ Full results saved to cross_domain_results.json")
