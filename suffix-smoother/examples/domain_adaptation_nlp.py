import sys, os
import numpy as np
from nlp_tagger import POSTagger

def run_demo():
    print("--- [v0.3.0] POS Tagging Domain Adaptation Demo ---")

    # 1. Base Tagger (General PTB)
    print("Fitting General Tagger...")
    base_tagger = POSTagger().fit()

    # 2. Medical Specialized Tagger
    # Medical context: 'establishment' -> NOUN, 'establishment' in a specific bio-context might be different
    # Let's say 'quantumize' is a VERB in general but in medical it might be a specific PROCESS (NOUN)
    # This is synthetic for demo.
    medical_data = [
        ("quantumize", "NOUN"), # In medical domain, let's pretend it's a noun
        ("stabilization", "NOUN"),
        ("cardio-establishment", "NOUN"),
    ]
    medical_tagger = POSTagger()
    # Manual training of medical tagger
    corpus = []
    for word, tag in medical_data:
        from nlp_tagger import UPOS_INV
        tag_id = UPOS_INV.get(tag, 12)
        for ln in range(1, min(len(word)+1, 7)):
            ctx = medical_tagger._encode(word[-ln:])
            corpus.append((ctx, tag_id))
    medical_tagger.smoother.train(corpus)

    # 3. Merge models
    print("\nMerging General + Medical models (weighted 1:5)...")
    from suffix_smoother import SuffixSmoother
    merged_smoother = SuffixSmoother.merge_weighted(base_tagger.smoother, medical_tagger.smoother, w_a=1.0, w_b=5.0)

    # 4. Compare
    test_word = "quantumize"

    base_pred = base_tagger._tag_word(test_word)[0]

    # For merged, we need a wrapper or just check smoother directly
    from nlp_tagger import UPOS
    merged_dist = merged_smoother.predict_distribution(base_tagger._encode(test_word))
    merged_pred = UPOS[max(merged_dist, key=merged_dist.get)]

    print(f"\nWord: '{test_word}'")
    print(f"  General Tagger Prediction: {base_pred}")
    print(f"  Merged Tagger Prediction:  {merged_pred}")

    if merged_pred == "NOUN":
        print("\n  ✓ Domain adaptation successful: Merged model correctly prioritized specialized knowledge.")
    else:
        print("\n  ❌ Domain adaptation failed.")

if __name__ == "__main__":
    export_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, export_path)
    run_demo()
