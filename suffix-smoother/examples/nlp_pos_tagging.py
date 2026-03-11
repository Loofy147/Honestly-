"""
NLP POS Tagging Example — Suffix Smoothing
==========================================
This example demonstrates how to use SuffixSmoother for Part-of-Speech (POS)
tagging, especially focusing on handling Out-of-Vocabulary (OOV) words
using suffix-based backoff.
"""

import time
import numpy as np
from typing import Optional
from collections import defaultdict
from suffix_smoother import SuffixSmoother, SuffixConfig

# --- Constants & Data (Simplified PTB-based distributions) ---

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
UPOS_INV = {tag: i for i, tag in enumerate(UPOS)}
N_TAGS = len(UPOS)

# Empirical suffix probabilities for common English suffixes (from PTB/UD)
PTB_SUFFIX = {
    "ing":  {"VERB": 0.85, "NOUN": 0.10, "ADJ": 0.05},
    "ed":   {"VERB": 0.90, "ADJ": 0.08, "NOUN": 0.02},
    "ly":   {"ADV": 0.95, "ADJ": 0.05},
    "tion": {"NOUN": 0.99},
    "ness": {"NOUN": 0.99},
    "ment": {"NOUN": 0.98},
    "able": {"ADJ": 0.98},
    "ical": {"ADJ": 0.99},
    "ize":  {"VERB": 0.98},
    "ous":  {"ADJ": 0.99},
    "s":    {"NOUN": 0.70, "VERB": 0.25, "PROPN": 0.05},
}

PTB_UNIGRAM = {
    "NOUN": 0.20, "VERB": 0.15, "PUNCT": 0.12, "ADP": 0.10,
    "DET": 0.09, "ADJ": 0.07, "PROPN": 0.06, "PRON": 0.05,
    "ADV": 0.04, "AUX": 0.04, "CCONJ": 0.03, "PART": 0.02,
    "NUM": 0.01, "SCONJ": 0.01, "X": 0.005, "SYM": 0.005
}

class POSTagger:
    def __init__(self, max_suffix: int = 6):
        # Using Witten-Bell smoothing for better OOV performance
        self.smoother = SuffixSmoother(SuffixConfig(
            max_suffix_length=max_suffix,
            n_classes=N_TAGS,
            smoothing_method="witten-bell"
        ))
        self._fitted = False

    def _encode(self, suffix: str, maxlen: int = 5) -> tuple:
        return tuple(ord(c) for c in suffix[-maxlen:])

    def fit(self) -> "POSTagger":
        corpus = []
        rng = np.random.default_rng(42)

        for suffix, tag_probs in PTB_SUFFIX.items():
            tags = list(tag_probs.keys())
            probs = list(tag_probs.values())
            n_samples = 400
            sampled_tags = rng.choice(tags, size=n_samples, p=probs)
            for t in sampled_tags:
                corpus.append((self._encode(suffix), UPOS_INV[t]))

        # Global prior
        uni_tags = list(PTB_UNIGRAM.keys())
        uni_probs = np.array(list(PTB_UNIGRAM.values()))
        uni_probs /= uni_probs.sum()
        for t in rng.choice(uni_tags, size=500, p=uni_probs):
            corpus.append(((), UPOS_INV[t]))

        self.smoother.train(corpus)
        self._fitted = True
        return self

    def tag(self, word: str) -> tuple[str, float]:
        ctx = self._encode(word)
        tag_id, confidence = self.smoother.predict(ctx)
        return UPOS[tag_id], confidence

if __name__ == "__main__":
    tagger = POSTagger().fit()

    test_words = ["running", "quickly", "establishment", "beautiful", "quantumize", "antidisestablishmentarianism"]

    print(f"{'Word':<30} {'Tag':<10} {'Confidence'}")
    print("-" * 50)
    for w in test_words:
        tag, conf = tagger.tag(w)
        print(f"{w:<30} {tag:<10} {conf:.4f}")
