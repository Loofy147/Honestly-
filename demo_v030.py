from suffix_smoother import SuffixSmoother, SuffixConfig
import numpy as np

# 1. Setup two models for merging
config = SuffixConfig(n_classes=3, smoothing_method="witten-bell")
model_a = SuffixSmoother(config)
model_b = SuffixSmoother(config)

# Train model_a on some patterns
model_a.train([
    ((1, 2, 3), 0),
    ((1, 2, 3), 0),
    ((4, 5, 6), 1),
])

# Train model_b on other patterns
model_b.train([
    ((7, 8, 9), 2),
    ((4, 5, 6), 1),
])

print("--- Model Merging ---")
merged = SuffixSmoother.merge(model_a, model_b)
print(f"Merged model nodes: {merged.n_nodes}")
print(f"Merged samples: {merged.training_samples}")
print(f"Predict (1,2,3) -> {merged.predict((1,2,3))[0]} (expected 0)")
print(f"Predict (7,8,9) -> {merged.predict((7,8,9))[0]} (expected 2)")

# 2. Batch Prediction
print("\n--- Batch Prediction ---")
queries = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (0, 0, 0)]
results = merged.predict_batch(queries)
for q, r in zip(queries, results):
    print(f"Query {q} -> Label {r[0]}, Conf {r[1]:.4f}")

# 3. Feature Importance
print("\n--- Feature Importance (Top 3) ---")
importance = merged.feature_importance(top_n=3)
for i, feat in enumerate(importance):
    print(f"{i+1}. Suffix {feat['suffix']} -> Top Label {feat['top_label']} (KL: {feat['kl_divergence']:.4f})")

# 4. Model Comparison
print("\n--- Model Comparison ---")
# Create a Jelinek-Mercer model for comparison
config_jm = SuffixConfig(n_classes=3, smoothing_method="jelinek-mercer")
model_jm = SuffixSmoother(config_jm)
model_jm.train([((1, 2, 3), 0), ((4, 5, 6), 1), ((7, 8, 9), 2)])

test_data = [((1, 2, 3), 0), ((4, 5, 6), 1), ((7, 8, 9), 2)]
comparison = SuffixSmoother.compare([
    ("Witten-Bell (Merged)", merged),
    ("Jelinek-Mercer", model_jm)
], test_data)

for row in comparison:
    print(row)
