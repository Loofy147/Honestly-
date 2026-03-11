import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from suffix_smoother import SuffixSmoother, SuffixConfig

# Example: Detecting anomalous log sequences
# Classes: 0 (Normal), 1 (Anomaly)
config = SuffixConfig(max_suffix_length=4, n_classes=2)
smoother = SuffixSmoother(config)

# Training data (highly simplified log event IDs)
# 101: LOGIN, 102: VIEW, 103: LOGOUT, 404: ERROR, 500: CRASH
training = [
    ((101, 102, 103), 0),
    ((101, 102, 102, 103), 0),
    ((101, 404, 404, 404), 1),
    ((102, 102, 500), 1),
    ((101, 500), 1),
]

smoother.train(training)

print("--- Log Anomaly Detection ---")
logs = [
    (101, 102, 103),       # Normal
    (101, 404, 404),       # Likely Anomaly
    (999, 102, 103),       # Unseen start, but normal suffix
    (101, 102, 500),       # Known anomaly
]

for log in logs:
    label, conf = smoother.predict(log)
    status = "ANOMALY" if label == 1 else "NORMAL"
    print(f"Sequence {log} -> {status} (confidence: {conf:.2f})")
