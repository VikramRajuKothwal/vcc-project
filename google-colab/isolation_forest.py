# ══════════════════════════════════════════════════════════════════
# Imports & Setup
# ══════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# Load & Inspect
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv("/content/drive/MyDrive/vcc-project/metrics_cleaned.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Rows       : {len(df)}")
print(f"Columns    : {list(df.columns)}")
print(f"Time range : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
print(f"\nNull values:\n{df.isnull().sum()}")
df.head()

# ══════════════════════════════════════════════════════════════════
# Feature Selection & Scaling
# ══════════════════════════════════════════════════════════════════
FEATURES = ["cpu", "memory", "latency", "instances"]

X = df[FEATURES].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix shape: {X_scaled.shape}")

# ══════════════════════════════════════════════════════════════════
# Train Isolation Forest
# ══════════════════════════════════════════════════════════════════
# contamination = expected fraction of anomalies in training data
# Since this is all "normal" data, set it low (0.01–0.05)
iso_forest = IsolationForest(
    n_estimators=200,        # more trees = more stable
    contamination=0.03,      # ~3% of rows may be edge-case normal
    max_samples="auto",
    random_state=42,
    verbose=0
)

iso_forest.fit(X_scaled)
print("✅ Isolation Forest trained.")

# ══════════════════════════════════════════════════════════════════
# Predict & Score on Training Data
# ══════════════════════════════════════════════════════════════════
# predict() returns: 1 = normal, -1 = anomaly
df["if_prediction"] = iso_forest.predict(X_scaled)
df["if_score"]      = iso_forest.score_samples(X_scaled)  # lower = more anomalous

# Map to human-readable
df["if_label"] = df["if_prediction"].map({1: "normal", -1: "anomaly"})

normal_count  = (df["if_label"] == "normal").sum()
anomaly_count = (df["if_label"] == "anomaly").sum()
print(f"\nNormal    : {normal_count}  ({normal_count/len(df)*100:.1f}%)")
print(f"Anomaly   : {anomaly_count}  ({anomaly_count/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# View Flagged Anomalies
# ══════════════════════════════════════════════════════════════════
anomalies = df[df["if_label"] == "anomaly"][
    ["timestamp", "cpu", "memory", "latency", "instances", "if_score"]
].sort_values("if_score")

print(f"\nFlagged anomalies ({len(anomalies)} rows):")
print(anomalies.to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# Anomaly Score Threshold (for inference/deployment)
# ══════════════════════════════════════════════════════════════════
threshold = df[df["if_label"] == "normal"]["if_score"].quantile(0.05)
print(f"\n📌 Decision threshold (5th percentile of normal scores): {threshold:.6f}")
print("→ Use this threshold in your Flask API: if score < threshold → ANOMALY")

# ══════════════════════════════════════════════════════════════════
# Visualizations
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle("Isolation Forest — Anomaly Detection on Normal Baseline", fontsize=14, fontweight="bold")

metrics_to_plot = [
    ("cpu",       "CPU Usage",         "steelblue"),
    ("memory",    "Memory Usage",      "darkorange"),
    ("latency",   "Latency (ms)",      "green"),
    ("instances", "Instance Count",    "purple"),
]

for ax, (col, title, color) in zip(axes, metrics_to_plot):
    ax.plot(df["timestamp"], df[col], color=color, linewidth=0.9, label=col)
    anomaly_mask = df["if_label"] == "anomaly"
    ax.scatter(df.loc[anomaly_mask, "timestamp"],
               df.loc[anomaly_mask, col],
               color="red", s=40, zorder=5, label="Flagged Anomaly")
    ax.set_ylabel(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Timestamp (UTC)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/vcc-project/isolation_forest_anomalies.png", dpi=150)
plt.show()
print("✅ Plot saved as isolation_forest_anomalies.png")

# ══════════════════════════════════════════════════════════════════
# Anomaly Score Distribution Plot
# ══════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 4))
plt.hist(df["if_score"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
plt.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.4f}")
plt.title("Isolation Forest — Anomaly Score Distribution")
plt.xlabel("Anomaly Score (lower = more anomalous)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("if_score_distribution.png", dpi=150)
plt.show()
print("✅ Score distribution plot saved.")

# ══════════════════════════════════════════════════════════════════
# Save Model & Scaler for Flask API
# ══════════════════════════════════════════════════════════════════
import joblib

joblib.dump(iso_forest, "/content/drive/MyDrive/vcc-project/isolation_forest_model.pkl")
joblib.dump(scaler,     "/content/drive/MyDrive/vcc-project/if_scaler.pkl")

# Save threshold
import json
with open("/content/drive/MyDrive/vcc-project/if_threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f)

print("✅ Saved:")
print("   → isolation_forest_model.pkl")
print("   → if_scaler.pkl")
print("   → if_threshold.json")
print(f"\n📌 Threshold value: {threshold:.6f}")