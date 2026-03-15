import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from anomaly_publisher import publish_anomaly
from tensorflow import keras

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ID     = "vcc-project-52"
DATASET        = "cloud_metrics"
TABLE          = "run_metrics"
SEQUENCE_LEN   = 10

IF_THRESHOLD   = -0.5849
LSTM_THRESHOLD = 0.1295

FEATURES = ["latency", "cpu", "memory", "instances"]

# ── Load models & scalers ────────────────────────────────────────────────
print("Loading models...")
if_scaler   = joblib.load("if_scaler.pkl")
if_model    = joblib.load("isolation_forest_model.pkl")
lstm_scaler = joblib.load("lstm_scaler.pkl")
lstm_model  = keras.models.load_model("lstm_autoencoder.keras")
print("All models loaded successfully.\n")

# ── Fetch latest rows from BigQuery ─────────────────────────────────────
def fetch_latest_rows(n=20):
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT timestamp, latency, cpu, memory, instances
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        ORDER BY timestamp DESC
        LIMIT {n}
    """
    job_config = bigquery.QueryJobConfig()
    df = client.query(query, location="US", job_config=job_config).to_dataframe()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ── Isolation Forest Scoring ─────────────────────────────────────────────
def score_isolation_forest(df):
    X = df[FEATURES].values
    X_scaled = if_scaler.transform(X)
    scores = if_model.score_samples(X_scaled)

    anomaly_found = False
    for i, score in enumerate(scores):
        ts = str(df["timestamp"].iloc[i])
        if score < IF_THRESHOLD:
            anomaly_found = True
            print(f"  [ANOMALY] score={score:.4f} at {ts}")
            publish_anomaly(
                metric="latency",
                score=round(float(score), 4),
                timestamp=ts,
                model_name="IsolationForest"
            )
        else:
            print(f"  [Normal]  score={score:.4f} at {ts}")

    if not anomaly_found:
        print("  → No anomalies detected by Isolation Forest.")

# ── LSTM Autoencoder Scoring ─────────────────────────────────────────────
def score_lstm(df):
    X = df[FEATURES].values
    X_scaled = lstm_scaler.transform(X)

    if len(X_scaled) < SEQUENCE_LEN:
        print(f"  Not enough rows ({len(X_scaled)}), need {SEQUENCE_LEN}. Skipping.")
        return

    seq = X_scaled[-SEQUENCE_LEN:]
    seq = seq.reshape(1, SEQUENCE_LEN, len(FEATURES))

    reconstructed = lstm_model.predict(seq, verbose=0)
    mse = float(np.mean((seq - reconstructed) ** 2))
    ts = str(df["timestamp"].iloc[-1])

    print(f"  Reconstruction error (MSE) = {mse:.6f}  |  threshold = {LSTM_THRESHOLD}")
    print(f"  Timestamp: {ts}")

    if mse > LSTM_THRESHOLD:
        print(f"  [ANOMALY] LSTM error {mse:.6f} exceeds threshold!")
        publish_anomaly(
            metric="latency",
            score=round(mse, 6),
            timestamp=ts,
            model_name="LSTM_Autoencoder"
        )
    else:
        print(f"  [Normal]  LSTM reconstruction looks fine.")

# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching latest metrics from BigQuery...")
    df = fetch_latest_rows(n=20)
    print(f"Fetched {len(df)} rows.")
    print(f"Time range: {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}\n")

    print("=" * 50)
    print("ISOLATION FOREST SCORING")
    print("=" * 50)
    score_isolation_forest(df)

    print()
    print("=" * 50)
    print("LSTM AUTOENCODER SCORING")
    print("=" * 50)
    score_lstm(df)

    print()
    print("Done. Check m25ai2033@iitj.ac.in for alert emails.")
