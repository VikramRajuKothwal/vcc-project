# ══════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, joblib, os

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# All paths point to your Drive folder
DRIVE_PATH   = "/content/drive/MyDrive/vcc-project/"
CLEAN_CSV    = DRIVE_PATH + "metrics_cleaned.csv"
MODEL_PATH   = DRIVE_PATH + "lstm_autoencoder.keras"
SCALER_PATH  = DRIVE_PATH + "lstm_scaler.pkl"
THRESH_PATH  = DRIVE_PATH + "lstm_threshold.json"
PLOT_PATH    = DRIVE_PATH + "lstm_training_loss.png"
RECON_PATH   = DRIVE_PATH + "lstm_reconstruction_error.png"

print("✅ Drive mounted. Paths configured.")

# ══════════════════════════════════════════════════════════════════
# Load & Prepare Data
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv(CLEAN_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Rows       : {len(df)}")
print(f"Time range : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
print(f"Null values:\n{df.isnull().sum()}")

FEATURES = ["cpu", "memory", "latency", "instances"]
data = df[FEATURES].values

# ══════════════════════════════════════════════════════════════════
# Scale Data
# ══════════════════════════════════════════════════════════════════
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved → {SCALER_PATH}")

# ══════════════════════════════════════════════════════════════════
# Create Sliding Window Sequences
# ══════════════════════════════════════════════════════════════════
WINDOW_SIZE = 10   # 10-minute rolling window
N_FEATURES  = len(FEATURES)

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)

X = create_sequences(data_scaled, WINDOW_SIZE)
print(f"Sequence shape : {X.shape}")  # (samples, 10, 4)

# 80/20 train-val split (no shuffle — time series!)
split = int(len(X) * 0.8)
X_train = X[:split]
X_val   = X[split:]
print(f"Train sequences: {X_train.shape}")
print(f"Val   sequences: {X_val.shape}")

# ══════════════════════════════════════════════════════════════════
# Build LSTM Autoencoder
# ══════════════════════════════════════════════════════════════════
def build_lstm_autoencoder(window_size, n_features, latent_dim=32):
    inputs = Input(shape=(window_size, n_features), name="encoder_input")

    # ── Encoder ──────────────────────────────────────────────────
    x = LSTM(64, activation="tanh", return_sequences=True, name="enc_lstm1")(inputs)
    x = LSTM(latent_dim, activation="tanh", return_sequences=False, name="enc_lstm2")(x)

    # ── Bottleneck ───────────────────────────────────────────────
    encoded = RepeatVector(window_size, name="bottleneck")(x)

    # ── Decoder ──────────────────────────────────────────────────
    x = LSTM(latent_dim, activation="tanh", return_sequences=True, name="dec_lstm1")(encoded)
    x = LSTM(64, activation="tanh", return_sequences=True, name="dec_lstm2")(x)
    outputs = TimeDistributed(Dense(n_features), name="reconstruction")(x)

    model = Model(inputs, outputs, name="LSTM_Autoencoder")
    return model

model = build_lstm_autoencoder(WINDOW_SIZE, N_FEATURES, latent_dim=32)
model.compile(optimizer="adam", loss="mse")
model.summary()

# ══════════════════════════════════════════════════════════════════
# Train
# ══════════════════════════════════════════════════════════════════
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train, X_train,          # autoencoder: input = target
    epochs=100,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    shuffle=False,             # keep time order
    verbose=1
)

print(f"✅ Best model saved → {MODEL_PATH}")

# ══════════════════════════════════════════════════════════════════
# Training Loss Plot
# ══════════════════════════════════════════════════════════════════
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"],     label="Train Loss", color="steelblue")
plt.plot(history.history["val_loss"], label="Val Loss",   color="darkorange")
plt.title("LSTM Autoencoder — Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.show()
print(f"✅ Loss plot saved → {PLOT_PATH}")

# ══════════════════════════════════════════════════════════════════
# Reconstruction Error on Full Dataset
# ══════════════════════════════════════════════════════════════════
X_full = create_sequences(data_scaled, WINDOW_SIZE)
X_pred = model.predict(X_full, verbose=0)

# Mean reconstruction error per sequence (across timesteps & features)
recon_errors = np.mean(np.mean(np.abs(X_full - X_pred), axis=2), axis=1)

print(f"Reconstruction error stats:")
print(f"  Min    : {recon_errors.min():.6f}")
print(f"  Max    : {recon_errors.max():.6f}")
print(f"  Mean   : {recon_errors.mean():.6f}")
print(f"  Std    : {recon_errors.std():.6f}")

# ══════════════════════════════════════════════════════════════════
# Set Anomaly Threshold
# ══════════════════════════════════════════════════════════════════
# Threshold = mean + 2*std (covers ~95% of normal behaviour)
threshold = float(recon_errors.mean() + 2 * recon_errors.std())
print(f"\n📌 LSTM Anomaly Threshold : {threshold:.6f}")
print("→ Sequences with recon_error > threshold are flagged as ANOMALY")

with open(THRESH_PATH, "w") as f:
    json.dump({"threshold": threshold, "window_size": WINDOW_SIZE}, f)
print(f"✅ Threshold saved → {THRESH_PATH}")

# ══════════════════════════════════════════════════════════════════
# Reconstruction Error Plot
# ══════════════════════════════════════════════════════════════════
# Align timestamps: each sequence maps to its last timestamp
timestamps = df["timestamp"].values[WINDOW_SIZE - 1:]

plt.figure(figsize=(16, 5))
plt.plot(timestamps, recon_errors, color="steelblue", linewidth=0.9, label="Reconstruction Error")
plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.4f}")
anomaly_mask = recon_errors > threshold
plt.scatter(timestamps[anomaly_mask], recon_errors[anomaly_mask],
            color="red", s=40, zorder=5, label="Flagged Anomaly")
plt.title("LSTM Autoencoder — Reconstruction Error over Time")
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.xticks(rotation=30)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RECON_PATH, dpi=150)
plt.show()
print(f"✅ Reconstruction error plot saved → {RECON_PATH}")

# ══════════════════════════════════════════════════════════════════
# Anomaly Summary
# ══════════════════════════════════════════════════════════════════
anomaly_indices  = np.where(anomaly_mask)[0]
anomaly_times    = timestamps[anomaly_mask]
anomaly_errors   = recon_errors[anomaly_mask]

print(f"\nFlagged anomalies : {anomaly_mask.sum()} sequences")
print(f"{'Timestamp':<40} {'Recon Error'}")
print("-" * 55)
for t, e in zip(anomaly_times, anomaly_errors):
    print(f"{str(t):<40} {e:.6f}")

# ══════════════════════════════════════════════════════════════════
# Verify All Saved Files
# ══════════════════════════════════════════════════════════════════
files_to_check = [MODEL_PATH, SCALER_PATH, THRESH_PATH, PLOT_PATH, RECON_PATH]
print("\n📁 Files saved to Google Drive:")
for f in files_to_check:
    exists = os.path.exists(f)
    size   = os.path.getsize(f) // 1024 if exists else 0
    status = f"✅  {size} KB" if exists else "❌  NOT FOUND"
    print(f"  {os.path.basename(f):<40} {status}")
