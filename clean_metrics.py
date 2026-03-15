# -*- coding: utf-8 -*-

import pandas as pd

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("metrics_actual.csv")
print(f"Raw rows        : {len(df)}")

# ── 2. Fill missing latency with 0 (before locust started) ───────────────────
df["latency"] = df["latency"].fillna(0)

# ── 3. Drop rows where memory is NaN (partial/incomplete metric series) ───────
df = df.dropna(subset=["memory"])
print(f"After dropna    : {len(df)}")

# ── 4. Deduplicate — first by all columns, then enforce 1 row per timestamp ──
df = df.drop_duplicates(subset=["timestamp", "cpu", "memory", "latency", "instances"])
df = df.drop_duplicates(subset=["timestamp"], keep="first")
print(f"After dedup     : {len(df)}")

# ── 5. Parse and sort timestamps ──────────────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 6. Verify ─────────────────────────────────────────────────────────────────
print(f"\nFinal row count : {len(df)}")
print(f"Time range      : {df['timestamp'].min()}  →  {df['timestamp'].max()}")
print(f"Null values     :\n{df.isnull().sum()}")
print(f"Duplicate timestamps: {df.duplicated(subset=['timestamp']).sum()}")
print(f"\nStats:\n{df[['cpu','memory','latency','instances']].describe().round(4)}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
df.to_csv("metrics_clean.csv", index=False)
print("\n✅ Saved as metrics_cleaned.csv")
