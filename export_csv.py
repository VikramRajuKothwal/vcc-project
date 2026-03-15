from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()
df = client.query("""
    SELECT timestamp, cpu, memory, latency, instances
    FROM `vcc-project-52.cloud_metrics.run_metrics`
    ORDER BY timestamp ASC
""").to_dataframe()
df["label"] = "normal"
df.to_csv("metrics_normal.csv", index=False)
print(f"Exported {len(df)} rows to metrics_actual.csv")
