# anomaly_publisher.py
from google.cloud import pubsub_v1
import json

publisher  = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("vcc-project-52", "anomaly-alerts")

def publish_anomaly(metric, score, timestamp, model_name):
    severity = "HIGH" if (
        (model_name == "IsolationForest"  and abs(score) > 0.7) or
        (model_name == "LSTM_Autoencoder" and score > 0.18)
    ) else "MEDIUM"

    message = {
        "metric":    metric,
        "score":     float(score),
        "timestamp": timestamp,
        "model":     model_name,
        "severity":  severity
    }
    future = publisher.publish(
        topic_path,
        json.dumps(message).encode("utf-8")
    )
    print(f"✅ Published [{severity}] anomaly alert: {future.result()}")
