# main.py
# Key metrics to collect:
# run.googleapis.com/container/cpu/utilizations
# run.googleapis.com/container/memory/utilizations
# run.googleapis.com/request_latencies
# run.googleapis.com/container/instance_count


from google.cloud import monitoring_v3, bigquery
import datetime

PROJECT_ID = "vcc-project-52"
BQ_TABLE   = f"{PROJECT_ID}.cloud_metrics.run_metrics"

def collect_and_store(request):
    """HTTP-triggered Cloud Function entry point."""
    monitoring_client = monitoring_v3.MetricServiceClient()
    bq_client = bigquery.Client()

    now = datetime.datetime.utcnow()
    interval = monitoring_v3.TimeInterval()
    interval.end_time.seconds = int(now.timestamp())
    interval.start_time.seconds = int(
        (now - datetime.timedelta(minutes=5)).timestamp()
    )

    metrics_to_collect = [
        "run.googleapis.com/container/cpu/utilizations",
        "run.googleapis.com/container/memory/utilizations",
        "run.googleapis.com/request_latencies",
        "run.googleapis.com/container/instance_count",
    ]

    rows = {}
    for metric_type in metrics_to_collect:
        try:
            results = monitoring_client.list_time_series(
                request={
                    "name": f"projects/{PROJECT_ID}",
                    "filter": f'metric.type="{metric_type}"',
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                }
            )
            for series in results:
                for point in series.points:
                    ts = point.interval.end_time.seconds
                    if ts not in rows:
                        rows[ts] = {"timestamp": datetime.datetime.utcfromtimestamp(ts).isoformat(),
                                    "cpu": None, "memory": None,
                                    "latency": None, "instances": None}
                    if "cpu" in metric_type:
                        rows[ts]["cpu"] = point.value.double_value
                    elif "memory" in metric_type:
                        rows[ts]["memory"] = point.value.double_value
                    elif "latency" in metric_type:
                        rows[ts]["latency"] = point.value.distribution_value.mean
                    elif "instance_count" in metric_type:
                        rows[ts]["instances"] = int(point.value.int64_value)
        except Exception as e:
            print(f"Error collecting {metric_type}: {e}")

    if rows:
        errors = bq_client.insert_rows_json(BQ_TABLE, list(rows.values()))
        if errors:
            print(f"BigQuery errors: {errors}")
        else:
            print(f"✅ Inserted {len(rows)} rows into BigQuery")
    else:
        print("No metric data found in this window")

    return f"Done. {len(rows)} rows inserted.", 200
