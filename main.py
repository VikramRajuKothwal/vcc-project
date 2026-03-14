# main.py
# Key metrics to collect:
# run.googleapis.com/container/cpu/utilizations
# run.googleapis.com/container/memory/utilizations
# run.googleapis.com/request_latencies
# run.googleapis.com/container/instance_count


from google.cloud import monitoring_v3, bigquery
import datetime, math

PROJECT_ID = "vcc-project-52"
BQ_TABLE   = f"{PROJECT_ID}.cloud_metrics.run_metrics"

def collect_and_store(request):
    monitoring_client = monitoring_v3.MetricServiceClient()
    bq_client = bigquery.Client()

    now = datetime.datetime.now(datetime.timezone.utc)
    start = now - datetime.timedelta(minutes=60)

    interval = monitoring_v3.TimeInterval({
        "end_time":   {"seconds": int(now.timestamp()),   "nanos": 0},
        "start_time": {"seconds": int(start.timestamp()), "nanos": 0},
    })

    def fetch(metric_type):
        return list(monitoring_client.list_time_series(request={
            "name": f"projects/{PROJECT_ID}",
            "filter": f'metric.type="{metric_type}"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }))

    rows = {}

    for series in fetch("run.googleapis.com/container/instance_count"):
        for point in series.points:
            ts = int(point.interval.end_time.timestamp())
            rows.setdefault(ts, {"timestamp": datetime.datetime.utcfromtimestamp(ts).isoformat(),
                                  "cpu": None, "memory": None, "latency": None, "instances": None})
            rows[ts]["instances"] = int(point.value.int64_value)

    for series in fetch("run.googleapis.com/container/cpu/utilizations"):
        for point in series.points:
            ts = int(point.interval.end_time.timestamp())
            rows.setdefault(ts, {"timestamp": datetime.datetime.utcfromtimestamp(ts).isoformat(),
                                  "cpu": None, "memory": None, "latency": None, "instances": None})
            v = point.value.distribution_value
            rows[ts]["cpu"] = v.mean if v.count > 0 else 0.0

    for series in fetch("run.googleapis.com/container/memory/utilizations"):
        for point in series.points:
            ts = int(point.interval.end_time.timestamp())
            rows.setdefault(ts, {"timestamp": datetime.datetime.utcfromtimestamp(ts).isoformat(),
                                  "cpu": None, "memory": None, "latency": None, "instances": None})
            v = point.value.distribution_value
            rows[ts]["memory"] = v.mean if v.count > 0 else 0.0

    for series in fetch("run.googleapis.com/request_latencies"):
        for point in series.points:
            ts = int(point.interval.end_time.timestamp())
            rows.setdefault(ts, {"timestamp": datetime.datetime.utcfromtimestamp(ts).isoformat(),
                                  "cpu": None, "memory": None, "latency": None, "instances": None})
            v = point.value.distribution_value
            rows[ts]["latency"] = v.mean if v.count > 0 else 0.0

    if not rows:
        return "No data found. Is Cloud Run receiving traffic?", 200

    errors = bq_client.insert_rows_json(BQ_TABLE, list(rows.values()))
    if errors:
        return f"BigQuery insert errors: {errors}", 500

    return f"Success! {len(rows)} rows inserted into BigQuery.", 200