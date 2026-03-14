# collect_metrics.py

# Key metrics to collect:
# run.googleapis.com/container/cpu/utilizations
# run.googleapis.com/container/memory/utilizations
# run.googleapis.com/request_latencies
# run.googleapis.com/container/instance_count

from google.cloud import monitoring_v3, bigquery
import datetime, json

monitoring_client = monitoring_v3.MetricServiceClient()
bq_client = bigquery.Client()
PROJECT_ID = "vcc-project-52"
TABLE_ID = f"{vcc-project-52}.cloud_metrics.run_metrics"

def get_latest_metric(metric_type, minutes=60):
    now = datetime.datetime.utcnow()
	interval = monitoring_v3.TimeInterval()
    interval.end_time.seconds = int(now.timestamp())
    interval.start_time.seconds = int((now - datetime.timedelta(minutes=6)).timestamp())

    results = monitoring_client.list_time_series(
        request={
            "name": f"projects/{PROJECT_ID}",
            "filter": f'metric.type="{metric_type}" AND '
					  f'resource.labels.service_name="vcc-workload"',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    )
    points = []
    for ts in results:
        for point in ts.points:
            points.append(point.value.double_value)
    return sum(points) / len(points) if points else 0.0

def collect_and_store(request=None):
    row = {
        " timestamp": datetime.datetime.utcnow().isoformat() ,
        "cpu": get_latest_metric("run.googleapis.com/container/cpu/utilizations"),
        " memory": get_latest_metric("run.googleapis.com/container/memory/utilizations"),
        " latency": get_latest_metric("run.googleapis.com/request_latencies"),
        " instances": int(get_latest_metric("run.googleapis.com/container/instance_count")),
        " label": "normal"
    }
    errors = bq_client.insert_rows_json(TABLE_ID, [row ])
    print (f" Inserted : {row}" if not errors else f"Error : {errors}")
    return "OK"
    
if __name__ == "__main__":
    collect_and_store()

