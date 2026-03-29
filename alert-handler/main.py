# main.py (alert_handler.py)
import base64, json, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

GMAIL_SENDER   = "mtech.vcc.project@gmail.com"
GMAIL_APP_PASS = "dqapexerxsmaqhez"    # ← paste your app password here
ALERT_RECEIVER = "m25ai2033@iitj.ac.in"

def send_email_alert(severity, model, metric, score, timestamp):
    subject = f"[{severity}] Anomaly Detected — VCC Project"
    body = f"""
Anomaly Alert — GCP Cloud Run Monitor
======================================
Severity  : {severity}
Model     : {model}
Metric    : {metric}
Score     : {score}
Timestamp : {timestamp}

Investigate here:
https://console.cloud.google.com/monitoring
======================================
Sent automatically by your VCC anomaly detection pipeline.
    """
    msg = MIMEMultipart()
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = ALERT_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASS)
            server.sendmail(GMAIL_SENDER, ALERT_RECEIVER, msg.as_string())
        print(f"✅ Email alert sent to {ALERT_RECEIVER}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

def alert_handler(event, context):
    data      = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    severity  = data.get("severity",  "UNKNOWN")
    model     = data.get("model",     "unknown")
    metric    = data.get("metric",    "unknown")
    timestamp = data.get("timestamp", "unknown")
    score     = data.get("score",     "N/A")

    print(f"[{severity}] ANOMALY via {model}: {metric} "
          f"score={score} at {timestamp}")
    send_email_alert(severity, model, metric, score, timestamp)
