import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from ml.config import EMAIL_USER, EMAIL_PASS, RECIPIENT_EMAIL

class EmailService:
    def __init__(self):
        self.enabled = bool(EMAIL_USER and EMAIL_PASS)
        if not self.enabled:
            print("[EmailService] ⚠️ EMAIL_USER or EMAIL_PASS not set. Email alerts disabled.")

    def send_anomaly_email(self, anomaly_data: dict):
        """
        Sends an alert email when an anomaly is detected.
        anomaly_data: {
            'label': str,
            'confidence': float,
            'timestamp': str,
            'message': str
        }
        """
        if not self.enabled:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_USER
            msg['To'] = RECIPIENT_EMAIL
            msg['Subject'] = "🚨 Audio Anomaly Detected"

            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            body = f"""
An anomaly was detected by the monitoring system.

Time: {time_str}
Detection status: {anomaly_data.get('label', 'ANOMALY')}
Confidence: {anomaly_data.get('confidence', 0)*100:.1f}%
Message: {anomaly_data.get('message', 'N/A')}

Please check the dashboard immediately.
"""
            msg.attach(MIMEText(body, 'plain'))

            # For Gmail, use smtp.gmail.com and port 587 (TLS)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
            server.quit()
            
            print(f"[EmailService] ✅ Alert email sent to {RECIPIENT_EMAIL}")
            return True
        except Exception as e:
            print(f"[EmailService] ❌ Failed to send email: {e}")
            traceback.print_exc()
            return False

email_service = EmailService()
