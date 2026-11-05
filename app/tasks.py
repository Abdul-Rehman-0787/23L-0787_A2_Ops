import smtplib
from email.message import EmailMessage

from celery import Celery

from .config import Config

celeryApp = Celery("visionguard", broker=Config.CELERY_BROKER_URL)
celeryApp.conf.broker_connection_retry_on_startup = True


@celeryApp.task(name="send_email", autoretry_for=(OSError,), retry_backoff=5, max_retries=3)
def sendEmail(toAddress, subject, bodyText):
    message = EmailMessage()
    message["From"] = Config.MAIL_FROM
    message["To"] = toAddress
    message["Subject"] = subject
    message.set_content(bodyText)
    with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=20) as server:
        server.send_message(message)