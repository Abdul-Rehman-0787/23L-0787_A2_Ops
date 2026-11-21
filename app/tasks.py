import logging
import smtplib
from email.message import EmailMessage
from email.utils import parseaddr

import requests
from celery import Celery

from .config import Config

log = logging.getLogger("visionguard")

celeryApp = Celery("visionguard", broker=Config.CELERY_BROKER_URL)
celeryApp.conf.broker_connection_retry_on_startup = True
# Hosted free tier has no worker: run tasks inside the web process instead.
celeryApp.conf.task_always_eager = Config.CELERY_EAGER
celeryApp.conf.task_eager_propagates = False

BREVO_URL = "https://api.brevo.com/v3/smtp/email"


def sendViaSmtp(toAddress, subject, bodyText, htmlBody=None):
    message = EmailMessage()
    message["From"] = Config.MAIL_FROM
    message["To"] = toAddress
    message["Subject"] = subject
    message.set_content(bodyText)
    if htmlBody:
        message.add_alternative(htmlBody, subtype="html")
    with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=20) as server:
        server.send_message(message)


def sendViaBrevo(toAddress, subject, bodyText, htmlBody=None):
    """HTTPS call, because Render's free tier blocks the SMTP ports."""
    senderName, senderEmail = parseaddr(Config.MAIL_FROM)
    body = {
        "sender": {"name": senderName or "VisionGuard", "email": senderEmail},
        "to": [{"email": toAddress}],
        "subject": subject,
        "textContent": bodyText,
    }
    if htmlBody:
        body["htmlContent"] = htmlBody
    headers = {"api-key": Config.BREVO_API_KEY, "accept": "application/json"}
    response = requests.post(BREVO_URL, json=body, headers=headers, timeout=20)
    if response.status_code >= 400:
        # Eager mode hides task errors, so say clearly why Brevo refused the email.
        log.error("Brevo refused the email: HTTP %d %s", response.status_code, response.text[:300])
    response.raise_for_status()
    log.info("Email sent through Brevo")


@celeryApp.task(name="send_email", autoretry_for=(OSError,), retry_backoff=5, max_retries=3)
def sendEmail(toAddress, subject, bodyText, htmlBody=None):
    if Config.MAIL_PROVIDER == "brevo":
        sendViaBrevo(toAddress, subject, bodyText, htmlBody)
    else:
        sendViaSmtp(toAddress, subject, bodyText, htmlBody)


graphHolder = {"store": None}


def getGraphStore():
    if graphHolder["store"] is None:
        from .graph import GraphStore
        store = GraphStore(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        if store.enabled:
            store.setup()
        graphHolder["store"] = store
    return graphHolder["store"]


@celeryApp.task(name="record_graph", autoretry_for=(Exception,), retry_backoff=5, max_retries=5)
def recordGraph(analysis):
    store = getGraphStore()
    if store.enabled:
        store.recordAnalysis(analysis)