import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
    MODEL_NAME = os.getenv("MODEL_NAME", "visionguard")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instance/visionguard.db")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
    INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local")
    API_URL = os.getenv("API_URL", "")
    API_KEY = os.getenv("API_KEY", "")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "90"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    JWT_SECRET = os.getenv("JWT_SECRET", "")
    JWT_MINUTES = int(os.getenv("JWT_MINUTES", "60"))
    OTP_SECRET = os.getenv("OTP_SECRET", "")
    SMTP_HOST = os.getenv("SMTP_HOST", "localhost")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "1025"))
    MAIL_FROM = os.getenv("MAIL_FROM", "VisionGuard <noreply@visionguard.local>")