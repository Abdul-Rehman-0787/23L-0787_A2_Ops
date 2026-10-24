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