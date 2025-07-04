import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "AI APP API"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # CORS settings
    CORS_ORIGINS: list = ["*"]

settings = Settings()
