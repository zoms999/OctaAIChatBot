from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    database_url: str
    database_url_sync: str
    openai_api_key: str = ""
    google_api_key: str
    default_llm_provider: Literal['openai', 'google'] = 'google'
    
    class Config:
        env_file = ".env"


settings = Settings()
