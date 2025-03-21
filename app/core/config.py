import os
from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = os.getenv("APP_NAME", "CallCenterSystem")
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", 5))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", 10))
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # LLM Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4")
    
    # gRPC Services
    GRPC_HOST: str = os.getenv("GRPC_HOST", "0.0.0.0")
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", 50051))
    
    # Caching
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    
    # Elasticsearch
    ELASTIC_HOST: str = os.getenv("ELASTIC_HOST", "localhost")
    ELASTIC_PORT: int = int(os.getenv("ELASTIC_PORT", 9200))
    ELASTIC_USERNAME: str = os.getenv("ELASTIC_USERNAME", "elastic")
    ELASTIC_PASSWORD: str = os.getenv("ELASTIC_PASSWORD", "changeme")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings() 