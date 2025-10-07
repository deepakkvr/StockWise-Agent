# config/settings.py

import os
from pathlib import Path
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "allow"
    }
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    VECTOR_DB_PATH: Path = PROJECT_ROOT / "vector_embeddings"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # API Keys (load from environment)
    MISTRAL_API_KEY: str = Field(..., env="MISTRAL_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    NEWSAPI_KEY: str = Field(..., env="NEWSAPI_KEY")
    REDDIT_CLIENT_ID: str = Field(default="", env="REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    
    # Model configurations
    DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    MISTRAL_MODEL: str = "mistral-large-latest"
    GEMINI_MODEL: str = "gemini-2.5-pro"
    
    # Data collection settings
    MAX_NEWS_ARTICLES_PER_STOCK: int = 50
    MAX_SOCIAL_POSTS_PER_STOCK: int = 100
    DATA_RETENTION_DAYS: int = 30
    UPDATE_INTERVAL_HOURS: int = 1
    
    # Stock symbols to track
    DEFAULT_STOCKS: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
        "NVDA", "META", "BRK-B", "UNH", "JNJ"
    ]
    
    # ChromaDB settings
    CHROMA_COLLECTION_NAME: str = "stock_insights"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Visualization settings
    CHART_THEME: str = "plotly_dark"
    DEFAULT_CHART_HEIGHT: int = 600
    DEFAULT_CHART_WIDTH: int = 1000
    
    # Streamlit settings
    STREAMLIT_THEME: Dict[str, Any] = {
        "primaryColor": "#FF6B6B",
        "backgroundColor": "#0E1117",
        "secondaryBackgroundColor": "#262730",
        "textColor": "#FAFAFA"
    }
    
    # Config is now replaced by model_config above

# Global settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.VECTOR_DB_PATH, settings.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)