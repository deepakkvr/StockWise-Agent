
import os
from dotenv import load_dotenv
from pathlib import Path

def load_api_keys():
    """Load API keys from environment variables"""
    # Load from .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Validate required keys
    required_keys = ["MISTRAL_API_KEY", "GEMINI_API_KEY", "NEWSAPI_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    return {
        "mistral": os.getenv("MISTRAL_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "newsapi": os.getenv("NEWSAPI_KEY"),
        "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET", "")
    }

def RunningApiKey():
    """Legacy function for backward compatibility"""
    keys = load_api_keys()
    for key, value in keys.items():
        if value:  # Only set non-empty values
            os.environ[f"{key.upper()}_API_KEY"] = value