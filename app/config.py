import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Azure AI Search
    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
    
    # Azure OpenAI (Grok)
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # Memory
    MEMORY_FILE = os.getenv("MEMORY_FILE", "data/long_term_memory.json")
    
    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

# Also export variables directly for modules that import them
SEARCH_ENDPOINT = Config.SEARCH_ENDPOINT
SEARCH_KEY = Config.SEARCH_KEY
SEARCH_INDEX = Config.SEARCH_INDEX
OPENAI_ENDPOINT = Config.OPENAI_ENDPOINT
OPENAI_API_KEY = Config.OPENAI_API_KEY
OPENAI_DEPLOYMENT = Config.OPENAI_DEPLOYMENT
MEMORY_FILE = Config.MEMORY_FILE
API_HOST = Config.API_HOST
API_PORT = Config.API_PORT