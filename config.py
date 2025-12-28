# ====================
# CONFIGURATION FILE
# ====================

from dotenv import load_dotenv
import os

load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama-3.3-70b-versatile")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")

# App Configuration
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "1000"))
DATA_PATH = "data"

# Model Configuration
EMBEDDING_DIMENSION = 4096  # Llama 3 context size

def validate_config():
    """Validate all required configurations"""
    errors = []
    
    if not NEO4J_URI:
        errors.append("NEO4J_URI not set in .env")
    if not NEO4J_PASSWORD:
        errors.append("NEO4J_PASSWORD not set in .env")
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY not set in .env")
    
    if errors:
        print("❌ CONFIGURATION ERRORS:")
        for error in errors:
            print(f"   - {error}")
        print("\n📝 Please update your .env file with correct values")
        return False
    
    print("✅ Configuration validated successfully")
    return True