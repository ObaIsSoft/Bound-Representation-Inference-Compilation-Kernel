import os
from enum import Enum
from dotenv import load_dotenv

# Robust .env loading
cwd_env = os.path.join(os.getcwd(), "backend", ".env")
local_env = os.path.join(os.path.dirname(__file__), "../.env")
if os.path.exists(cwd_env):
    load_dotenv(cwd_env, override=True)
elif os.path.exists(local_env):
    load_dotenv(local_env, override=True)

class RAGProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    # FUTURE: OLLAMA, DEEPSEEK

# --- CONFIGURATION ---
# Default to Gemini if not specified
ACTIVE_PROVIDER = RAGProvider(os.getenv("RAG_PROVIDER", "gemini").lower())

# Provider-Specific Defaults
CONFIG = {
    RAGProvider.GEMINI: {
        "embed_model": "models/gemini-embedding-001",
        "embed_dim": 3072, # Gemini-001 is 3072d. Text-embedding-004 is 768d.
        "vlm_model": "models/gemini-flash-latest"
    },
    RAGProvider.OPENAI: {
        "embed_model": "text-embedding-3-small",
        "embed_dim": 1536,
        "vlm_model": "gpt-4o-mini"
    }
}

# Active Settings
current_config = CONFIG[ACTIVE_PROVIDER]
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", current_config["embed_model"])
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", current_config["embed_dim"]))
VLM_MODEL = os.getenv("RAG_VLM_MODEL", current_config["vlm_model"])

print(f"🔧 RAG Config: Provider={ACTIVE_PROVIDER.value}, Embeddng={EMBEDDING_MODEL} ({EMBEDDING_DIM}d), VLM={VLM_MODEL}")
