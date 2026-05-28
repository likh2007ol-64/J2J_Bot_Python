import os
from dotenv import load_dotenv

load_dotenv()

VK_TOKEN = os.environ.get("VK_TOKEN", "")
VK_GROUP_ID = int(os.environ.get("VK_GROUP_ID", "0"))
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
ADMIN_IDS = [int(x.strip()) for x in os.environ.get("ADMIN_IDS", "0").split(",") if x.strip()]

SHARED_DIR = os.environ.get("SHARED_DIR", "/app/shared")
KNOWLEDGE_ROOT = os.environ.get("KNOWLEDGE_ROOT", os.path.join(SHARED_DIR, "Литература"))

# Fallback to local folder if running on Replit
if not os.path.exists(KNOWLEDGE_ROOT):
    KNOWLEDGE_ROOT = os.path.join(os.path.dirname(__file__), "knowledge")

CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(SHARED_DIR, "chroma_db"))
if not os.path.exists(os.path.dirname(CHROMA_DIR)):
    CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")

DB_PATH = os.environ.get("DB_PATH", os.path.join(SHARED_DIR, "bot_data", "j2j_bot.db"))
if not os.path.exists(os.path.dirname(DB_PATH)):
    DB_PATH = os.path.join(os.path.dirname(__file__), "data", "j2j_bot.db")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# JDoodle (note: env vars intentionally spelled "JDODDLE" per user setup)
JDOODLE_CLIENT_ID = os.environ.get("JDODDLE_CLIENT_ID", "")
JDOODLE_CLIENT_SECRET = os.environ.get("JDODDLE_CLIENT_SECRET", "")
JDOODLE_PLAN = os.environ.get("JDODDLE_PLAN", "free").lower().strip()
JDOODLE_FREE_LIMIT = 20
JDOODLE_ENABLED = bool(JDOODLE_CLIENT_ID and JDOODLE_CLIENT_SECRET)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.2

RAG_TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

THINKING_DELAY_MIN = 1.0
THINKING_DELAY_MAX = 3.0
