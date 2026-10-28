from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DOSYA YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"


# ============================================================
# EMBEDDING MODEL
# ============================================================

# Türkçe ve çok dilli retrieval için local embedding modeli.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024


# ============================================================
# GOOGLE GEMINI
# ============================================================

# Gemini yalnızca final cevap üretiminde kullanılacak.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "gemini-3.1-flash-lite"


# ============================================================
# RETRIEVAL
# ============================================================

# FAISS'ten ilk aşamada getirilecek aday sayısı.
FETCH_K = 20

# Final olarak kullanılacak maksimum chunk sayısı.
TOP_K = 3

# Minimum cosine similarity değeri.
SIMILARITY_THRESHOLD = 0.6