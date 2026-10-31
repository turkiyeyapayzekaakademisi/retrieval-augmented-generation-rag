# Dosya ve klasör yollarını platformdan bağımsız yönetmek için Path sınıfını içe aktarır.
from pathlib import Path


# ============================================================
# PROJE KLASÖRLERİ
# ============================================================

# config.py dosyasının bulunduğu ana proje klasörünü belirler.
BASE_DIR = Path(__file__).resolve().parent

# PDF dokümanlarının tutulduğu data klasörünü belirler.
DATA_DIR = BASE_DIR / "data"

# RAG sisteminde kullanılacak PDF dokümanının tam yolunu belirler.
PDF_PATH = DATA_DIR / "buyuk_dil_modelleri_teknik_dokuman.pdf"


# ============================================================
# CHUNKING AYARLARI
# ============================================================

# Her chunk'ın hedef maksimum karakter sayısını belirler.
CHUNK_SIZE = 1000

# Ardışık chunk'ların paylaşacağı karakter miktarını belirler.
CHUNK_OVERLAP = 200


# ============================================================
# EMBEDDING AYARLARI
# ============================================================

# Local embedding modelini belirler.
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Qwen3-Embedding-0.6B modelinin vector boyutunu belirler.
EMBEDDING_DIMENSION = 1024


# ============================================================
# MILVUS AYARLARI
# ============================================================

# Local Milvus sunucusunun bağlantı adresini belirler.
MILVUS_URI = "http://localhost:19530"

# RAG dokümanlarının tutulacağı Milvus collection adını belirler.
COLLECTION_NAME = "llm_technical_documents"


# ============================================================
# LOCAL LLM AYARLARI
# ============================================================

# vLLM tarafından sunulacak OpenAI-compatible API adresini belirler.
VLLM_BASE_URL = "http://localhost:8000/v1"

# Local vLLM gerçek API anahtarı gerektirmediği için örnek bir değer belirler.
VLLM_API_KEY = "EMPTY"

# vLLM üzerinde çalıştırılacak Qwen modelini belirler.
LLM_MODEL = "Qwen/Qwen3-4B"


# ============================================================
# RERANKER AYARLARI
# ============================================================

# Hybrid Search sonuçlarını yeniden sıralamak için kullanılacak modeli belirler.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


# ============================================================
# RETRIEVAL AYARLARI
# ============================================================

# Baseline Dense Search sırasında getirilecek chunk sayısını belirler.
DENSE_TOP_K = 10

# Hybrid Search sırasında getirilecek aday chunk sayısını belirler.
HYBRID_TOP_K = 20

# Reranking sonrasında kullanılacak final chunk sayısını belirler.
FINAL_TOP_K = 5