# Qwen ve Milvus ile On-Prem Gelişmiş RAG Projesi

Bu proje, tamamen local/on-prem bileşenler kullanarak gelişmiş bir **Retrieval-Augmented Generation (RAG)** sistemi geliştirmeyi amaçlar.

Proje eğitim odaklı tasarlanmıştır. Sistem tek seferde hazır verilmez; her bölümde yeni bir özellik eklenerek `rag_core.py` dosyası adım adım geliştirilir.

## Final Mimari

```text
Conversation History
        ↓
Query Rewriting — Qwen3-4B
        ↓
Dense Embedding — Qwen3-Embedding-0.6B
        ↓
Dense Search + BM25
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
BGE Reranker
        ↓
Top-K Context
        ↓
Qwen3-4B
        ↓
Grounded Answer
```

## Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|---|---|
| PDF Loading | LangChain `PyPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embedding | `Qwen/Qwen3-Embedding-0.6B` |
| Vector Database | Milvus Standalone |
| Dense Search | Milvus + Cosine Similarity |
| Lexical Search | Milvus Built-in BM25 |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Local LLM | `Qwen/Qwen3-4B` |
| LLM Serving | vLLM OpenAI-Compatible API |
| Evaluation | Hit@3 + MRR |

---

# Proje Yapısı

```text
onprem-advanced-rag-qwen-milvus/
│
├── data/
│   └── buyuk_dil_modelleri_teknik_dokuman.pdf
│
├── config.py
├── rag_core.py
│
├── chunking.py
├── embedding.py
├── ingestion.py
│
├── test_qwen.py
├── 01_dense_search.py
├── 02_query_rewrite_dense.py
├── 03_hybrid_search.py
├── 04_hybrid_rerank.py
├── 05_conversational_rag.py
├── 06_final_rag.py
│
├── evaluation.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Eğitim Akışı

```text
12.1 Python Ortamı ve Proje Altyapısı
↓
12.2 PDF Loading + Character-Based Chunking
↓
12.3 Qwen3 Local Embedding
↓
12.4 Milvus + Data Ingestion
↓
12.5 Baseline Dense Search
↓
12.6 Query Rewriting + Dense Search
↓
12.7 Query Rewriting + Hybrid Search
↓
12.8 Hybrid Search + Reranking
↓
12.9 Conversational RAG
↓
12.10 Final On-Prem RAG Pipeline
↓
12.11 Retrieval Evaluation: Hit@3 + MRR
```

---

# 12.1 Proje Ortamının Hazırlanması

## Proje klasörü

```powershell
cd Desktop
mkdir onprem-advanced-rag-qwen-milvus
cd onprem-advanced-rag-qwen-milvus
mkdir data
```

PDF dosyası aşağıdaki konumda bulunmalıdır:

```text
data/buyuk_dil_modelleri_teknik_dokuman.pdf
```

## Virtual Environment

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## Python Paketleri

```powershell
pip install torch langchain-core langchain-community langchain-text-splitters pypdf numpy "sentence-transformers>=2.7.0" "transformers>=4.51.0" accelerate pymilvus==3.0.2 openai
```

## requirements.txt

```text
torch
langchain-core
langchain-community
langchain-text-splitters
pypdf
numpy
sentence-transformers>=2.7.0
transformers>=4.51.0
accelerate
pymilvus==3.0.2
openai
```

Daha sonra ortamı yeniden kurmak için:

```powershell
pip install -r requirements.txt
```

## .gitignore

```text
.venv/
__pycache__/
*.pyc
```

---

# config.py

Projedeki ortak ayarlar `config.py` içerisinde tutulur.

Temel ayarlar:

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "llm_technical_documents"

VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
LLM_MODEL = "Qwen/Qwen3-4B"

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

DENSE_TOP_K = 10
HYBRID_TOP_K = 20
FINAL_TOP_K = 5
```

---

# 12.2 PDF Loading ve Character-Based Chunking

PDF, `PyPDFLoader` ile yüklenir ve `RecursiveCharacterTextSplitter` ile parçalanır.

Chunking parametreleri:

```text
Chunk Size    = 1000 karakter
Chunk Overlap = 200 karakter
```

Her chunk için aşağıdaki metadata alanları oluşturulur:

```text
document
page
chunk_id
```

Bu bölümde ilk kez `rag_core.py` oluşturulur ve şu fonksiyonlar eklenir:

```python
load_and_chunk_pdf()
print_chunks()
```

`chunking.py` sadece bu ortak fonksiyonları kullanır.

Çalıştırma:

```powershell
python chunking.py
```

---

# 12.3 Qwen3 Embedding

Embedding modeli:

```text
Qwen/Qwen3-Embedding-0.6B
```

Bu model doküman ve query metinlerini dense vector'lere dönüştürür.

Bu bölümde `rag_core.py` içerisine şu fonksiyonlar eklenir:

```python
get_embedding_model()
embed_documents()
embed_query()
```

Embedding modeli yalnızca ilk ihtiyaçta yüklenir ve sonraki fonksiyon çağrılarında tekrar kullanılır.

Çalıştırma:

```powershell
python embedding.py
```

Beklenen embedding boyutu:

```text
1024
```

---

# 12.4 Milvus Standalone Kurulumu

Milvus, Docker üzerinde Standalone olarak çalıştırılır.

## WSL Kontrolü

```powershell
wsl --version
```

Kurulu değilse:

```powershell
wsl --install
wsl --update
```

## Docker Kontrolü

Docker Desktop içerisinde WSL2 backend aktif olmalıdır.

```powershell
docker info --format "{{.OSType}}"
docker compose version
```

Beklenen:

```text
linux
```

## Milvus Standalone

```powershell
cd C:\
mkdir milvus-standalone
cd milvus-standalone
```

Milvus resmi Standalone Compose dosyası indirilir ve servisler başlatılır:

```powershell
Invoke-WebRequest -Uri "https://github.com/milvus-io/milvus/releases/download/v3.0.3/milvus-standalone-docker-compose.yaml" -OutFile docker-compose.yml

docker compose up -d
docker compose ps
```

Milvus:

```text
http://localhost:19530
```

Web UI:

```text
http://127.0.0.1:9091/webui/
```

Bu bölümde `rag_core.py` içerisine:

```python
get_milvus_client()
```

eklenir.

---

# Milvus Collection Yapısı

Collection içerisinde aşağıdaki alanlar bulunur:

```text
id
text
document
page
chunk_id
dense_vector
sparse_vector
```

Dense vector:

```text
Qwen3-Embedding-0.6B
↓
dense_vector → FLOAT_VECTOR(1024)
```

Sparse vector:

```text
text
↓
Milvus BM25 Function
↓
sparse_vector
```

Index yapısı:

```text
dense_vector
→ AUTOINDEX
→ COSINE

sparse_vector
→ SPARSE_INVERTED_INDEX
→ BM25
```

---

# ingestion.py

`ingestion.py` aşağıdaki pipeline'ı çalıştırır:

```text
PDF
↓
Character-Based Chunking
↓
Qwen3 Embedding
↓
Milvus Collection
↓
Dense Vector + BM25 Sparse Vector
```

Çalıştırma:

```powershell
python ingestion.py
```

Collection schema'nın `ingestion.py` içerisinde açıkça tutulmasının amacı, eğitim sırasında primary key, text field, dense vector, sparse vector, BM25 Function ve index yapılarının görülebilmesidir.

---

# 12.5 Baseline Dense Semantic Search

İlk retrieval sistemi yalnızca dense semantic search kullanır.

```text
Query
↓
Qwen3 Embedding
↓
Milvus Dense Search
↓
Top-K
```

Henüz aşağıdaki özellikler kullanılmaz:

```text
Query Rewriting
BM25
Hybrid Search
Reranker
Conversation
LLM Answer Generation
```

Bu bölümde `rag_core.py` içerisine:

```python
dense_search()
print_results()
```

eklenir.

Çalıştırma:

```powershell
python 01_dense_search.py
```

Örnek sorgu:

```text
KV Cache ne işe yarar?
```

Bu sonuç sonraki yöntemleri karşılaştırmak için baseline olarak kullanılır.

---

# 12.6 Query Rewriting + Dense Search

Bu bölümde kullanıcı query'si retrieval'dan önce Qwen3-4B tarafından yeniden yazılır.

```text
Original Query
↓
Qwen3-4B
↓
Rewritten Query
↓
Qwen3 Embedding
↓
Dense Search
```

## vLLM + Qwen3-4B

GPU kontrolü:

```powershell
nvidia-smi
```

Qwen3-4B, vLLM üzerinde OpenAI-compatible API olarak çalıştırılır:

```powershell
docker run -d --name qwen3-vllm --runtime nvidia --gpus all --ipc=host -p 8000:8000 -v hf-cache:/root/.cache/huggingface vllm/vllm-openai:latest --model Qwen/Qwen3-4B --max-model-len 8192 --gpu-memory-utilization 0.85
```

Log:

```powershell
docker logs -f qwen3-vllm
```

API:

```text
http://localhost:8000/v1
```

Container'ı durdurmak:

```powershell
docker stop qwen3-vllm
```

Tekrar başlatmak:

```powershell
docker start qwen3-vllm
```

Bu bölümde `rag_core.py` içerisine:

```python
get_llm_client()
chat_completion()
rewrite_query()
```

eklenir.

Local Qwen bağlantısını test etmek için:

```powershell
python test_qwen.py
```

Query Rewrite + Dense Search:

```powershell
python 02_query_rewrite_dense.py
```

Pipeline:

```text
rewrite_query()
↓
dense_search()
```

---

# 12.7 Hybrid Search

Dense Semantic Search ile BM25 birlikte kullanılır.

```text
Rewritten Query
       ↓
 ┌─────┴─────┐
 ↓           ↓
Dense       BM25
Search      Search
 ↓           ↓
 └─────┬─────┘
       ↓
      RRF
       ↓
Hybrid Search
```

Dense Search anlamsal benzerliği yakalarken BM25 özellikle teknik terimlerin birebir eşleşmesinde faydalıdır.

Örnek:

```text
KV Cache
vLLM
RoPE
BM25
RAG
```

Bu bölümde `rag_core.py` içerisine:

```python
hybrid_search()
```

eklenir.

Çalıştırma:

```powershell
python 03_hybrid_search.py
```

---

# 12.8 Hybrid Search + Reranking

Hybrid Search ilk adayları hızlı şekilde getirir.

```text
Hybrid Search
↓
Top 20
↓
BGE Reranker
↓
Top 5
```

Reranker modeli:

```text
BAAI/bge-reranker-v2-m3
```

Embedding modeli query ve document'ı ayrı vector'lere dönüştürürken reranker query ve passage metinlerini birlikte değerlendirir.

Bu bölümde `rag_core.py` içerisine:

```python
get_reranker()
rerank_results()
```

eklenir.

Çalıştırma:

```powershell
python 04_hybrid_rerank.py
```

Pipeline:

```text
rewrite_query()
↓
hybrid_search()
↓
rerank_results()
```

---

# 12.9 Conversational RAG

Bu bölümde takip soruları desteklenir.

Örnek:

```text
User:
KV Cache nedir?

User:
Peki GPU belleğine etkisi ne?
```

İkinci soru tek başına retrieval için yeterli değildir.

Qwen3-4B bunu conversation context kullanarak standalone query'ye dönüştürür:

```text
KV Cache kullanımının büyük dil modellerinde GPU bellek tüketimine etkisi nedir?
```

Bu bölümde mevcut:

```python
rewrite_query()
```

fonksiyonu geliştirilir.

Yeni kullanım:

```python
rewrite_query(query, history=None, summary="")
```

Eski kullanım çalışmaya devam eder:

```python
rewrite_query(query)
```

Conversation-aware kullanım:

```python
rewrite_query(query, history=history)
```

Çalıştırma:

```powershell
python 05_conversational_rag.py
```

Bu bölüm yalnızca conversation-aware retrieval'a odaklanır; henüz final cevap üretimi yapılmaz.

---

# 12.10 Final On-Prem RAG Pipeline

Bu bölümde retrieval ile answer generation birleştirilir.

```text
Conversation History
        ↓
Query Rewriting
        ↓
Qwen3 Embedding
        ↓
Dense Search + BM25
        ↓
RRF
        ↓
BGE Reranker
        ↓
Top-K
        ↓
Context Building
        ↓
Qwen3-4B
        ↓
Grounded Answer
        ↓
Conversation Summary
```

Bu bölümde `rag_core.py` içerisine:

```python
build_context()
generate_answer()
create_summary()
```

eklenir.

Çalıştırma:

```powershell
python 06_final_rag.py
```

Final RAG sisteminin temel çağrıları:

```python
rewritten_query = rewrite_query(query, history=history, summary=conversation_summary)

hybrid_results = hybrid_search(rewritten_query)

final_results = rerank_results(rewritten_query, hybrid_results)

context = build_context(final_results)

answer = generate_answer(query, context, history=history)

conversation_summary = create_summary(conversation_summary, query, answer)
```

---

# Grounded Answer Mantığı

Final LLM cevabı yalnızca retrieved context içerisindeki bilgiler kullanılarak oluşturulur.

```text
Context içinde bilgi varsa:
→ Kaynaklara dayalı cevap üret.

Context içinde bilgi yoksa:
→ Bilginin mevcut kaynaklarda bulunamadığını söyle.

Context dışında bilgi üretme.
```

Retrieved context örneği:

```text
[KAYNAK 1 | Sayfa 5 | chunk_18]
...

[KAYNAK 2 | Sayfa 6 | chunk_19]
...
```

---

# Conversation Memory

Sistem iki farklı conversation bilgisi kullanır.

## Recent History

Son kullanıcı ve asistan mesajları:

```text
user: ...
assistant: ...
```

## Conversation Summary

Uzayan sohbetin kısa teknik özeti:

```text
Conversation Recap
```

Bu yapı sayesinde conversation context korunurken prompt'un sürekli büyümesi azaltılır.

---

# 12.11 Retrieval Evaluation

Son bölümde retrieval tekniklerinin gerçekten kaliteyi artırıp artırmadığı ölçülür.

Karşılaştırılan sistemler:

```text
Dense

Rewrite + Dense

Rewrite + Hybrid

Rewrite + Hybrid + Reranker
```

## Ground Truth

Evaluation sırasında doğru sonuçları yalnızca anahtar kelime eşleşmesiyle belirlemek yerine manuel olarak ilgili chunk'lar etiketlenir.

Örnek:

```python
test_set = [
    {
        "query": "KV Cache ne işe yarar?",
        "relevant_chunk_ids": ["chunk_18", "chunk_19"]
    }
]
```

`chunk_XX` değerleri `chunking.py` çıktısı incelenerek belirlenmelidir.

## Hit@3

**Hit@3**, doğru/relevant chunk'ın getirilen ilk 3 sonuç içerisinde bulunup bulunmadığını ölçer.

```text
İlk 3 içerisinde relevant chunk varsa → 1
Yoksa → 0
```

Örnek:

```text
1. chunk_5
2. chunk_18 ← Relevant
3. chunk_9

Hit@3 = 1
```

## MRR

**MRR — Mean Reciprocal Rank**, ilk doğru sonucun sıralamada ne kadar yukarıda olduğunu ölçer.

```text
1. sıradaysa → 1.00
2. sıradaysa → 0.50
3. sıradaysa → 0.33
4. sıradaysa → 0.25
```

Formül:

```text
Reciprocal Rank = 1 / İlk relevant sonucun sırası
```

Birden fazla query için ortalaması alınarak MRR elde edilir.

## Evaluation Çalıştırma

```powershell
python evaluation.py
```

Örnek çıktı:

```text
Dense                          Hit@3: 0.60 | MRR: 0.47
Rewrite + Dense                Hit@3: 0.80 | MRR: 0.64
Rewrite + Hybrid               Hit@3: 1.00 | MRR: 0.78
Rewrite + Hybrid + Rerank      Hit@3: 1.00 | MRR: 0.91
```

Bu değerler örnektir. Gerçek skorlar kullanılan PDF, test soruları ve manuel etiketlenen relevant chunk'lara göre değişir.

---

# rag_core.py Nasıl Gelişiyor?

```text
12.2
load_and_chunk_pdf()
print_chunks()

        ↓

12.3
get_embedding_model()
embed_documents()
embed_query()

        ↓

12.4
get_milvus_client()

        ↓

12.5
dense_search()
print_results()

        ↓

12.6
get_llm_client()
chat_completion()
rewrite_query()

        ↓

12.7
hybrid_search()

        ↓

12.8
get_reranker()
rerank_results()

        ↓

12.9
rewrite_query()
→ history + summary desteği

        ↓

12.10
build_context()
generate_answer()
create_summary()

        ↓

12.11
Mevcut fonksiyonlar evaluation.py içerisinde tekrar kullanılır.
```

Böylece aynı retrieval kodları farklı dosyalarda tekrar tekrar yazılmaz.

---

# Eğitim Scriptlerinin Görevi

## 01_dense_search.py

```text
dense_search()
```

## 02_query_rewrite_dense.py

```text
rewrite_query()
↓
dense_search()
```

## 03_hybrid_search.py

```text
rewrite_query()
↓
hybrid_search()
```

## 04_hybrid_rerank.py

```text
rewrite_query()
↓
hybrid_search()
↓
rerank_results()
```

## 05_conversational_rag.py

```text
history
↓
rewrite_query()
↓
hybrid_search()
↓
rerank_results()
```

## 06_final_rag.py

```text
history + summary
↓
rewrite_query()
↓
hybrid_search()
↓
rerank_results()
↓
build_context()
↓
generate_answer()
↓
create_summary()
```

---

# Servisleri Başlatma

Projeyi tekrar çalıştırmak için önce Milvus ve Qwen servislerinin aktif olduğundan emin olun.

## Milvus

```powershell
cd C:\milvus-standalone
docker compose up -d
```

## Qwen3-4B / vLLM

```powershell
docker start qwen3-vllm
```

Kontrol:

```powershell
docker ps
```

---

# Servisleri Durdurma

Milvus:

```powershell
cd C:\milvus-standalone
docker compose stop
```

Qwen:

```powershell
docker stop qwen3-vllm
```

---

# Milvus'u Eğitim Öncesi Tamamen Sıfırlama

Milvus verilerini ve collection'ları tamamen temizlemek için:

```powershell
cd C:\milvus-standalone

docker compose down --remove-orphans

Remove-Item -Recurse -Force .\volumes
```

Milvus kurulum klasörünü de kaldırmak için:

```powershell
cd C:\
Remove-Item -Recurse -Force C:\milvus-standalone
```

Qwen vLLM container'ını sıfırlamak için:

```powershell
docker stop qwen3-vllm
docker rm qwen3-vllm
```

Model cache'lerini silmek zorunlu değildir.

---

# Önerilen Çalıştırma Sırası

```text
1. python chunking.py

2. python embedding.py

3. Milvus'u başlat

4. python ingestion.py

5. python 01_dense_search.py

6. vLLM + Qwen3-4B'yi başlat

7. python test_qwen.py

8. python 02_query_rewrite_dense.py

9. python 03_hybrid_search.py

10. python 04_hybrid_rerank.py

11. python 05_conversational_rag.py

12. python 06_final_rag.py

13. Relevant chunk ID'lerini manuel etiketle

14. python evaluation.py
```

---

# Final Sistem

```text
PDF
↓
Character-Based Chunking
↓
Qwen3-Embedding-0.6B
↓
Milvus

        ┌───────────────┐
        │               │
        ↓               ↓
   Dense Search       BM25
        │               │
        └───────┬───────┘
                ↓
               RRF
                ↓
        BGE Reranker
                ↓
              Top-K
                ↓
          Final Context
                ↓
          Qwen3-4B/vLLM
                ↓
         Grounded Answer
```

---

# Projenin Kazanımları

Bu projeyi tamamlayan kişi:

- PDF dokümanlarını RAG için hazırlayabilir.
- Character-based chunking uygulayabilir.
- Local embedding modeli çalıştırabilir.
- Milvus üzerinde vector collection oluşturabilir.
- Dense Semantic Search geliştirebilir.
- Query Rewriting uygulayabilir.
- Dense Search ve BM25'i Hybrid Search olarak birleştirebilir.
- Reciprocal Rank Fusion kullanabilir.
- CrossEncoder reranker ile retrieval sonuçlarını yeniden sıralayabilir.
- Conversational RAG geliştirebilir.
- Local Qwen modelini vLLM ile servis edebilir.
- Kaynaklara dayalı grounded RAG cevapları üretebilir.
- Hit@K ve MRR ile retrieval sistemlerini karşılaştırabilir.

---

# Kullanılan Ana Modeller

```text
Embedding:
Qwen/Qwen3-Embedding-0.6B

LLM:
Qwen/Qwen3-4B

Reranker:
BAAI/bge-reranker-v2-m3
```

# Ana Altyapı

```text
Vector Database:
Milvus Standalone

LLM Serving:
vLLM

Container:
Docker

Python:
Sentence Transformers
LangChain
PyMilvus
OpenAI Python SDK
```
