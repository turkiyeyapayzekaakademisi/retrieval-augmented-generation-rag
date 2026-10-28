import json

import faiss
import numpy as np
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    LLM_MODEL,
    FETCH_K,
    TOP_K,
    SIMILARITY_THRESHOLD
)


# ============================================================
# MODEL VE VECTOR STORE YÜKLEME
# ============================================================

# Ingestion sırasında kullandığımız BGE-M3 modelinin aynısını yüklüyoruz.
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# FAISS index'i yüklüyoruz.
faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

# Chunk metinlerini ve metadata bilgilerini yüklüyoruz.
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks_data = json.load(file)

# JSON kayıtlarını tekrar LangChain Document nesnelerine dönüştürüyoruz.
chunks = []

for item in chunks_data:
    chunks.append(
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
    )

print(f"\nFAISS yüklendi. Vector sayısı: {faiss_index.ntotal}")


# ============================================================
# KULLANICI SORUSU
# ============================================================

query = input("\nSorunuzu yazın: ")


# ============================================================
# 7.6 KULLANICI SORUSUNU EMBEDDING'E ÇEVİRME
# ============================================================

print("\n=== 7.6 KULLANICI SORUSUNU EMBEDDING'E ÇEVİRME ===\n")

# Kullanıcı sorusunu BGE-M3 ile embedding'e dönüştürüyoruz.
query_vector = embedding_model.encode([query], normalize_embeddings=True)

# FAISS float32 formatı beklediği için dönüştürüyoruz.
query_vector = np.asarray(query_vector, dtype="float32")

print(f"Soru             : {query}")
print(f"Embedding boyutu : {query_vector.shape[1]}")
print("\nİlk 10 embedding değeri:")
print(query_vector[0][:10])


# ============================================================
# 7.7 SEMANTIC SEARCH
# ============================================================

print("\n=== 7.7 SEMANTIC SEARCH ===\n")

# Semantic search bütün chunk'lar üzerinde yapılır.
# Metadata filtering uygulanmaz.
# FAISS soruya anlamsal olarak en yakın chunk'ları bulur.
fetch_k = min(FETCH_K, faiss_index.ntotal)

scores, ids = faiss_index.search(query_vector, fetch_k)

results = []

for score, chunk_id in zip(scores[0], ids[0]):
    if chunk_id == -1:
        continue

    chunk = chunks[chunk_id]

    results.append({
        "document": chunk,
        "similarity": float(score)
    })


# İlk semantic search sonuçlarını terminalde gösteriyoruz.
print("--- SEMANTIC SEARCH SONUÇLARI ---\n")

for result in results[:8]:
    document = result["document"]

    print(f"Similarity : {result['similarity']:.4f}")
    print(f"Doküman    : {document.metadata['document']}")
    print(f"Kategori   : {document.metadata['category']}")
    print(f"Bölüm      : {document.metadata['section']} - {document.metadata['section_title']}")
    print(f"Metin      : {document.page_content[:150]}...")
    print("-" * 70)


# ============================================================
# 7.8 TOP-K VE SIMILARITY THRESHOLD
# ============================================================

print("\n=== 7.8 TOP-K VE SIMILARITY THRESHOLD ===\n")

# Threshold altında kalan chunk'ları eliyoruz.
threshold_results = []

for result in results:
    if result["similarity"] >= SIMILARITY_THRESHOLD:
        threshold_results.append(result)

# Sonuçları similarity değerine göre sıralıyoruz.
threshold_results.sort(key=lambda result: result["similarity"], reverse=True)

# En iyi Top-K chunk'ı seçiyoruz.
final_results = threshold_results[:TOP_K]

print(f"Similarity Threshold : {SIMILARITY_THRESHOLD}")
print(f"Top-K                : {TOP_K}")

print("\n--- THRESHOLD KONTROLÜ ---\n")

for result in results[:8]:
    document = result["document"]
    status = "KALDI" if result["similarity"] >= SIMILARITY_THRESHOLD else "ELENDİ"

    print(f"{result['similarity']:.4f} | {status:<6} | {document.metadata['document']} | {document.metadata['section_title']}")

print("\n--- FINAL TOP-K ---\n")

for result in final_results:
    document = result["document"]
    print(f"{result['similarity']:.4f} | {document.metadata['document']} | {document.metadata['section']} - {document.metadata['section_title']}")


# ============================================================
# 7.9 LLM'E BESLEME VE CEVAP ÜRETME
# ============================================================

print("\n=== 7.9 LLM'E BESLEME VE CEVAP ÜRETME ===\n")

# Final Top-K chunk'ları Gemini'ye gönderilecek context haline getiriyoruz.
context_parts = []

for index, result in enumerate(final_results, start=1):
    document = result["document"]
    metadata = document.metadata

    context_parts.append(
        f"""
[KAYNAK {index}]

Doküman: {metadata['document']}
Kategori: {metadata['category']}
Bölüm: {metadata['section']} - {metadata['section_title']}
Sayfa: {metadata['page']}

Metin:
{document.page_content}
""".strip()
    )

context = "\n\n".join(context_parts)


# Gemini'ye gönderilecek gerçek context'i terminalde gösteriyoruz.
print("--- LLM'E GİDEN CONTEXT ---\n")

if context:
    print(context)
else:
    print("Context oluşturulamadı.")


# LangChain PromptTemplate ile RAG prompt'unu hazırlıyoruz.
prompt_template = PromptTemplate.from_template(
    """
Sen kurumsal dokümanlar üzerinde çalışan bir RAG asistanısın.

Kurallar:
- Sadece verilen CONTEXT içerisindeki bilgileri kullan.
- Context içerisinde olmayan bilgi üretme.
- Tahmin yapma.
- Cevabı Türkçe ver.
- Cevabı açık ve kısa tut.
- Cevabın sonunda kullandığın doküman ve bölüm bilgisini belirt.
- Context soruyu cevaplamak için yeterli değilse "Bu bilgi mevcut kaynaklarda bulunamadı." de.

KULLANICI SORUSU:
{question}

CONTEXT:
{context}
"""
)

# Kullanıcı sorusu ve retrieved context'i prompt'a ekliyoruz.
prompt = prompt_template.format(question=query, context=context)


# ============================================================
# GEMINI FINAL ANSWER
# ============================================================

if final_results:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY .env dosyasında bulunamadı.")

    # Gemini yalnızca final cevap üretiminde kullanılır.
    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )

    print("\n--- GEMINI CEVABI ---\n")
    print(response.text)

else:
    print("\n--- GEMINI CEVABI ---\n")
    print("Bu soruyu cevaplamak için yeterli kaynak bulunamadı.")