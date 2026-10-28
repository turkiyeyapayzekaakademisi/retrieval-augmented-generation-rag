import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config import DATA_DIR, VECTOR_STORE_DIR, FAISS_INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL


# ============================================================
# DOKÜMANLAR
# ============================================================

documents = [
    {
        "path": DATA_DIR / "insan_kaynaklari_surecleri.pdf",
        "title": "İnsan Kaynakları Süreçleri ve Çalışan Politikaları",
        "category": "İnsan Kaynakları"
    },
    {
        "path": DATA_DIR / "bilgi_guvenligi_bt_politikalari.pdf",
        "title": "Bilgi Güvenliği ve BT Kullanım Politikaları",
        "category": "Bilgi Güvenliği"
    }
]


# ============================================================
# 7.2 PDF DOKÜMANLARININ YÜKLENMESİ
# ============================================================

print("\n=== 7.2 PDF DOKÜMANLARININ YÜKLENMESİ ===\n")

loaded_documents = []

for document in documents:
    # PDF'i LangChain PyPDFLoader ile sayfa sayfa yüklüyoruz.
    loader = PyPDFLoader(str(document["path"]))
    pages = loader.load()

    loaded_documents.append({
        "path": document["path"],
        "title": document["title"],
        "category": document["category"],
        "pages": pages
    })

    print(f"Doküman      : {document['path'].name}")
    print(f"Başlık       : {document['title']}")
    print(f"Kategori     : {document['category']}")
    print(f"Sayfa sayısı : {len(pages)}")
    print(f"İlk metin    : {pages[0].page_content[:150]}...")
    print("-" * 80)


# ============================================================
# 7.3 CHUNKING VE METADATA ÜRETİMİ
# ============================================================

print("\n=== 7.3 CHUNKING VE METADATA ÜRETİMİ ===\n")


# Yakalanacak başlık örnekleri:
#
# 1. Giriş
# 1.1 Amaç
# 2. İzin Süreçleri
# 2.1 Yıllık İzin
# 2.1.1 Alt Başlık

heading_pattern = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$")

all_chunks = []


for loaded_document in loaded_documents:
    document_name = loaded_document["path"].name
    document_title = loaded_document["title"]
    category = loaded_document["category"]
    pages = loaded_document["pages"]

    current_section = None
    current_title = None
    current_text = []
    current_page = None


    # Mevcut bölümün metnini LangChain Document nesnesine dönüştürür.
    def save_current_chunk():
        if not current_section or not current_text:
            return

        text = " ".join(current_text).strip()

        if not text:
            return

        chunk_id = f"{document_name.replace('.pdf', '')}_{current_section.replace('.', '_')}"

        chunk = Document(
            page_content=text,
            metadata={
                "document": document_name,
                "document_title": document_title,
                "category": category,
                "section": current_section,
                "section_title": current_title,
                "page": current_page,
                "chunk_id": chunk_id
            }
        )

        all_chunks.append(chunk)


    # PDF sayfalarını geziyoruz.
    for page_number, page in enumerate(pages, start=1):
        for line in page.page_content.splitlines():
            line = line.strip()

            if not line:
                continue

            # PDF footer bilgilerini chunk içerisine almıyoruz.
            if line.startswith("RAG Eğitimi - Örnek Kurumsal Doküman"):
                continue

            if line.startswith("Sayfa "):
                continue

            # Satırın numaralı bir başlık olup olmadığını kontrol ediyoruz.
            match = heading_pattern.match(line)

            if match:
                # Yeni başlığa geçmeden önce önceki chunk'ı kaydediyoruz.
                save_current_chunk()

                current_section = match.group(1).rstrip(".")
                current_title = match.group(2).strip()
                current_text = []
                current_page = page_number
            else:
                # İlk numaralı başlıktan önceki ana doküman başlığı
                # ve açıklamalar chunk'a dahil edilmez.
                if current_section:
                    current_text.append(line)

    # Dokümanın son chunk'ını kaydediyoruz.
    save_current_chunk()


# ============================================================
# TÜM CHUNK'LARI TERMİNALDE GÖSTER
# ============================================================

print(f"\nToplam chunk sayısı: {len(all_chunks)}\n")

for index, chunk in enumerate(all_chunks, start=1):
    print("=" * 100)
    print(f"CHUNK {index}")
    print("=" * 100)
    print(f"Chunk ID        : {chunk.metadata['chunk_id']}")
    print(f"Doküman         : {chunk.metadata['document']}")
    print(f"Doküman Başlığı : {chunk.metadata['document_title']}")
    print(f"Kategori        : {chunk.metadata['category']}")
    print(f"Bölüm           : {chunk.metadata['section']}")
    print(f"Başlık          : {chunk.metadata['section_title']}")
    print(f"Sayfa           : {chunk.metadata['page']}")
    print(f"Metin           : {chunk.page_content}")
    print()


# ============================================================
# 7.4 EMBEDDING OLUŞTURMA
# ============================================================

print("\n=== 7.4 EMBEDDING OLUŞTURMA ===\n")


# BGE-M3 modelini local olarak yüklüyoruz.
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# Embedding'e yalnızca body text vermiyoruz.
# Bölüm başlığını da metne ekliyoruz.
#
# Örnek:
# Yıllık İzin. Yıllık izin talebi mümkün olduğunca...
texts = [
    f"{chunk.metadata['section_title']}. {chunk.page_content}"
    for chunk in all_chunks
]


# Tüm chunk'ların embedding'lerini oluşturuyoruz.
# normalize_embeddings=True ile cosine similarity için normalize ediyoruz.
embedding_matrix = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# FAISS float32 formatı bekler.
embedding_matrix = np.asarray(embedding_matrix, dtype="float32")


print(f"\nEmbedding sayısı : {embedding_matrix.shape[0]}")
print(f"Embedding boyutu : {embedding_matrix.shape[1]}")
print("\nİlk embedding'in ilk 10 değeri:")
print(embedding_matrix[0][:10])


# ============================================================
# 7.5 FAISS VECTOR INDEX OLUŞTURMA
# ============================================================

print("\n=== 7.5 FAISS VECTOR INDEX OLUŞTURMA ===\n")


dimension = embedding_matrix.shape[1]


# Embedding'ler normalize olduğu için IndexFlatIP
# cosine similarity tabanlı semantic search için kullanılabilir.
faiss_index = faiss.IndexFlatIP(dimension)


# Tüm chunk embedding'lerini FAISS'e ekliyoruz.
faiss_index.add(embedding_matrix)


# Vector store klasörünü oluşturuyoruz.
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# FAISS index'i diske kaydediyoruz.
faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))


# FAISS yalnızca vector'leri saklar.
# Chunk metinlerini ve metadata bilgilerini ayrıca JSON'a kaydediyoruz.
chunks_data = []

for chunk in all_chunks:
    chunks_data.append({
        "page_content": chunk.page_content,
        "metadata": chunk.metadata
    })


with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
    json.dump(chunks_data, file, ensure_ascii=False, indent=2)


print(f"FAISS vector sayısı : {faiss_index.ntotal}")
print(f"Embedding boyutu    : {dimension}")

print("\nOluşturulan dosyalar:")
print(FAISS_INDEX_PATH)
print(CHUNKS_PATH)

print("\nIngestion işlemi tamamlandı.")