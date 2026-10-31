# Milvus collection schema ve BM25 Function oluşturmak için gerekli sınıfları içe aktarır.
from pymilvus import DataType, Function, FunctionType

# Ortak chunking, embedding ve Milvus bağlantı fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import load_and_chunk_pdf, embed_documents, get_milvus_client

# Collection adı ve embedding boyutunu config.py dosyasından içe aktarır.
from config import COLLECTION_NAME, EMBEDDING_DIMENSION


# ============================================================
# 12.4 MILVUS INGESTION
# ============================================================

# PDF'i yükler ve character-based chunk'lara ayırır.
chunks = load_and_chunk_pdf()

# Bütün chunk'ların Qwen3 dense embedding vector'lerini oluşturur.
dense_vectors = embed_documents(chunks)

# Local Milvus istemcisini alır.
client = get_milvus_client()


# Aynı collection daha önce oluşturulduysa temiz bir ingestion için kontrol eder.
if client.has_collection(COLLECTION_NAME):

    # Mevcut collection'ı siler.
    client.drop_collection(COLLECTION_NAME)


# Auto ID kullanan ve dynamic field kabul etmeyen collection schema nesnesini oluşturur.
schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

# Milvus tarafından otomatik oluşturulacak primary key alanını ekler.
schema.add_field("id", DataType.INT64, is_primary=True)

# Chunk'ın ham metnini saklayacak ve BM25 analyzer tarafından kullanılacak text alanını ekler.
schema.add_field("text", DataType.VARCHAR, max_length=4096, enable_analyzer=True, analyzer_params={"type": "standard"})

# Chunk'ın kaynak PDF dosyasını saklayacak alanı ekler.
schema.add_field("document", DataType.VARCHAR, max_length=512)

# Chunk'ın kaynak PDF sayfasını saklayacak alanı ekler.
schema.add_field("page", DataType.INT64)

# Her chunk'ın uygulama seviyesindeki benzersiz ID bilgisini saklayacak alanı ekler.
schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)

# Qwen3 embedding vector'ünü saklayacak dense vector alanını ekler.
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)

# Milvus BM25 fonksiyonunun üreteceği sparse vector alanını ekler.
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)


# Raw text alanından otomatik BM25 sparse vector üretecek built-in Milvus fonksiyonunu oluşturur.
bm25_function = Function(name="text_bm25", input_field_names=["text"], output_field_names=["sparse_vector"], function_type=FunctionType.BM25)

# Hazırlanan BM25 fonksiyonunu collection schema içerisine ekler.
schema.add_function(bm25_function)


# Vector index ayarlarını saklayacak nesneyi oluşturur.
index_params = client.prepare_index_params()

# Dense vector alanında cosine similarity kullanacak otomatik index oluşturur.
index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")

# Sparse vector alanında BM25 inverted index oluşturur.
index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")


# Hazırlanan schema ve index ayarlarıyla Milvus collection oluşturur.
client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


# Milvus'a eklenecek kayıtları tutmak için boş liste oluşturur.
records = []


# Chunk'ları ve karşılık gelen dense embedding vector'lerini birlikte dolaşır.
for chunk, vector in zip(chunks, dense_vectors):

    # Mevcut chunk'ın text, metadata ve vector bilgisini Milvus kayıt formatında listeye ekler.
    records.append({"text": chunk.page_content, "document": chunk.metadata["document"], "page": chunk.metadata["page"], "chunk_id": chunk.metadata["chunk_id"], "dense_vector": vector.tolist()})


# Hazırlanan bütün kayıtları Milvus collection içerisine ekler.
result = client.insert(collection_name=COLLECTION_NAME, data=records)

# Insert işlemlerinin storage katmanına yazılmasını sağlar.
client.flush(collection_name=COLLECTION_NAME)


# Ingestion işleminin tamamlandığını gösterir.
print("\nMilvus ingestion tamamlandı.")

# Milvus'a eklenen toplam chunk sayısını gösterir.
print("Eklenen chunk sayısı:", result["insert_count"])

# Kullanılan collection adını gösterir.
print("Collection:", COLLECTION_NAME)