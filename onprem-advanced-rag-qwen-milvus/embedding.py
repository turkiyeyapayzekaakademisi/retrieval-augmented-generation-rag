# PDF yükleme ve embedding fonksiyonlarını ortak rag_core.py dosyasından içe aktarır.
from rag_core import load_and_chunk_pdf, embed_documents, embed_query


# PDF'i yükler ve character-based chunk'lara ayırır.
chunks = load_and_chunk_pdf()

# Bütün doküman chunk'larının dense embedding vector'lerini oluşturur.
dense_vectors = embed_documents(chunks)


# Toplam chunk sayısını terminalde gösterir.
print("\nChunk sayısı:", len(chunks))

# Oluşturulan embedding sayısını gösterir.
print("Embedding sayısı:", dense_vectors.shape[0])

# Her embedding vector'ünün boyutunu gösterir.
print("Embedding boyutu:", dense_vectors.shape[1])

# İlk embedding vector'ünün ilk on değerini kontrol amacıyla gösterir.
print("\nİlk embedding'in ilk 10 değeri:")

# İlk vector'ün ilk on değerini terminalde yazdırır.
print(dense_vectors[0][:10])


# Query embedding davranışını test etmek için örnek soru oluşturur.
query = "KV Cache büyük dil modellerinde ne işe yarar?"

# Örnek query'yi dense embedding vector'üne dönüştürür.
query_vector = embed_query(query)

# Kullanılan sorguyu terminalde gösterir.
print("\nQuery:", query)

# Query embedding boyutunu terminalde gösterir.
print("Query embedding boyutu:", len(query_vector))