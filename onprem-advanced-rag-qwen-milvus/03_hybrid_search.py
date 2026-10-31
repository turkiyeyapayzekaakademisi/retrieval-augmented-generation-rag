# Query rewriting, Hybrid Search ve terminal çıktı fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import rewrite_query, hybrid_search, print_results


# Kullanıcıdan doğal dilde sorgu alır.
original_query = input("\nSorunuzu yazın: ")

# Kullanıcının sorgusunu retrieval için yeniden yazar.
rewritten_query = rewrite_query(original_query)

# Rewritten query üzerinde Dense + BM25 Hybrid Search gerçekleştirir.
results = hybrid_search(rewritten_query)


# Kullanıcının orijinal query'sini terminalde gösterir.
print("\nOriginal Query :", original_query)

# Query rewriting sonucunu terminalde gösterir.
print("Rewritten Query:", rewritten_query)

# Hybrid Search sonuç başlığını gösterir.
print("\n--- QUERY REWRITE + HYBRID SEARCH ---\n")

# Hybrid Search sonuçlarını terminalde gösterir.
print_results(results)