# Query rewriting, Dense Search ve terminal çıktı fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import rewrite_query, dense_search, print_results


# Kullanıcıdan doğal dilde sorgu alır.
original_query = input("\nSorunuzu yazın: ")

# Kullanıcı sorgusunu retrieval için daha açık hale getirir.
rewritten_query = rewrite_query(original_query)

# Rewritten query ile Dense Semantic Search gerçekleştirir.
results = dense_search(rewritten_query)


# Kullanıcının orijinal sorgusunu terminalde gösterir.
print("\nOriginal Query :", original_query)

# Qwen tarafından yeniden yazılan sorguyu terminalde gösterir.
print("Rewritten Query:", rewritten_query)

# Retrieval sonuçlarının başlığını gösterir.
print("\n--- QUERY REWRITE + DENSE SEARCH ---\n")

# Dense Search sonuçlarını terminalde gösterir.
print_results(results)