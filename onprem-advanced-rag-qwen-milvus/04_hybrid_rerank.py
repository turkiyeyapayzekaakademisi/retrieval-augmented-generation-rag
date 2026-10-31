# Query rewriting, Hybrid Search, reranking ve terminal çıktı fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import rewrite_query, hybrid_search, rerank_results, print_results


# Kullanıcıdan doğal dilde sorgu alır.
original_query = input("\nSorunuzu yazın: ")

# Kullanıcı sorgusunu retrieval için daha açık hale getirir.
rewritten_query = rewrite_query(original_query)

# Hybrid Search ile geniş aday listesini getirir.
hybrid_results = hybrid_search(rewritten_query)

# Hybrid Search sonuçlarını BGE reranker ile yeniden sıralar.
final_results = rerank_results(rewritten_query, hybrid_results)


# Query rewriting sonucunu terminalde gösterir.
print("\nRewritten Query:", rewritten_query)

# Hybrid Search sonrasında ilk beş adayın başlığını gösterir.
print("\n--- RERANKER ÖNCESİ İLK 5 ---\n")

# Hybrid Search sıralamasındaki ilk beş sonucu gösterir.
print_results(hybrid_results[:5])

# Reranking sonrası final sonuç başlığını gösterir.
print("\n--- RERANKER SONRASI TOP-K ---\n")

# BGE reranking sonrasında oluşan final sonuçları gösterir.
print_results(final_results)