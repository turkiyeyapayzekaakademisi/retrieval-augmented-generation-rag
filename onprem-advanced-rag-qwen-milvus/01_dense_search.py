# Ortak Dense Search ve terminal çıktı fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import dense_search, print_results


# Kullanıcıdan doğal dilde retrieval sorgusu alır.
query = input("\nSorunuzu yazın: ")

# Kullanıcı sorgusu üzerinde Dense Semantic Search gerçekleştirir.
results = dense_search(query)

# Dense Search sonuçlarının başlığını terminalde gösterir.
print("\n--- DENSE SEARCH SONUÇLARI ---\n")

# Retrieval sonuçlarını standart formatta terminalde gösterir.
print_results(results)