# Evaluation sırasında kullanılacak ortak retrieval fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import dense_search, rewrite_query, hybrid_search, rerank_results


# ============================================================
# EVALUATION TEST SET
# ============================================================

# Her test sorusu için manuel olarak doğrulanmış relevant chunk ID listesini tanımlar.
test_set = [
    {"query": "KV Cache ne işe yarar?", "relevant_chunk_ids": ["chunk_18"]},
    {"query": "Quantization neden kullanılır?", "relevant_chunk_ids": ["chunk_21"]},
    {"query": "Self-attention nasıl çalışır?", "relevant_chunk_ids": ["chunk_9"]},
    {"query": "vLLM inference performansını nasıl artırır?", "relevant_chunk_ids": ["chunk_21"]},
    {"query": "Reranking ne işe yarar?", "relevant_chunk_ids": ["chunk_27"]}
]


# ============================================================
# EVALUATION METRICS
# ============================================================

# Retrieval sonuçlarından yalnızca chunk ID listesini çıkaran yardımcı fonksiyonu tanımlar.
def get_chunk_ids(results):

    # Her retrieval sonucundaki chunk_id değerini Python listesi olarak döndürür.
    return [result["chunk_id"] for result in results]


# Relevant chunk'lardan en az birinin ilk K sonuç içerisinde olup olmadığını hesaplayan fonksiyonu tanımlar.
def hit_at_k(results, relevant_chunk_ids, k=3):

    # Retrieval sonuçlarının ilk K elemanındaki chunk ID'leri alır.
    retrieved_ids = get_chunk_ids(results[:k])

    # Relevant chunk'lardan herhangi biri ilk K içerisinde bulunuyorsa 1 döndürür.
    return 1 if any(chunk_id in relevant_chunk_ids for chunk_id in retrieved_ids) else 0


# İlk relevant chunk'ın retrieval sırasına göre Reciprocal Rank skorunu hesaplayan fonksiyonu tanımlar.
def reciprocal_rank(results, relevant_chunk_ids):

    # Retrieval sonuçlarını birden başlayan sıra numarasıyla dolaşır.
    for rank, result in enumerate(results, start=1):

        # Mevcut chunk'ın relevant chunk listesinin içerisinde olup olmadığını kontrol eder.
        if result["chunk_id"] in relevant_chunk_ids:

            # İlk relevant sonuç bulunduğunda 1 / rank değerini döndürür.
            return 1 / rank

    # Hiç relevant sonuç bulunamadıysa sıfır döndürür.
    return 0


# ============================================================
# SCORE STORAGE
# ============================================================

# Karşılaştırılacak dört retrieval sisteminin metric sonuçlarını saklayacak dictionary oluşturur.
scores = {
    "Dense": {"hit": [], "mrr": []},
    "Rewrite + Dense": {"hit": [], "mrr": []},
    "Rewrite + Hybrid": {"hit": [], "mrr": []},
    "Rewrite + Hybrid + Rerank": {"hit": [], "mrr": []}
}


# ============================================================
# EVALUATION LOOP
# ============================================================

# Test setindeki bütün soruları sırasıyla dolaşır.
for test in test_set:

    # Mevcut test sorusunu değişkene alır.
    original_query = test["query"]

    # Mevcut sorunun manuel olarak etiketlenmiş relevant chunk listesini alır.
    relevant_chunk_ids = test["relevant_chunk_ids"]

    # Qwen3-4B kullanarak sorunun rewritten retrieval query'sini oluşturur.
    rewritten_query = rewrite_query(original_query)


    # Baseline olarak orijinal query üzerinde yalnızca Dense Search çalıştırır.
    dense_results = dense_search(original_query)

    # Rewritten query üzerinde Dense Search çalıştırır.
    rewrite_dense_results = dense_search(rewritten_query)

    # Rewritten query üzerinde Dense + BM25 Hybrid Search çalıştırır.
    hybrid_results = hybrid_search(rewritten_query)

    # Hybrid Search sonuçlarını BGE reranker ile yeniden sıralar.
    reranked_results = rerank_results(rewritten_query, hybrid_results)


    # Dört retrieval yaklaşımının sonuçlarını tek dictionary içerisinde toplar.
    systems = {
        "Dense": dense_results,
        "Rewrite + Dense": rewrite_dense_results,
        "Rewrite + Hybrid": hybrid_results,
        "Rewrite + Hybrid + Rerank": reranked_results
    }


    # Her retrieval sistemini ayrı ayrı değerlendirir.
    for name, results in systems.items():

        # Mevcut retrieval sistemi için Hit@3 skorunu hesaplayıp listeye ekler.
        scores[name]["hit"].append(hit_at_k(results, relevant_chunk_ids, k=3))

        # Mevcut retrieval sistemi için Reciprocal Rank skorunu hesaplayıp listeye ekler.
        scores[name]["mrr"].append(reciprocal_rank(results, relevant_chunk_ids))


# ============================================================
# FINAL RESULTS
# ============================================================

# Evaluation sonuçlarının başlığını terminalde gösterir.
print("\n--- RETRIEVAL EVALUATION ---\n")


# Bütün retrieval yöntemlerinin toplu skorlarını dolaşır.
for name, result in scores.items():

    # Mevcut yöntemin ortalama Hit@3 skorunu hesaplar.
    hit3 = sum(result["hit"]) / len(result["hit"])

    # Mevcut yöntemin Mean Reciprocal Rank skorunu hesaplar.
    mrr = sum(result["mrr"]) / len(result["mrr"])

    # Mevcut retrieval yönteminin Hit@3 ve MRR sonuçlarını terminalde gösterir.
    print(f"{name:<30} Hit@3: {hit3:.2f} | MRR: {mrr:.2f}")