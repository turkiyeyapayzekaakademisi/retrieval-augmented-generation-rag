# Conversation-aware query rewriting, Hybrid Search, reranking ve çıktı fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import rewrite_query, hybrid_search, rerank_results, print_results


# Önceki kullanıcı mesajlarını saklamak için boş conversation history listesi oluşturur.
history = []


# Kullanıcı çıkış komutu verene kadar conversation döngüsünü çalıştırır.
while True:

    # Kullanıcıdan yeni soru alır.
    query = input("\nSiz: ").strip()

    # Kullanıcının çıkış komutu yazıp yazmadığını kontrol eder.
    if query.lower() in {"exit", "quit", "çık"}:

        # Conversation döngüsünü sonlandırır.
        break


    # Önceki conversation context'i kullanarak standalone retrieval query oluşturur.
    rewritten_query = rewrite_query(query, history=history)

    # Standalone query ile Dense + BM25 Hybrid Search gerçekleştirir.
    hybrid_results = hybrid_search(rewritten_query)

    # Hybrid Search sonuçlarını BGE reranker ile yeniden sıralar.
    final_results = rerank_results(rewritten_query, hybrid_results)


    # Conversation-aware standalone query'yi terminalde gösterir.
    print("\nStandalone Query:", rewritten_query)

    # Final retrieval sonuçlarının başlığını gösterir.
    print("\n--- CONVERSATIONAL RETRIEVAL TOP-K ---\n")

    # Reranking sonucundaki final chunk'ları terminalde gösterir.
    print_results(final_results)


    # Mevcut kullanıcı sorusunu sonraki conversation turunda kullanılmak üzere history listesine ekler.
    history.append({"role": "user", "content": query})

"""
Siz: KV Cache nedir?

Standalone Query:
Büyük dil modellerinde KV Cache nedir ve ne amaçla kullanılır?

Siz: Peki GPU belleğine etkisi ne?

Standalone Query:
KV Cache kullanımının büyük dil modellerinde GPU bellek tüketimine etkisi nedir?
"""