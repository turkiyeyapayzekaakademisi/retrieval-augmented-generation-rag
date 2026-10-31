# Final RAG pipeline için gerekli ortak fonksiyonları rag_core.py dosyasından içe aktarır.
from rag_core import rewrite_query, hybrid_search, rerank_results, build_context, generate_answer, create_summary


# Kullanıcı ve asistan mesajlarını saklamak için boş conversation history listesi oluşturur.
history = []

# Uzun conversation history'nin kısa özetini saklamak için başlangıçta boş değer oluşturur.
conversation_summary = ""


# Kullanıcı çıkış komutu verene kadar final RAG chatbot döngüsünü çalıştırır.
while True:

    # Kullanıcıdan yeni soru alır.
    query = input("\nSiz: ").strip()

    # Kullanıcının çıkış komutu yazıp yazmadığını kontrol eder.
    if query.lower() in {"exit", "quit", "çık"}:

        # Chatbot döngüsünü sonlandırır.
        break


    # Conversation history ve summary kullanarak standalone retrieval query oluşturur.
    rewritten_query = rewrite_query(query, history=history, summary=conversation_summary)

    # Rewritten query ile Dense + BM25 Hybrid Search gerçekleştirir.
    hybrid_results = hybrid_search(rewritten_query)

    # Hybrid Search adaylarını BGE reranker ile yeniden sıralar.
    final_results = rerank_results(rewritten_query, hybrid_results)

    # En iyi retrieved chunk'ları Qwen'e gönderilecek context formatına dönüştürür.
    context = build_context(final_results)

    # Kullanıcının orijinal sorusuna yalnızca retrieved context üzerinden cevap oluşturur.
    answer = generate_answer(query, context, history=history)


    # Retrieval için oluşturulan rewritten query'yi terminalde gösterir.
    print("\nRewritten Query:", rewritten_query)

    # Qwen3-4B tarafından oluşturulan final grounded cevabı gösterir.
    print("\nAsistan:", answer)


    # Mevcut kullanıcı sorusunu conversation history içerisine ekler.
    history.append({"role": "user", "content": query})

    # Asistanın oluşturduğu final cevabı conversation history içerisine ekler.
    history.append({"role": "assistant", "content": answer})


    # Yeni conversation turunu kullanarak kısa conversation summary bilgisini günceller.
    conversation_summary = create_summary(conversation_summary, query, answer)

    # Güncellenmiş conversation summary bilgisini terminalde gösterir.
    print("\nConversation Recap:", conversation_summary)