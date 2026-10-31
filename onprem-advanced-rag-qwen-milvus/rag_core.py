# PDF dosyasını LangChain Document nesnelerine dönüştürmek için PyPDFLoader sınıfını içe aktarır.
from langchain_community.document_loaders import PyPDFLoader

# Metni character-based chunk'lara ayırmak için RecursiveCharacterTextSplitter sınıfını içe aktarır.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding vector'lerini NumPy formatında yönetmek için NumPy kütüphanesini içe aktarır.
import numpy as np

# Qwen embedding modelini kullanmak için SentenceTransformer sınıfını içe aktarır.
from sentence_transformers import SentenceTransformer, CrossEncoder

# Local Milvus sunucusuna bağlanmak için MilvusClient sınıfını içe aktarır.
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

# Local Qwen3-4B modeline OpenAI-compatible API üzerinden bağlanmak için OpenAI sınıfını içe aktarır.
from openai import OpenAI

# PDF yolu ve chunking ayarlarını config.py dosyasından içe aktarır.
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, MILVUS_URI, COLLECTION_NAME, VLLM_BASE_URL, VLLM_API_KEY, LLM_MODEL, RERANK_MODEL, FINAL_TOP_K

# ============================================================
# PDF VE CHUNKING
# ============================================================

# PDF dokümanını yükleyip character-based chunk'lara ayıran ortak fonksiyonu tanımlar.
def load_and_chunk_pdf():

    # PDF dosyasını okuyacak loader nesnesini oluşturur.
    loader = PyPDFLoader(str(PDF_PATH))

    # PDF içerisindeki bütün sayfaları LangChain Document nesneleri olarak yükler.
    pages = loader.load()

    # Character-based chunking yapacak splitter nesnesini oluşturur.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len, separators=["\n\n", "\n", ". ", " ", ""])

    # PDF içerisindeki bütün sayfaları chunk'lara ayırır.
    chunks = text_splitter.split_documents(pages)

    # Oluşturulan her chunk üzerinde dolaşır.
    for index, chunk in enumerate(chunks):

        # Chunk'ın kaynak PDF dosyasını metadata içerisine ekler.
        chunk.metadata["document"] = PDF_PATH.name

        # PyPDFLoader'ın sıfırdan başlayan sayfa numarasını birden başlayan hale dönüştürür.
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

        # Her chunk için benzersiz bir ID oluşturur.
        chunk.metadata["chunk_id"] = f"chunk_{index + 1}"

    # Hazırlanan chunk listesini fonksiyon sonucu olarak döndürür.
    return chunks


# Chunk'ların içeriğini eğitim sırasında terminalde incelemek için kullanılacak fonksiyonu tanımlar.
def print_chunks(chunks):

    # Toplam chunk sayısını terminalde gösterir.
    print(f"\nToplam chunk sayısı: {len(chunks)}\n")

    # Bütün chunk'ları sırasıyla dolaşır.
    for index, chunk in enumerate(chunks, start=1):

        # Chunk'lar arasında görsel ayırıcı oluşturur.
        print("=" * 100)

        # Chunk'ın sıra numarasını gösterir.
        print(f"CHUNK {index}")

        # Chunk ID bilgisini gösterir.
        print(f"Chunk ID : {chunk.metadata['chunk_id']}")

        # Chunk'ın kaynak sayfa numarasını gösterir.
        print(f"Sayfa    : {chunk.metadata['page']}")

        # Chunk içerisindeki toplam karakter sayısını gösterir.
        print(f"Karakter : {len(chunk.page_content)}")

        # Chunk'ın gerçek metin içeriğini gösterir.
        print(f"Metin    : {chunk.page_content}")

        # Bir sonraki chunk öncesinde boş satır bırakır.
        print()

# ============================================================
# EMBEDDING
# ============================================================

# Embedding modelinin yalnızca ilk ihtiyaçta yüklenebilmesi için başlangıç değerini None yapar.
_embedding_model = None

# BGE reranker modelinin yalnızca ilk ihtiyaçta yüklenmesi için başlangıç değerini None yapar.
_reranker_model = None

# Qwen embedding modelini ilk ihtiyaç olduğunda yükleyen fonksiyonu tanımlar.
def get_embedding_model():

    # Global embedding model değişkenini fonksiyon içerisinde kullanacağımızı belirtir.
    global _embedding_model

    # Model henüz belleğe yüklenmediyse yükleme işlemini gerçekleştirir.
    if _embedding_model is None:

        # Qwen3 embedding modelini Hugging Face üzerinden indirir ve local belleğe yükler.
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Hazır embedding modelini döndürür.
    return _embedding_model


# Doküman chunk'larını embedding vector'lerine dönüştüren fonksiyonu tanımlar.
def embed_documents(chunks):

    # Local embedding modelini alır.
    model = get_embedding_model()

    # Bütün chunk metinlerini Python listesine dönüştürür.
    texts = [chunk.page_content for chunk in chunks]

    # Doküman metinlerinin normalized dense embedding vector'lerini oluşturur.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Embedding vector'lerini Milvus ile uyumlu float32 NumPy formatına dönüştürür.
    vectors = np.asarray(vectors, dtype="float32")

    # Oluşturulan embedding matrix'ini döndürür.
    return vectors


# Kullanıcı sorgusunu dense embedding vector'üne dönüştüren fonksiyonu tanımlar.
def embed_query(query):

    # Local embedding modelini alır.
    model = get_embedding_model()

    # Query embedding oluştururken Qwen'in hazır retrieval instruction prompt'unu kullanır.
    vector = model.encode([query], prompt_name="query", normalize_embeddings=True)

    # İlk ve tek query embedding vector'ünü Python listesi olarak döndürür.
    return vector[0].tolist()

# ============================================================
# MILVUS
# ============================================================

# Milvus istemcisinin yalnızca ilk ihtiyaçta oluşturulabilmesi için başlangıç değerini None yapar.
_milvus_client = None


# Local Milvus bağlantısını oluşturan ortak fonksiyonu tanımlar.
def get_milvus_client():

    # Global Milvus istemci değişkenini kullanacağımızı belirtir.
    global _milvus_client

    # Milvus bağlantısı daha önce oluşturulmadıysa yeni istemci oluşturur.
    if _milvus_client is None:

        # Local Milvus sunucusuna bağlantı oluşturur.
        _milvus_client = MilvusClient(uri=MILVUS_URI)

    # Hazır Milvus istemcisini döndürür.
    return _milvus_client


# ============================================================
# DENSE SEARCH
# ============================================================

# Milvus üzerinde Dense Semantic Search gerçekleştiren ortak fonksiyonu tanımlar.
def dense_search(query, top_k=10):

    # Kullanıcının query'sini dense embedding vector'üne dönüştürür.
    query_vector = embed_query(query)

    # Local Milvus istemcisini alır.
    client = get_milvus_client()

    # Retrieval yapılacak collection'ı belleğe yükler.
    client.load_collection(collection_name=COLLECTION_NAME)

    # Dense vector alanında cosine similarity kullanarak semantic search gerçekleştirir.
    results = client.search(collection_name=COLLECTION_NAME, data=[query_vector], anns_field="dense_vector", limit=top_k, output_fields=["text", "document", "page", "chunk_id"], search_params={"metric_type": "COSINE"})

    # Sonuçları diğer proje dosyalarının kolayca kullanabileceği formata dönüştürmek için boş liste oluşturur.
    formatted_results = []

    # İlk query için dönen bütün Milvus sonuçlarını dolaşır.
    for hit in results[0]:

        # Retrieved chunk'ın text ve metadata alanlarını entity değişkenine alır.
        entity = hit["entity"]

        # Sonucu standart Python dictionary formatında listeye ekler.
        formatted_results.append({"text": entity["text"], "document": entity["document"], "page": entity["page"], "chunk_id": entity["chunk_id"], "score": float(hit["distance"])})

    # Standartlaştırılmış retrieval sonuçlarını döndürür.
    return formatted_results

# Retrieval ve reranking sonuçlarını ortak formatta terminalde gösteren fonksiyonu tanımlar.
def print_results(results):

    # Sonuçları sırasıyla dolaşır.
    for index, result in enumerate(results, start=1):

        # Sonuçta reranker skoru olup olmadığını kontrol eder.
        if "rerank_score" in result:

            # Reranking uygulanmışsa BGE relevance skorunu gösterir.
            print(f"{index}. Reranker Skor: {result['rerank_score']:.4f} | {result['chunk_id']} | Sayfa {result['page']}")

        # Reranking uygulanmamışsa normal retrieval skorunu gösterir.
        else:

            # Dense veya Hybrid Search skorunu gösterir.
            print(f"{index}. Retrieval Skor: {result['score']:.4f} | {result['chunk_id']} | Sayfa {result['page']}")

        # Retrieved chunk'ın ilk 500 karakterini gösterir.
        print(result["text"][:500])

        # Sonuçlar arasında görsel ayırıcı oluşturur.
        print("-" * 100)

# ============================================================
# LOCAL LLM
# ============================================================

# Local vLLM istemcisinin yalnızca ilk ihtiyaçta oluşturulabilmesi için başlangıç değerini None yapar.
_llm_client = None


# Local Qwen3-4B API bağlantısını oluşturan ortak fonksiyonu tanımlar.
def get_llm_client():

    # Global LLM istemcisi değişkenini kullanacağımızı belirtir.
    global _llm_client

    # İstemci daha önce oluşturulmadıysa yeni bağlantı oluşturur.
    if _llm_client is None:

        # Local vLLM OpenAI-compatible endpoint'ine bağlanır.
        _llm_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

    # Hazır LLM istemcisini döndürür.
    return _llm_client


# Qwen3-4B modeline ortak chat completion isteği gönderen fonksiyonu tanımlar.
def chat_completion(messages, max_tokens=300):

    # Local LLM istemcisini alır.
    client = get_llm_client()

    # Qwen3-4B modeline OpenAI-compatible chat completion isteği gönderir.
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=0.7, top_p=0.8, max_tokens=max_tokens, extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})

    # Model tarafından üretilen text cevabını temizleyerek döndürür.
    return response.choices[0].message.content.strip()

# ============================================================
# QUERY REWRITING
# ============================================================

# Kullanıcı sorgusunu conversation context kullanarak standalone retrieval query'sine dönüştüren fonksiyonu tanımlar.
def rewrite_query(query, history=None, summary=""):

    # History verilmediyse boş liste kullanır.
    history = history or []

    # Son altı kullanıcı-asistan mesajını okunabilir text formatına dönüştürür.
    history_text = "\n".join([f"{item['role']}: {item['content']}" for item in history[-6:]])

    # Conversation history boşsa modele açık şekilde bilgi olmadığını söyler.
    history_text = history_text if history_text else "Yok"

    # Conversation summary boşsa modele açık şekilde bilgi olmadığını söyler.
    summary_text = summary if summary else "Yok"

    # Query rewriting davranışını belirleyen system mesajını oluşturur.
    system_message = "Konuşma bağlamını kullanarak kullanıcının yeni sorusunu açık, bağımsız ve teknik tek bir RAG retrieval sorgusuna dönüştür. Soruyu cevaplama. Yeni bilgi ekleme. Açıklama yazma. Sadece standalone query döndür."

    # Conversation context ve mevcut kullanıcı sorusunu tek mesajda birleştirir.
    user_message = f"Konuşma özeti:\n{summary_text}\n\nSon mesajlar:\n{history_text}\n\nYeni soru:\n{query}"

    # Qwen3-4B modeline gönderilecek mesaj listesini oluşturur.
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    # Conversation-aware rewritten query oluşturur.
    rewritten_query = chat_completion(messages, max_tokens=120)

    # Standalone retrieval query'sini döndürür.
    return rewritten_query

# ============================================================
# HYBRID SEARCH
# ============================================================

# Dense Semantic Search ile BM25 Search sonuçlarını birleştiren ortak fonksiyonu tanımlar.
def hybrid_search(query, top_k=20):

    # Query'nin dense embedding vector'ünü oluşturur.
    query_vector = embed_query(query)

    # Local Milvus istemcisini alır.
    client = get_milvus_client()

    # Retrieval yapılacak collection'ı belleğe yükler.
    client.load_collection(collection_name=COLLECTION_NAME)

    # Dense vector alanında semantic retrieval isteğini oluşturur.
    dense_request = AnnSearchRequest(data=[query_vector], anns_field="dense_vector", param={"metric_type": "COSINE"}, limit=top_k)

    # Raw query metni üzerinde built-in BM25 lexical retrieval isteğini oluşturur.
    bm25_request = AnnSearchRequest(data=[query], anns_field="sparse_vector", param={"metric_type": "BM25"}, limit=top_k)

    # Dense ve BM25 sonuçlarını Reciprocal Rank Fusion kullanarak birleştirir.
    results = client.hybrid_search(collection_name=COLLECTION_NAME, reqs=[dense_request, bm25_request], ranker=RRFRanker(k=60), limit=top_k, output_fields=["text", "document", "page", "chunk_id"])

    # Sonuçları ortak uygulama formatına çevirmek için boş liste oluşturur.
    formatted_results = []

    # Hybrid Search sonucundaki bütün chunk'ları dolaşır.
    for hit in results[0]:

        # Retrieved chunk'ın text ve metadata alanlarını alır.
        entity = hit["entity"]

        # Sonucu ortak dictionary formatında listeye ekler.
        formatted_results.append({"text": entity["text"], "document": entity["document"], "page": entity["page"], "chunk_id": entity["chunk_id"], "score": float(hit["distance"])})

    # Standartlaştırılmış Hybrid Search sonuçlarını döndürür.
    return formatted_results

# ============================================================
# RERANKING
# ============================================================

# BGE reranker modelini ilk ihtiyaçta yükleyen ortak fonksiyonu tanımlar.
def get_reranker():

    # Global reranker değişkenini fonksiyon içerisinde kullanacağımızı belirtir.
    global _reranker_model

    # Reranker henüz belleğe yüklenmediyse modeli yükler.
    if _reranker_model is None:

        # BGE multilingual CrossEncoder reranker modelini yükler.
        _reranker_model = CrossEncoder(RERANK_MODEL)

    # Hazır reranker modelini döndürür.
    return _reranker_model


# Retrieval sonuçlarını BGE reranker ile yeniden sıralayan fonksiyonu tanımlar.
def rerank_results(query, results, top_k=FINAL_TOP_K):

    # Retrieval sonucu boşsa gereksiz model çağrısı yapmadan boş liste döndürür.
    if not results:

        # Boş sonuç listesini döndürür.
        return []

    # Local BGE reranker modelini alır.
    reranker = get_reranker()

    # Reranker'a gönderilecek query-passage çiftlerini oluşturur.
    pairs = [(query, result["text"]) for result in results]

    # Bütün query-passage çiftleri için relevance skoru üretir.
    scores = reranker.predict(pairs)

    # Reranked sonuçları saklamak için boş liste oluşturur.
    reranked_results = []

    # Retrieval sonuçları ile karşılık gelen reranker skorlarını birlikte dolaşır.
    for result, score in zip(results, scores):

        # Orijinal retrieval sonucunun kopyasını oluşturur.
        reranked_result = result.copy()

        # BGE tarafından hesaplanan relevance skorunu sonuca ekler.
        reranked_result["rerank_score"] = float(score)

        # Güncellenmiş sonucu listeye ekler.
        reranked_results.append(reranked_result)

    # Sonuçları reranker skoruna göre yüksekten düşüğe sıralar.
    reranked_results.sort(key=lambda item: item["rerank_score"], reverse=True)

    # En yüksek skora sahip final Top-K chunk'ı döndürür.
    return reranked_results[:top_k]

# ============================================================
# CONTEXT BUILDING
# ============================================================

# Retrieved chunk'ları LLM'e gönderilecek context formatına dönüştüren fonksiyonu tanımlar.
def build_context(results):

    # Her chunk'ı kaynak numarası, sayfa, chunk ID ve metniyle birlikte tek context içerisinde birleştirir.
    context = "\n\n".join([f"[KAYNAK {index} | Sayfa {result['page']} | {result['chunk_id']}]\n{result['text']}" for index, result in enumerate(results, start=1)])

    # Hazırlanan final context metnini döndürür.
    return context

# ============================================================
# ANSWER GENERATION
# ============================================================

# Retrieved context kullanarak grounded final cevap üreten fonksiyonu tanımlar.
def generate_answer(query, context, history=None):

    # History verilmediyse boş liste kullanır.
    history = history or []

    # Son dört conversation mesajını mevcut sorunun anlamını çözmek için text formatına dönüştürür.
    history_text = "\n".join([f"{item['role']}: {item['content']}" for item in history[-4:]])

    # History boşsa modele geçmiş conversation olmadığını bildirir.
    history_text = history_text if history_text else "Yok"

    # Modelin yalnızca retrieved kaynaklara dayanmasını sağlayan system prompt oluşturur.
    system_message = "Sen kaynaklara dayalı çalışan bir RAG asistanısın. Conversation history bilgisini yalnızca kullanıcının neyi kastettiğini anlamak için kullan. Gerçek bilgi kaynağı olarak sadece verilen context'i kullan. Context içerisinde bulunmayan bilgiyi üretme. Bilgi yoksa mevcut kaynaklarda bulunamadığını açıkça söyle. Türkçe cevap ver ve sonunda kullandığın kaynak numaralarını belirt."

    # Conversation history, kullanıcı sorusu ve retrieved context'i tek mesajda birleştirir.
    user_message = f"Conversation History:\n{history_text}\n\nSoru:\n{query}\n\nContext:\n{context}"

    # Qwen3-4B modeline gönderilecek mesaj listesini oluşturur.
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    # Retrieved context üzerinden final grounded cevap oluşturur.
    answer = chat_completion(messages, max_tokens=600)

    # Final cevabı döndürür.
    return answer

# ============================================================
# CONVERSATION SUMMARY
# ============================================================

# Uzun conversation history'nin kısa teknik özetini oluşturan fonksiyonu tanımlar.
def create_summary(current_summary, query, answer):

    # Conversation summary üretimini yönlendiren system prompt'u oluşturur.
    system_message = "Konuşmanın gelecekteki kullanıcı sorularını anlamak için gerekli teknik bağlamını en fazla 5 kısa cümlede özetle. Gereksiz ayrıntıları çıkar."

    # Önceki summary ile son soru ve cevabı tek mesajda birleştirir.
    user_message = f"Önceki özet:\n{current_summary or 'Yok'}\n\nSon soru:\n{query}\n\nSon cevap:\n{answer}"

    # Qwen3-4B modeline gönderilecek mesaj listesini oluşturur.
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    # Yeni conversation summary oluşturur.
    summary = chat_completion(messages, max_tokens=180)

    # Güncellenmiş conversation summary bilgisini döndürür.
    return summary