# RAG Project - BGE-M3 + FAISS + Gemini

## Projenin Amacı

Bu projede iki farklı PDF dokümanı kullanarak temel bir **Retrieval Augmented Generation (RAG)** sistemi geliştiriyoruz.

Projede:

- LangChain ile PDF dokümanları yüklenir.
- Dokümanlar numaralı başlıklara göre chunk'lara ayrılır.
- Her chunk için metadata oluşturulur.
- BGE-M3 ile local embedding oluşturulur.
- Embedding'ler FAISS vector index içerisine kaydedilir.
- Kullanıcı sorusu BGE-M3 ile embedding'e dönüştürülür.
- FAISS bütün chunk'lar üzerinde semantic search yapar.
- Similarity Threshold ile ilgisiz sonuçlar elenir.
- Top-K ile en ilgili chunk'lar seçilir.
- Seçilen chunk'lar context haline getirilir.
- Context Gemini LLM'e gönderilir.
- Kaynaklara dayalı final cevap üretilir.

Embedding ve retrieval işlemleri tamamen local çalışır.
Gemini API yalnızca final cevap üretiminde kullanılır.

---

# Kullanılan Teknolojiler

- Python
- LangChain
- PyPDFLoader
- LangChain Document
- LangChain PromptTemplate
- BAAI/bge-m3
- Sentence Transformers
- FAISS
- NumPy
- Google Gemini
- Google Gen AI SDK
- python-dotenv

---

# Proje Yapısı

```text
google-gemini-rag-project/
│
├── data/
│   ├── insan_kaynaklari_surecleri.pdf
│   └── bilgi_guvenligi_bt_politikalari.pdf
│
├── vector_store/
│   ├── index.faiss
│   └── chunks.json
│
├── config.py
├── ingestion.py
├── rag.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 7.1 Proje Mimarisi ve Ortamın Hazırlanması

## Sanal Ortam Oluşturma

Proje klasörüne girilir:

```powershell
cd google-gemini-rag-project
```

Sanal ortam oluşturulur:

```powershell
py -m venv .venv
```

Sanal ortam aktif edilir:

```powershell
.venv\Scripts\activate
```

Pip güncellenir:

```powershell
python -m pip install --upgrade pip
```

---

## Gerekli Paketlerin Kurulması

```powershell
pip install google-genai langchain-core langchain-community pypdf faiss-cpu numpy python-dotenv sentence-transformers
```

---

## requirements.txt

```text
google-genai
langchain-core
langchain-community
pypdf
faiss-cpu
numpy
python-dotenv
sentence-transformers
```

Paketlerin toplu kurulumu:

```powershell
pip install -r requirements.txt
```

---

## .env

Gemini API key `.env` dosyasına eklenir:

```text
GEMINI_API_KEY=BURAYA_API_KEY
```

Gemini bu projede yalnızca final cevap üretiminde kullanılır.

---

## .gitignore

```text
.venv/
.env
__pycache__/
*.pyc
vector_store/
```

---

## Kullanılan Modeller

Embedding modeli:

```text
BAAI/bge-m3
```

Embedding boyutu:

```text
1024
```

LLM:

```text
Gemini
```

BGE-M3 local çalışır.

Model ilk kullanımda bilgisayara indirilir. Sonraki çalıştırmalarda local cache üzerinden kullanılır.

---

# config.py

`config.py`, proje genelinde kullanılan yolları, model isimlerini ve retrieval parametrelerini içerir.

Temel ayarlar:

```text
DATA_DIR
VECTOR_STORE_DIR
FAISS_INDEX_PATH
CHUNKS_PATH

EMBEDDING_MODEL
EMBEDDING_DIMENSION

GEMINI_API_KEY
LLM_MODEL

FETCH_K
TOP_K
SIMILARITY_THRESHOLD
```

Örnek retrieval ayarları:

```text
FETCH_K = 20
TOP_K = 3
SIMILARITY_THRESHOLD = 0.60
```

`FETCH_K`, FAISS'ten ilk aşamada getirilecek aday sayısını belirler.

`TOP_K`, final olarak LLM'e gönderilecek maksimum chunk sayısını belirler.

`SIMILARITY_THRESHOLD`, yeterince ilgili olmayan chunk'ların elenmesi için kullanılır.

---

# ingestion.py

`ingestion.py`, PDF dokümanlarının hazırlanmasından FAISS vector index oluşturulmasına kadar olan ingestion pipeline'ını içerir.

Akış:

```text
PDF
↓
Doküman Yükleme
↓
Heading-Based Chunking
↓
Metadata
↓
Section Title + Chunk Text
↓
BGE-M3 Embedding
↓
FAISS
```

Çalıştırma:

```powershell
python ingestion.py
```

---

# 7.2 PDF Dokümanlarının Yüklenmesi

Bu bölümde iki PDF dokümanı LangChain `PyPDFLoader` ile yüklenir.

Kullanılan dokümanlar:

```text
insan_kaynaklari_surecleri.pdf
bilgi_guvenligi_bt_politikalari.pdf
```

PDF dosyaları sayfa sayfa okunur.

Her sayfa LangChain `Document` nesnesi olarak elde edilir.

---

## Test

`ingestion.py` çalıştırıldığında terminalde doküman bilgileri görüntülenir.

Örnek:

```text
=== 7.2 PDF DOKÜMANLARININ YÜKLENMESİ ===

Doküman      : insan_kaynaklari_surecleri.pdf
Başlık       : İnsan Kaynakları Süreçleri ve Çalışan Politikaları
Kategori     : İnsan Kaynakları
Sayfa sayısı : 3
İlk metin    : ...
```

İkinci PDF için de benzer bir çıktı görülmelidir.

---

## Beklenen Sonuç

- PDF dosyaları bulunabilmelidir.
- PDF dosyaları başarılı şekilde açılmalıdır.
- Sayfa sayıları doğru görünmelidir.
- PDF içerisindeki metin terminalde okunabilmelidir.

---

# 7.3 Chunking ve Metadata Üretimi

PDF içerisindeki metin numaralı başlıklara göre chunk'lara ayrılır.

Karakter veya token sayısına göre sabit chunking yerine **heading-based chunking** kullanılır.

Örnek doküman yapısı:

```text
1. Giriş

1.1 Amaç

1.2 Kapsam

2. İzin Süreçleri

2.1 Yıllık İzin

2.2 Mazeret İzni

2.3 Hastalık İzni
```

Her numaralı başlık ayrı bir chunk oluşturur.

Daha derin başlıklar da desteklenir:

```text
2.1.1 Alt Başlık
2.1.2 Alt Başlık
```

Ana doküman başlığı ayrı bir chunk yapılmaz.

Örneğin:

```text
İnsan Kaynakları Süreçleri ve Çalışan Politikaları
```

bilgisi `document_title` metadata alanında tutulur.

---

## Metadata

Her chunk için aşağıdaki metadata alanları oluşturulur:

```text
document
document_title
category
section
section_title
page
chunk_id
```

Örnek:

```text
document       : insan_kaynaklari_surecleri.pdf
document_title : İnsan Kaynakları Süreçleri ve Çalışan Politikaları
category       : İnsan Kaynakları
section        : 2.1
section_title  : Yıllık İzin
page           : 1
chunk_id       : insan_kaynaklari_surecleri_2_1
```

Metadata semantic search sırasında filtreleme amacıyla kullanılmaz.

Metadata yalnızca:

- kaynak dokümanı göstermek,
- bölüm bilgisini korumak,
- sayfa bilgisini taşımak,
- final cevapta kaynak belirtmek

için kullanılır.

---

## LangChain Document Yapısı

Her chunk şu mantıkta tutulur:

```text
Document
├── page_content
└── metadata
```

`page_content`, ilgili bölümün gerçek metnini içerir.

`metadata`, chunk'ın hangi dokümana ve bölüme ait olduğunu belirtir.

---

## Test

Bu aşamada yakalanan **bütün chunk'lar terminalde tam metinleriyle gösterilir**.

Örnek:

```text
====================================================================================================
CHUNK 5
====================================================================================================

Chunk ID        : insan_kaynaklari_surecleri_2_1
Doküman         : insan_kaynaklari_surecleri.pdf
Doküman Başlığı : İnsan Kaynakları Süreçleri ve Çalışan Politikaları
Kategori        : İnsan Kaynakları
Bölüm           : 2.1
Başlık          : Yıllık İzin
Sayfa           : 1
Metin           : Yıllık izin talebi...
```

Terminalde toplam chunk sayısı da görüntülenir:

```text
Toplam chunk sayısı: 41
```

Chunk sayısı doküman içeriği değiştiğinde farklı olabilir.

---

## Beklenen Sonuç

- `1.`, `2.`, `3.` gibi ana bölümler ayrı chunk olmalıdır.
- `1.1`, `2.1`, `3.2` gibi alt bölümler ayrı chunk olmalıdır.
- Chunk metinleri birbirine karışmamalıdır.
- Her chunk doğru başlık ve bölüm numarasına sahip olmalıdır.
- Metadata bilgileri doğru oluşturulmalıdır.
- Bütün chunk'lar terminalde kontrol edilebilmelidir.

---

# 7.4 Embedding Oluşturma

Bu bölümde oluşturulan bütün chunk'lar local çalışan BGE-M3 modeli ile embedding'e dönüştürülür.

Kullanılan model:

```text
BAAI/bge-m3
```

Embedding boyutu:

```text
1024
```

Akış:

```text
Section Title
+
Chunk Text
↓
BGE-M3
↓
1024 Boyutlu Embedding
```

Embedding oluşturulurken yalnızca `page_content` kullanılmaz.

Bölüm başlığı da metnin başına eklenir.

Örneğin:

```text
Yıllık İzin. Yıllık izin talebi mümkün olduğunca planlanan izin başlangıç tarihinden...
```

Bu sayede bölüm başlığındaki anlam da embedding içerisinde temsil edilir.

---

## Normalization

Embedding oluşturulurken:

```text
normalize_embeddings=True
```

kullanılır.

Bu sayede vector'ler normalize edilir ve FAISS `IndexFlatIP` ile cosine similarity tabanlı arama yapılabilir.

---

## İlk Çalıştırma

BGE-M3 modeli ilk çalıştırmada bilgisayara indirilir.

Sonraki çalıştırmalarda model tekrar indirilmez ve local cache üzerinden kullanılır.

---

## Test

Terminalde:

```text
=== 7.4 EMBEDDING OLUŞTURMA ===

Embedding sayısı : 41
Embedding boyutu : 1024
```

gibi bir çıktı görülmelidir.

Ayrıca ilk embedding'in ilk birkaç değeri görüntülenir:

```text
İlk embedding'in ilk 10 değeri:

[...]
```

---

## Beklenen Sonuç

- Her chunk için bir embedding oluşturulmalıdır.
- Embedding sayısı chunk sayısıyla aynı olmalıdır.
- Embedding boyutu 1024 olmalıdır.
- Embedding işlemi tamamen local çalışmalıdır.
- Gemini API embedding aşamasında kullanılmamalıdır.

---

# 7.5 FAISS Vector Index Oluşturma

BGE-M3 tarafından oluşturulan embedding'ler FAISS içerisine eklenir.

Kullanılan index:

```text
IndexFlatIP
```

Vector'ler normalize edildiği için Inner Product değeri cosine similarity olarak kullanılabilir.

Akış:

```text
Embedding Matrix
↓
FAISS IndexFlatIP
↓
Vector Index
```

---

## FAISS ve Chunk Bilgileri

FAISS yalnızca vector'leri saklar.

Vector index:

```text
index.faiss
```

dosyasına kaydedilir.

Chunk metinleri ve metadata bilgileri ise:

```text
chunks.json
```

dosyasına kaydedilir.

FAISS içerisindeki vector sırası ile `chunks.json` içerisindeki chunk sırası aynıdır.

Bu sayede FAISS'ten dönen ID ile ilgili chunk bulunabilir.

---

## Test

Terminalde:

```text
=== 7.5 FAISS VECTOR INDEX OLUŞTURMA ===

FAISS vector sayısı : 41
Embedding boyutu    : 1024
```

görülmelidir.

---

## Oluşması Gereken Dosyalar

```text
vector_store/
├── index.faiss
└── chunks.json
```

---

## Beklenen Sonuç

- FAISS vector sayısı chunk sayısıyla aynı olmalıdır.
- `index.faiss` oluşmalıdır.
- `chunks.json` oluşmalıdır.
- Ingestion pipeline hata vermeden tamamlanmalıdır.

---

# Ingestion Pipeline Sonucu

`python ingestion.py` çalıştırıldığında sırasıyla:

```text
7.2 PDF Dokümanlarının Yüklenmesi
↓
7.3 Chunking ve Metadata Üretimi
↓
7.4 Embedding Oluşturma
↓
7.5 FAISS Vector Index Oluşturma
```

adımları gerçekleştirilir.

Final ingestion akışı:

```text
PDF
↓
PyPDFLoader
↓
Heading-Based Chunking
↓
LangChain Document
↓
Metadata
↓
Section Title + Chunk Text
↓
BGE-M3
↓
FAISS
```

---

# rag.py

`rag.py`, kullanıcı sorusundan final cevaba kadar olan retrieval ve generation sürecini içerir.

Kullanıcı yalnızca sorusunu yazar.

Herhangi bir doküman veya kategori seçimi yapılmaz.

Akış:

```text
Kullanıcı Sorusu
↓
BGE-M3 Query Embedding
↓
FAISS Semantic Search
↓
Similarity Threshold
↓
Top-K
↓
Context
↓
Gemini
↓
Kaynaklara Dayalı Cevap
```

Çalıştırma:

```powershell
python rag.py
```

---

# 7.6 Kullanıcı Sorusunu Embedding'e Çevirme

Kullanıcıdan soru alınır.

Örnek:

```text
Sorunuzu yazın: VPN ne zaman kullanılmalıdır?
```

Kullanıcı sorusu BGE-M3 ile local olarak embedding'e dönüştürülür.

Doküman embedding'leri ile kullanıcı sorusu için aynı embedding modeli kullanılır.

---

## Test

Terminalde:

```text
=== 7.6 KULLANICI SORUSUNU EMBEDDING'E ÇEVİRME ===

Soru             : VPN ne zaman kullanılmalıdır?
Embedding boyutu : 1024
```

görülmelidir.

Ayrıca query embedding'in ilk değerleri terminalde gösterilir.

---

## Beklenen Sonuç

- Kullanıcı sorusu embedding'e dönüşmelidir.
- Embedding boyutu 1024 olmalıdır.
- İşlem local çalışmalıdır.

---

# 7.7 Semantic Search

Kullanıcı sorusunun embedding'i FAISS üzerinde bütün chunk'lar ile karşılaştırılır.

Herhangi bir metadata filtresi uygulanmaz.

Kullanıcı hangi PDF içerisinde arama yapılacağını seçmez.

FAISS, kullanıcı sorusuna anlamsal olarak en yakın chunk'ları bütün vector store içerisinden bulur.

Akış:

```text
Query Embedding
↓
FAISS
↓
Tüm Chunk'lar
↓
Cosine Similarity
↓
En Yakın Chunk'lar
```

FAISS ilk aşamada:

```text
FETCH_K = 20
```

adet aday getirir.

---

## Test

Örneğin:

```text
Sorunuzu yazın: VPN ne zaman kullanılmalıdır?
```

sorusu için terminalde:

```text
=== 7.7 SEMANTIC SEARCH ===

--- SEMANTIC SEARCH SONUÇLARI ---

Similarity : 0.81
Doküman    : Bilgi Güvenliği ve BT Kullanım Politikaları
Kategori   : Bilgi Güvenliği
Bölüm      : 4.1 - VPN Kullanımı
Metin      : ...
```

gibi sonuçlar görülmesi beklenir.

Metadata burada semantic search kararını etkilemez.

Sadece bulunan chunk'ın hangi dokümana ve bölüme ait olduğunu göstermek için kullanılır.

---

## Beklenen Sonuç

- Semantic search bütün chunk'lar üzerinde yapılmalıdır.
- Kullanıcıdan kategori seçmesi istenmemelidir.
- En ilgili chunk'lar üst sıralarda gelmelidir.
- Similarity değerleri terminalde görülmelidir.

---

# 7.8 Top-K ve Similarity Threshold

Semantic search sonucunda bulunan bütün chunk'lar LLM'e gönderilmez.

İki kontrol uygulanır:

```text
Similarity Threshold
Top-K
```

---

## Similarity Threshold

Örnek:

```text
SIMILARITY_THRESHOLD = 0.60
```

Similarity değeri threshold altında kalan chunk'lar elenir.

Örnek:

```text
0.84 → KALDI
0.72 → KALDI
0.63 → KALDI
0.48 → ELENDİ
```

Threshold değeri her embedding modeli ve veri seti için aynı olmak zorunda değildir.

Proje geliştirilirken test sorularına göre optimize edilebilir.

---

## Top-K

Threshold'u geçen sonuçlardan en iyi:

```text
TOP_K = 3
```

adet chunk seçilir.

Akış:

```text
FAISS Sonuçları
↓
Similarity Threshold
↓
Uygun Chunk'lar
↓
Top-K
↓
Final Retrieval
```

---

## Test

Terminalde:

```text
=== 7.8 TOP-K VE SIMILARITY THRESHOLD ===

Similarity Threshold : 0.60
Top-K                : 3

--- THRESHOLD KONTROLÜ ---

0.84 | KALDI  | bilgi_guvenligi_bt_politikalari.pdf | VPN Kullanımı
0.72 | KALDI  | bilgi_guvenligi_bt_politikalari.pdf | Ağ ve Uzaktan Erişim
0.63 | KALDI  | bilgi_guvenligi_bt_politikalari.pdf | Uzaktan Destek
0.48 | ELENDİ | insan_kaynaklari_surecleri.pdf | Çalışma Saatleri

--- FINAL TOP-K ---

0.84 | bilgi_guvenligi_bt_politikalari.pdf | 4.1 - VPN Kullanımı
0.72 | bilgi_guvenligi_bt_politikalari.pdf | 4 - Ağ ve Uzaktan Erişim
0.63 | bilgi_guvenligi_bt_politikalari.pdf | 4.3 - Uzaktan Destek
```

gibi bir sonuç görülebilir.

Similarity değerleri örnektir ve gerçek sonuçlar modele göre değişebilir.

---

## Beklenen Sonuç

- Düşük similarity değerine sahip chunk'lar elenmelidir.
- Threshold'u geçen sonuçlar kalmalıdır.
- En iyi Top-K sonuç final retrieval sonucuna dönüşmelidir.

---

# 7.9 LLM'e Besleme ve Cevap Üretme

Final retrieval sonucunda kalan chunk'lar tek bir context haline getirilir.

Bu context Gemini LLM'e gönderilir.

Gemini API yalnızca bu aşamada kullanılır.

---

## Context Yapısı

Her retrieved chunk kaynak bilgisiyle birlikte context'e eklenir.

Örnek:

```text
[KAYNAK 1]

Doküman: Bilgi Güvenliği ve BT Kullanım Politikaları
Bölüm: 4.1 - VPN Kullanımı
Sayfa: 2

Metin:
Kurumsal sistemlere şirket ağı dışından erişim sağlanırken...
```

Birden fazla chunk bulunursa bütün kaynaklar aynı context içerisinde birleştirilir.

---

## Prompt

Gemini'ye aşağıdaki temel kurallar verilir:

- Sadece verilen context içerisindeki bilgileri kullan.
- Kaynaklarda olmayan bilgi üretme.
- Tahmin yapma.
- Türkçe cevap ver.
- Cevabı açık ve kısa tut.
- Kullanılan doküman ve bölüm bilgisini belirt.
- Context yeterli değilse bunu açıkça söyle.

---

## Test

Terminalde önce:

```text
--- LLM'E GİDEN CONTEXT ---
```

görülür.

Burada Gemini'ye gönderilen gerçek chunk'lar kontrol edilir.

Daha sonra:

```text
--- GEMINI CEVABI ---
```

görülür.

Örnek:

```text
VPN, şirket ağı dışından kurumsal sistemlere erişim sağlanırken kullanılmalıdır.

Kaynak:
Bilgi Güvenliği ve BT Kullanım Politikaları - 4.1 VPN Kullanımı
```

---

## Beklenen Sonuç

- Top-K chunk'lar context haline gelmelidir.
- Context terminalde görüntülenmelidir.
- Gemini yalnızca retrieval sonucunda verilen context'i kullanmalıdır.
- Final cevap kaynaklara dayanmalıdır.
- Kaynak doküman ve bölüm bilgisi cevapta gösterilmelidir.

---

# Projeyi Çalıştırma

## 1. Ingestion

İlk olarak:

```powershell
python ingestion.py
```

çalıştırılır.

Bu işlem:

```text
PDF
↓
Chunking
↓
Metadata
↓
BGE-M3 Embedding
↓
FAISS
```

adımlarını gerçekleştirir.

PDF içeriği veya chunking mantığı değişirse `ingestion.py` tekrar çalıştırılmalıdır.

---

## 2. RAG

Ingestion tamamlandıktan sonra:

```powershell
python rag.py
```

çalıştırılır.

Kullanıcı yalnızca sorusunu girer:

```text
Sorunuzu yazın: VPN ne zaman kullanılmalıdır?
```

Sistem otomatik olarak:

```text
Soru
↓
Query Embedding
↓
Semantic Search
↓
Similarity Threshold
↓
Top-K
↓
Context
↓
Gemini
↓
Cevap
```

akışını çalıştırır.

---

# Örnek Sorular

## İnsan Kaynakları

```text
Yıllık izin talebi nasıl yapılır?
```

```text
Hastalık izninde çalışan ne yapmalıdır?
```

```text
Uzaktan çalışma sırasında çalışanların sorumlulukları nelerdir?
```

```text
Çalışma saatleri nelerdir?
```

```text
Eğitim talebi nasıl değerlendirilir?
```

---

## Bilgi Güvenliği

```text
VPN ne zaman kullanılmalıdır?
```

```text
Şüpheli bir e-posta aldığımda ne yapmalıyım?
```

```text
Kurumsal dosyaları kişisel bulut hesabımda saklayabilir miyim?
```

```text
Parola kullanırken nelere dikkat edilmelidir?
```

```text
Cihaz kaybolursa ne yapılmalıdır?
```

---

# Final Mimari

```text
PDF 1
   \
    \
     ↓
Heading-Based Chunking
     ↑
    /
   /
PDF 2

     ↓

Metadata
- document
- document_title
- category
- section
- section_title
- page
- chunk_id

     ↓

Section Title + Chunk Text

     ↓

BGE-M3 Local Embedding

     ↓

FAISS Vector Index

==============================

Kullanıcı Sorusu

     ↓

BGE-M3 Query Embedding

     ↓

FAISS Semantic Search
Tüm Chunk'lar

     ↓

Similarity Threshold

     ↓

Top-K

     ↓

Context

     ↓

Gemini LLM

     ↓

Kaynaklara Dayalı Cevap
```

---

# Projede Kullanılan Temel Bileşenler

```text
PDF Loading      → LangChain PyPDFLoader

Chunking         → Heading-Based Chunking

Chunk Yapısı     → LangChain Document

Metadata         → Document.metadata

Embedding        → BAAI/bge-m3

Embedding Boyutu → 1024

Vector Search    → FAISS IndexFlatIP

Similarity       → Cosine Similarity

Filtering        → Similarity Threshold

Retrieval        → Top-K

Prompt           → LangChain PromptTemplate

LLM              → Gemini
```

---

# Projenin Tam Akışı

```text
PDF Dokümanları
↓
PyPDFLoader
↓
Numaralı Başlıkların Tespiti
↓
Heading-Based Chunking
↓
Chunk + Metadata
↓
Section Title + Chunk Text
↓
BGE-M3 Embedding
↓
FAISS Index
↓
────────────────────────────
↓
Kullanıcı Sorusu
↓
BGE-M3 Query Embedding
↓
Tüm Chunk'larda Semantic Search
↓
Similarity Threshold
↓
Top-K
↓
Retrieved Context
↓
Gemini
↓
Kaynaklara Dayalı Final Cevap
```

Bu proje tamamlandığında temel bir **Classic RAG pipeline** uçtan uca uygulanmış olur.