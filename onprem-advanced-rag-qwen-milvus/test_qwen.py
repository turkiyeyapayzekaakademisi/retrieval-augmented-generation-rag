# Ortak Qwen chat fonksiyonunu rag_core.py dosyasından içe aktarır.
from rag_core import chat_completion


# Qwen3-4B'ye gönderilecek basit test mesajını oluşturur.
messages = [{"role": "user", "content": "Sadece 'Qwen çalışıyor' yaz."}]

# Local Qwen3-4B modeline test isteği gönderir.
answer = chat_completion(messages, max_tokens=30)

# Model tarafından üretilen cevabı terminalde gösterir.
print(answer)