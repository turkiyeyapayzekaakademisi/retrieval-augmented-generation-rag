# Ortak PDF yükleme ve chunking fonksiyonlarını rag_core.py dosyasından içe aktarır.
from rag_core import load_and_chunk_pdf, print_chunks


# PDF dosyasını yükler ve character-based chunk'lara ayırır.
chunks = load_and_chunk_pdf()

# Oluşturulan bütün chunk'ları terminalde gösterir.
print_chunks(chunks)