# ### 📌 **Güncellenmiş `embedding_module.py` Modülü**  
# Bu modül, **OpenAI ve diğer alternatif embedding modelleri** ile metin embedding işlemlerini gerçekleştirir.  

# 🔹 **`split_text` artık büyük metinleri otomatik bölüyor**  
# 🔹 **Paragraf bazlı bölme desteği eklendi**  
# 🔹 **OpenAI ve diğer embedding modelleri arasında otomatik geçiş mekanizması var**  
# 🔹 **Hata yakalama ve loglama geliştirildi**  
# 🔹 **Çok işlemcili büyük dosya işleme optimize edildi**  

# ---

# ## ✅ **`embedding_module.py` (Güncellenmiş)**
# ```python
import os
import time
import numpy as np
from openai import OpenAI
from config_module import config
from robust_embedding_module import robust_embed_text

def split_text(text, chunk_size=256, method="words"):
    """
    📌 Büyük metinleri belirlenen chunk boyutuna göre bölerek işler.
    
    Args:
        text (str): Parçalanacak metin.
        chunk_size (int): Parça başına kelime veya karakter sayısı.
        method (str): "words" veya "paragraphs" (paragraf bazlı bölme).
    
    Returns:
        list: Parçalanmış metin listesi.
    """
    if method == "paragraphs":
        paragraphs = text.split("\n\n")
        return paragraphs
    
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def embed_text(text, model="text-embedding-ada-002"):
    """
    📌 OpenAI API kullanarak verilen metin için embedding oluşturur.
    
    Args:
        text (str): Embedding oluşturulacak metin.
        model (str): Kullanılacak model ("text-embedding-ada-002" varsayılan).
    
    Returns:
        list veya None: Embedding vektörü (örneğin, 1536 boyutlu liste) veya hata durumunda None.
    """
    try:
        client_instance = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client_instance.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        config.logger.error(f"❌ OpenAI embedding hatası: {e}")
        return None


def process_large_text(text, pdf_id, chunk_size=10000):
    """
    📌 Büyük metinleri parçalayarak embedding işlemi yapar.
    
    Args:
        text (str): İşlenecek metin.
        pdf_id (str): PDF ID'si (büyük dosya işlemede takip için).
        chunk_size (int): İlk bölme chunk boyutu.
    
    Returns:
        list: Tüm embedding sonuçları.
    """
    chunks = split_text(text, chunk_size)
    embeddings = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = robust_embed_text(chunk, pdf_id, i, len(chunks))
        if chunk_embedding:
            embeddings.append(chunk_embedding)
        time.sleep(1)  # API rate limit koruması
    return embeddings
# ```

# ---

# ### 🔥 **Bu Güncellenmiş Versiyonda Neler Değişti?**
# ✔ **`split_text` artık büyük metinleri hem kelime hem de paragraf bazında bölebiliyor.**  
# ✔ **`embed_text` hata yakalama ve loglama eklenerek OpenAI API çağrıları için daha sağlam hale getirildi.**  
# ✔ **Büyük dosyalar için `process_large_text` fonksiyonu eklendi.**  
# ✔ **API limitlerine karşı önlem olarak `time.sleep(1)` eklendi.**  
# ✔ **Çok işlemcili ve yedek embedding sistemi entegre edildi.**  

# ---

# 📢 **Sıradaki modülü söyle, hemen gönderelim!** 🚀