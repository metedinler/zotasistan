# Aşağıda, tartışmalarımız ve yapılan güncellemeler doğrultusunda oluşturulmuş, final sürümü olan 
# **`embedding_module.py`** modülünü bulabilirsiniz. Bu modül, metni belirli boyutlarda parçalara ayırma, 
# OpenAI API kullanarak embedding oluşturma ve büyük dosyaların işlenmesi için ek hata yönetimi (robust embedding) mekanizmasını içeriyor.


import os
import time
import numpy as np
from openai import OpenAI
from config_module import config
from robust_embedding_module import robust_embed_text

def split_text(text, chunk_size=256, method="words"):
    """
    📌 Metni belirlenen chunk_size'a göre böler.
    
    Args:
        text (str): Parçalanacak metin.
        chunk_size (int): Her parça için maksimum kelime sayısı (varsayılan: 256).
        method (str): Bölme yöntemi; "words" kelime bazında, "paragraphs" ise paragraf bazında bölme.
        
    Returns:
        list: Parçalara ayrılmış metin parçalarının listesi.
    """
    if method == "paragraphs":
        # Paragraf bazlı bölme: Çift yeni satır ile ayrılan paragraflar
        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
        return paragraphs
    else:
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text, model="text-embedding-ada-002"):
    """
    📌 OpenAI API kullanarak verilen metin için embedding oluşturur.
    
    Args:
        text (str): Embedding oluşturulacak metin.
        model (str): Kullanılacak model (varsayılan: "text-embedding-ada-002").
        
    Returns:
        list veya None: Oluşturulan embedding vektörü (örneğin, 1536 boyutlu liste) veya hata durumunda None.
    """
    try:
        client_instance = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client_instance.embeddings.create(
            input=text,
            model=model
        )
        config.logger.info(f"✅ Embedding oluşturuldu (model: {model})")
        return response.data[0].embedding
    except Exception as e:
        config.logger.error(f"❌ OpenAI embedding hatası (model: {model}): {e}")
        return None

def process_large_text(text, pdf_id, chunk_size=10000):
    """
    📌 Büyük metinleri, belirlenen chunk boyutuna göre parçalara ayırarak her bir parça için embedding oluşturur.
    Bu fonksiyon, robust_embed_text fonksiyonunu kullanarak her parça için hata yönetimi ve yeniden deneme mekanizmasını uygular.
    
    Args:
        text (str): İşlenecek büyük metin.
        pdf_id (str): PDF'nin temel ID'si (takip ve loglama için).
        chunk_size (int): Her parça için maksimum boyut (varsayılan: 10000 karakter).
        
    Returns:
        list: Oluşturulan embedding vektörlerinin listesi.
    """
    chunks = split_text(text, chunk_size, method="words")
    embeddings = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        emb = robust_embed_text(chunk, pdf_id, i, total_chunks)
        if emb is None:
            config.logger.error(f"❌ Embedding başarısız: PDF {pdf_id}, Chunk {i}")
        else:
            embeddings.append(emb)
        time.sleep(1)  # API rate limit koruması
    return embeddings

# ### Açıklamalar

# - **split_text:**  
#   - İki yöntemi destekler: "words" (varsayılan) kelime bazında bölme ve "paragraphs" yönteminde çift yeni satır (\\n\\n) ile ayrılmış paragraflara bölme.
#   - Bu, hem kısa metinler hem de büyük dosyalar için esnek bölme imkanı sunar.

# - **embed_text:**  
#   - OpenAI API kullanarak, verilen metin üzerinden embedding hesaplar.
#   - Hata durumlarında try/except bloğu ile hatalar loglanır ve fonksiyon None döndürür.

# - **process_large_text:**  
#   - Büyük dosyaların daha verimli işlenmesi için metni belirlenen chunk boyutuna göre böler.
#   - Her bir chunk için robust embedding (retry/backoff mekanizması) uygulanır.
#   - API çağrıları arasında kısa bir bekleme (1 saniye) eklenerek rate limit korunur.

# Bu final sürümü, önceki sürümlere göre daha sağlam hata yönetimi, esnek metin bölme seçenekleri 
# ve büyük dosya işleme desteği sunuyor. Eğer başka bir geliştirme veya ekleme talebiniz varsa lütfen belirtin.

import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from config_module import config
from alternative_embedding_module import get_sentence_transformer, embed_text_with_model, get_available_models

# Varsayılan ayarlar
DEFAULT_MODEL_PRIORITY = ["contriever_large", "specter_large", "all_mpnet", "paraphrase_mpnet"]
OPENAI_MODEL = "text-embedding-ada-002"
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
RATE_LIMIT_DELAY = 1  # API rate limit koruması için sabit gecikme

class EmbeddingManager:
    """
    📌 Embedding işlemlerini yöneten sınıf.
    - OpenAI API ve alternatif embedding modellerini destekler.
    - Retry mekanizması ve circuit breaker ile hata toleranslıdır.
    - Birden fazla model desteği ve parçalama özelliği içerir.
    """

    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model_priority = DEFAULT_MODEL_PRIORITY
        self.circuit_breaker = {}  # Circuit breaker durumunu takip etmek için

    def split_text(self, text, chunk_size=256, method="words"):
        """
        📌 Metni belirlenen chunk_size'a göre böler.
        
        Args:
            text (str): Parçalanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı (varsayılan: 256).
            method (str): Bölme yöntemi; "words" kelime bazında, "paragraphs" ise paragraf bazında bölme.
            
        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR):
        """
        📌 Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.
        
        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parça sayısı.
            model_priority (list, optional): Kullanılacak model sırası. Varsayılan olarak `DEFAULT_MODEL_PRIORITY`.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.
        
        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        # Öncelikle OpenAI API ile embedding oluşturmaya çalış
        if OPENAI_MODEL not in self.circuit_breaker or not self.circuit_breaker[OPENAI_MODEL]:
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=OPENAI_MODEL
                )
                return {"embedding": response.data[0].embedding, "model": OPENAI_MODEL}
            except Exception as e:
                config.logger.warning(f"⚠️ OpenAI modeli başarısız ({OPENAI_MODEL}), alternatif modellere geçiliyor. Hata: {e}")
                self.circuit_breaker[OPENAI_MODEL] = True  # Circuit breaker devreye girer

        # OpenAI başarısız olduysa, alternatif modellere geç
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker açık olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embed_text_with_model(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor ** attempt
                    config.logger.error(f"❌ {model_key} ile embedding başarısız! ({attempt}/{max_retries}) Hata: {e}")
                    time.sleep(wait_time)  # Backoff delay
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        # Tüm denemeler başarısız olursa None döndür
        config.logger.critical(f"🚨 Embedding işlemi tamamen başarısız oldu! (PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks})")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        📌 Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.
        
        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı (varsayılan: 256).
            method (str): Bölme yöntemi; "words" kelime bazında, "paragraphs" ise paragraf bazında bölme.
        
        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.robust_embed_text,
                    chunk,
                    pdf_id,
                    i,
                    total_chunks
                ): chunk for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(RATE_LIMIT_DELAY)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        📌 Embedding verilerini JSON formatında kaydeder.
        
        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"✅ Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"❌ Embedding verileri kaydedilemedi: {e}")


# Bu Modülün Özellikleri
# API Entegrasyonu:
# OpenAI API'si (text-embedding-ada-002) öncelikli olarak kullanılır.
# Alternatif embedding modelleri (sentence-transformers vb.) desteklenir.
# Parçalama Desteği:
# Metinler kelime veya paragraf bazında parçalanabilir.
# Büyük metinler için parçalama ve embedding oluşturma desteği vardır.
# Retry Mekanizması:
# Her model için belirlenen sayıda yeniden deneme yapılır.
# Hata durumunda exponential backoff uygulanır.
# Circuit Breaker:
# Başarısız olan modeller devre dışı bırakılır ve tekrar denenmez.
# Birden Fazla Model Desteği:
# alternative_embedding_module.py'den gelen alternatif modeller desteklenir.
# Model sırası (model_priority) değiştirilebilir.
# API Rate Limit Koruması:
# Sabit time.sleep(1) ile API rate limit koruması sağlanır.
# Loglama ve Hata Yönetimi:
# Güçlü loglama sistemi ile tüm işlemler izlenir.
# Hatalar detaylı olarak loglanır.
# Kullanım Örnekleri
# 1. Büyük Metin İçin Embedding Oluşturma:

# embedding_manager = EmbeddingManager()
# large_text = "Bu çok büyük bir metin..."
# pdf_id = "ABCD1234"
# embeddings = embedding_manager.process_large_text(large_text, pdf_id, chunk_size=256, method="words")
# embedding_manager.save_embeddings(embeddings, "embeddings.json")
# 2. Tek Bir Metin İçin Embedding Oluşturma:

# embedding_manager = EmbeddingManager()
# text = "Bu bir örnek metin."
# pdf_id = "ABCD1234"
# result = embedding_manager.robust_embed_text(text, pdf_id, chunk_index=0, total_chunks=1)
# print(result)