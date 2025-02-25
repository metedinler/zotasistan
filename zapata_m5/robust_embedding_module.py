

import time
from config_module import config
from alternative_embedding_module import embed_text_with_model, get_available_models

def robust_embed_text(text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.0):
    """
    Robust embedding oluşturma fonksiyonu.
    
    Bu fonksiyon, verilen metin için öncelikle model_priority listesinde yer alan modelleri
    sırasıyla dener. Her model için maksimum max_retries deneme yapılır; başarısız olursa
    exponential backoff uygulanır ve sonraki modele geçilir.
    
    Args:
        text (str): Embedding oluşturulacak metin.
        pdf_id (str): PDF dosyasının temel ID'si.
        chunk_index (int): İşlenen chunk numarası (0 tabanlı).
        total_chunks (int): Toplam chunk sayısı.
        model_priority (list, optional): Denenecek modellerin listesi. Varsayılan olarak, tüm alternatif modeller.
        max_retries (int, optional): Her model için maksimum deneme sayısı (varsayılan: 3).
        backoff_factor (float, optional): Denemeler arasında kullanılacak exponential backoff katsayısı (varsayılan: 1.0).
    
    Returns:
        list veya None: Başarılı ise embedding vektörü (liste formatında), aksi halde None.
    """
    if model_priority is None:
        model_priority = get_available_models()
    
    for model in model_priority:
        config.logger.info(f"Denenecek model: {model} (PDF: {pdf_id}, Chunk: {chunk_index+1}/{total_chunks})")
        attempt = 0
        while attempt < max_retries:
            try:
                embedding = embed_text_with_model(text, model)
                if embedding is not None:
                    config.logger.info(f"✅ Model {model} başarılı: PDF {pdf_id}, Chunk {chunk_index+1}/{total_chunks}")
                    # Burada, istenirse yerel cache (örn. embedding_cache.json) güncellenebilir.
                    return embedding
            except Exception as e:
                config.logger.error(f"❌ Hata: Model {model}, Deneme {attempt+1}/{max_retries}: {e}")
            attempt += 1
            sleep_time = backoff_factor * (2 ** attempt)
            config.logger.info(f"⏳ Bekleniyor: {sleep_time} saniye")
            time.sleep(sleep_time)
        config.logger.warning(f"⚠️ Model {model} için maksimum deneme sayısına ulaşıldı, sonraki modele geçiliyor.")
    
    config.logger.error(f"❌ Tüm modeller başarısız oldu: PDF {pdf_id}, Chunk {chunk_index+1}/{total_chunks}")
    return None

# Aşağıda, tartışmalarımız ve güncellemeler doğrultusunda oluşturulmuş, final versiyonu olan 
# **robust_embedding_module.py** modülünü bulabilirsiniz. Bu modül, bir metni embedding’e dönüştürürken
# hata toleransı, yeniden deneme (retry) ve alternatif model geçişi mekanizmasını içerir. Özellikle:

# - **Model Priority Listesi:** Eğer model_priority parametresi verilmezse, alternatif embedding modelleri
# (alternative_embedding_module.py içindeki get_available_models fonksiyonundan) kullanılır.
# - **Retry Mekanizması:** Her model için maksimum `max_retries` deneme yapılır. Her denemede başarısız olursa,
# `backoff_factor` çarpanıyla artan bekleme süresi uygulanır (exponential backoff).
# - **Circuit Breaker Mantığı:** Eğer bir model belirlenen deneme sayısına ulaşırsa, o model için ek denemeye girilmeden sonraki modele geçilir.
# - **Loglama:** Tüm denemeler, hata ve başarı durumları config.logger üzerinden ayrıntılı şekilde loglanır.

# Aşağıdaki kod, final versiyonudur:

# ### Açıklamalar

# - **Model Priority:** Eğer `model_priority` parametresi verilmezse, `get_available_models()` fonksiyonu kullanılarak alternatif embedding modellerinin listesi alınır.
# - **Retry & Exponential Backoff:** Her model için `max_retries` deneme yapılır. Her başarısız denemeden sonra, bekleme süresi `backoff_factor * (2 ** attempt)` şeklinde artar.
# - **Circuit Breaker:** Belirlenen deneme sayısına ulaşıldığında, o model için ek deneme yapılmaz ve sonraki model denenir.
# - **Loglama:** Başarı, hata ve bekleme süreleri konfigürasyon loglama sistemi (config.logger) üzerinden bildirilmektedir.

# Bu modül, robust embedding işlemlerinde maksimum esneklik ve hata toleransı sağlamaktadır. Herhangi bir ek geliştirme veya değişiklik talebiniz varsa, lütfen belirtin.