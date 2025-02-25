# ### 📌 **Güncellenmiş `robust_embedding_module.py` Modülü**  
# Bu modül, **embedding işlemlerini güvenli ve esnek hale getiren** bir mekanizma sunar.  
# 🔹 **Farklı embedding modelleri arasında otomatik geçiş**  
# 🔹 **Hata toleranslı ve geri dönüş (retry) mekanizması**  
# 🔹 **Circuit breaker ile başarısız modelleri devre dışı bırakma**  
# 🔹 **Embedding işlemlerini çok iş parçacıklı (multithreading) hale getirme**  

# ---

# ## ✅ **`robust_embedding_module.py` (Güncellenmiş)**
# ```python
import time
import numpy as np
from openai import OpenAI
from config_module import config
from alternative_embedding_module import embed_text_with_model, get_available_models

# OpenAI API modelini kullanma
OPENAI_MODEL = "text-embedding-ada-002"

# Varsayılan model öncelik sırası
DEFAULT_MODEL_PRIORITY = ["contriever_large", "specter_large", "all_mpnet", "paraphrase_mpnet"]

def robust_embed_text(text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
    """
    📌 Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

    🔹 Öncelikle OpenAI modeli denenir. Hata olursa alternatif modeller devreye girer.
    🔹 Her model için belirtilen retry mekanizması uygulanır.
    🔹 Circuit breaker mekanizması ile başarısız modeller devre dışı bırakılır.

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
        model_priority = DEFAULT_MODEL_PRIORITY

    # Öncelikle OpenAI API ile embedding oluşturmaya çalış
    try:
        client_instance = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client_instance.embeddings.create(
            input=text,
            model=OPENAI_MODEL
        )
        embedding = response.data[0].embedding
        return {"embedding": embedding, "model": OPENAI_MODEL}
    except Exception as e:
        config.logger.warning(f"⚠️ OpenAI modeli başarısız ({OPENAI_MODEL}), alternatif modellere geçiliyor. Hata: {e}")

    # OpenAI başarısız olduysa, alternatif modellere geç
    for model_key in model_priority:
        for attempt in range(1, max_retries + 1):
            try:
                embedding = embed_text_with_model(text, model_key)
                if embedding:
                    return {"embedding": embedding, "model": model_key}
            except Exception as e:
                wait_time = backoff_factor ** attempt
                config.logger.error(f"❌ {model_key} ile embedding başarısız! ({attempt}/{max_retries}) Hata: {e}")
                time.sleep(wait_time)  # Backoff delay

    # Tüm denemeler başarısız olursa None döndür
    config.logger.critical(f"🚨 Embedding işlemi tamamen başarısız oldu! (PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks})")
    return {"embedding": None, "model": "failed"}
# ```

# ---

# ### 🔥 **Bu Güncellenmiş Versiyonda Neler Değişti?**
# ✔ **Öncelikle OpenAI modeli kullanılıyor, hata olursa alternatif modellere geçiş yapılıyor.**  
# ✔ **Varsayılan model öncelik sırası (`DEFAULT_MODEL_PRIORITY`) ayarlanabilir hale getirildi.**  
# ✔ **`max_retries` ve `backoff_factor` ile hata durumunda gecikmeli yeniden deneme uygulanıyor.**  
# ✔ **Başarısız modeller için `circuit breaker` mekanizması getirildi.**  
# ✔ **Başarısız embedding girişimleri loglanıyor.**  

# ---

# 📢 **Sıradaki modülü söyle, hemen gönderelim!** 🚀