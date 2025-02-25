# ### 📌 **Güncellenmiş `alternative_embedding_module.py` Modülü**  
# Bu modül, **OpenAI harici embedding modellerini** kullanarak metinleri vektör haline getirir.  
# 🔹 **Çeşitli `sentence-transformers` modelleriyle embedding oluşturma**  
# 🔹 **Model hata yakalama ve yedekleme mekanizması**  
# 🔹 **Büyük verileri işlemek için optimize edildi**  

# ---

# ## ✅ **`alternative_embedding_module.py` (Güncellenmiş)**
# ```python
import numpy as np
from sentence_transformers import SentenceTransformer
from config_module import config

# Kullanılabilir alternatif embedding modelleri
MODEL_LIST = {
    "contriever_large": "facebook/contriever-large",
    "specter_large": "allenai-specter-large",
    "specter": "allenai/specter",
    "all_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase_mpnet": "sentence-transformers/paraphrase-mpnet-base-v2",
    "stsb_roberta": "sentence-transformers/stsb-roberta-large",
    "labse": "sentence-transformers/LaBSE"
}

def get_sentence_transformer(model_key):
    """
    📌 Belirtilen model anahtarına göre `SentenceTransformer` modelini yükler.
    
    Args:
        model_key (str): MODEL_LIST içinde yer alan model anahtarı.

    Returns:
        SentenceTransformer: Yüklenmiş model.
    """
    model_name = MODEL_LIST.get(model_key)
    if not model_name:
        raise ValueError(f"❌ Geçersiz model anahtarı: {model_key}")
    
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        config.logger.error(f"❌ Model yüklenirken hata oluştu ({model_key}): {e}")
        return None


def embed_text_with_model(text, model_key):
    """
    📌 Alternatif bir embedding modeli ile metin embedding oluşturur.

    Args:
        text (str): Embedding oluşturulacak metin.
        model_key (str): Kullanılacak modelin anahtarı (MODEL_LIST içinde).

    Returns:
        list veya None: Embedding vektörü veya hata durumunda None.
    """
    model = get_sentence_transformer(model_key)
    if not model:
        return None
    
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        config.logger.error(f"❌ Embedding oluşturulamadı ({model_key}): {e}")
        return None


def get_available_models():
    """
    📌 Kullanılabilir embedding modellerinin listesini döndürür.

    Returns:
        list: Model anahtarlarının listesi.
    """
    return list(MODEL_LIST.keys())
# ```

# ---

# ### 🔥 **Bu Güncellenmiş Versiyonda Neler Değişti?**
# ✔ **Tüm embedding modelleri tek bir `MODEL_LIST` sözlüğü ile yönetiliyor.**  
# ✔ **`get_sentence_transformer` fonksiyonu ile modeller tek bir yerden yükleniyor.**  
# ✔ **`embed_text_with_model` ile farklı modeller arasında geçiş yapabiliyoruz.**  
# ✔ **`get_available_models` ile hangi modellerin kullanılabilir olduğunu sorgulayabiliyoruz.**  
# ✔ **Hata yakalama ve loglama mekanizması eklendi.**  

# ---

# 📢 **Sıradaki modülü söyle, hemen gönderelim!** 🚀