# Açıklama: Bu modül, alternatif embedding modelleri kullanarak metin embedding'i oluşturmayı sağlar.
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
    "stsb_roberta_large": "sentence-transformers/stsb-roberta-large",
    "labse": "sentence-transformers/LaBSE",
    "universal_sentence_encoder": "universal-sentence-encoder",  # TF Hub modeli veya benzeri
    "universal_sentence_encoder_lite": "universal-sentence-encoder-lite"
}

def get_sentence_transformer(model_key):
    """
    📌 Belirtilen model anahtarına göre SentenceTransformer modelini yükler.
    
    Args:
        model_key (str): MODEL_LIST içinde yer alan model anahtarı.
    
    Returns:
        SentenceTransformer veya None: Yüklenmiş model, yüklenemezse None.
    """
    model_name = MODEL_LIST.get(model_key)
    if not model_name:
        raise ValueError(f"❌ Geçersiz model anahtarı: {model_key}")
    try:
        model = SentenceTransformer(model_name)
        config.logger.info(f"✅ {model_key} modeli yüklendi (model adı: {model_name}).")
        return model
    except Exception as e:
        config.logger.error(f"❌ Model yüklenirken hata oluştu ({model_key}): {e}")
        return None

def embed_text_with_model(text, model_key):
    """
    📌 Belirtilen alternatif embedding modeli ile metin embedding'i oluşturur.
    
    Args:
        text (str): Embedding oluşturulacak metin.
        model_key (str): Kullanılacak model anahtarı (MODEL_LIST içinde).
    
    Returns:
        list veya None: Embedding vektörü (örneğin, liste formatında) veya hata durumunda None.
    """
    model = get_sentence_transformer(model_key)
    if not model:
        config.logger.error(f"❌ Model {model_key} yüklenemedi, embedding oluşturulamıyor.")
        return None
    try:
        embedding = model.encode(text)
        config.logger.info(f"✅ Embedding oluşturuldu ({model_key}).")
        return embedding.tolist()
    except Exception as e:
        config.logger.error(f"❌ Embedding oluşturulamadı ({model_key}): {e}")
        return None

def get_available_models():
    """
    📌 Kullanılabilir alternatif embedding modellerinin anahtarlarını döndürür.
    
    Returns:
        list: MODEL_LIST içindeki model anahtarlarının listesi.
    """
    return list(MODEL_LIST.keys())

# ### Açıklamalar

# Aşağıda, tartışmalarımız ve yapılan güncellemeler doğrultusunda oluşturulmuş, final versiyonu olan **`alternative_embedding_module.py`**
# modülünü bulabilirsiniz. Bu modül, alternatif embedding modellerini kullanarak metin embedding’leri üretmek üzere tasarlandı.
# Modül, bir model listesi (MODEL_LIST) üzerinden çalışıyor, istenen modeli yüklemek için `get_sentence_transformer` 
# fonksiyonunu kullanıyor ve `embed_text_with_model` fonksiyonu ile verilen metni embedding’e dönüştürüyor. 
# Ayrıca, mevcut modellerin listesini `get_available_models` fonksiyonu ile döndürüyor.

# - **MODEL_LIST:**  
#   Alternatif embedding modelleri, her bir modelin anahtar ve varsayılan model adını içeriyor. 
# Bu yapı, .env üzerinden de özelleştirilebilecek şekilde genişletilebilir.

# - **get_sentence_transformer:**  
#   Belirtilen model anahtarına göre ilgili SentenceTransformer modelini yükler. Hata durumunda loglama yapar ve `None` döndürür.

# - **embed_text_with_model:**  
#   Yüklenen modeli kullanarak verilen metni embedding’e dönüştürür. Hata durumunda ilgili hata mesajını loglar.

# - **get_available_models:**  
#   Kullanıcıya hangi alternatif modellerin mevcut olduğunu döndürür.

# Bu final sürümü, önceki sürümlerle kıyaslandığında daha kapsamlı hata yakalama, loglama ve kullanıcı geri bildirimi özellikleri eklenmiş,
# eksik fonksiyonlar tamamlanmış bir yapı sunuyor. Herhangi bir ek geliştirme veya değişiklik isterseniz, lütfen belirtin.