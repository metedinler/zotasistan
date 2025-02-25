
# alternative_embedding_module.py

import os
from config_module import config
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from rapidfuzz import fuzz

# Global model cache: Model tekrar yüklemelerini önlemek için kullanılır.
_MODEL_CACHE = {}
def sentence_transformer_embed(text):
    """
    Varsayılan bir SentenceTransformer modeli kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "SENTENCE_TRANSFORMER_MODEL" değişkeni okunur; varsayılan olarak "all-mpnet-base-v2" kullanılır.
    
    Args:
        text (str): Embedding oluşturulacak metin.
    
    Returns:
        list: Oluşturulan embedding vektörü (örneğin, 768 boyutlu liste).
    """
    model = get_sentence_transformer("SENTENCE_TRANSFORMER_MODEL", "all-mpnet-base-v2")
    embedding = model.encode(text)
    return embedding.tolist()

def get_sentence_transformer(model_env_var, default_model):
    model_name = os.getenv(model_env_var, default_model)
    if model_name not in _MODEL_CACHE:
        config.logger.info(f"SentenceTransformer modeli yükleniyor: {model_name}")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]

# --- OpenAI API embedding fonksiyonu için embed_text_with_retry ekliyoruz ---
def embed_text_with_retry(text, max_retries=3):
    """
    OpenAI API kullanarak metin için embedding oluşturur. 
    Hata durumunda maksimum max_retries denemesi yapar.
    """
    from openai import OpenAI
    attempt = 0
    while attempt < max_retries:
        try:
            client_instance = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client_instance.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            config.logger.error(f"OpenAI embedding deneme {attempt+1} hatası: {e}")
            attempt += 1
    return None

def contriever_large_embed(text):
    model = get_sentence_transformer("CONTRIEVER_LARGE_MODEL", "facebook/contriever-large")
    embedding = model.encode(text)
    return embedding.tolist()

def specter_large_embed(text):
    model = get_sentence_transformer("SPECTER_LARGE_MODEL", "allenai-specter-large")
    embedding = model.encode(text)
    return embedding.tolist()

def specter_embed(text):
    model = get_sentence_transformer("SPECTER_MODEL", "allenai-specter")
    embedding = model.encode(text)
    return embedding.tolist()

def all_mpnet_base_v2_embed(text):
    model = get_sentence_transformer("ALL_MPNET_BASE_V2_MODEL", "all-mpnet-base-v2")
    embedding = model.encode(text)
    return embedding.tolist()

def paraphrase_mpnet_base_v2_embed(text):
    model = get_sentence_transformer("PARAPHRASE_MPNET_BASE_V2_MODEL", "paraphrase-mpnet-base-v2")
    embedding = model.encode(text)
    return embedding.tolist()

def stsb_roberta_large_embed(text):
    model = get_sentence_transformer("STSB_ROBERTA_LARGE_MODEL", "stsb-roberta-large")
    embedding = model.encode(text)
    return embedding.tolist()

def labse_embed(text):
    model = get_sentence_transformer("LABSE_MODEL", "LaBSE")
    embedding = model.encode(text)
    return embedding.tolist()

def universal_sentence_encoder_embed(text):
    if "USE" not in _MODEL_CACHE:
        model_url = os.getenv("USE_MODEL", "https://tfhub.dev/google/universal-sentence-encoder/4")
        config.logger.info(f"Universal Sentence Encoder modeli yükleniyor: {model_url}")
        _MODEL_CACHE["USE"] = hub.load(model_url)
    model = _MODEL_CACHE["USE"]
    embedding = model([text])
    return embedding.numpy()[0].tolist()

def universal_sentence_encoder_lite_embed(text):
    if "USE_LITE" not in _MODEL_CACHE:
        model_url = os.getenv("USE_LITE_MODEL", "https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        config.logger.info(f"Universal Sentence Encoder Lite modeli yükleniyor: {model_url}")
        _MODEL_CACHE["USE_LITE"] = hub.load(model_url)
    model = _MODEL_CACHE["USE_LITE"]
    embedding = model([text])
    return embedding.numpy()[0].tolist()


# import os
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import tensorflow_hub as hub
# from config_module import config

# # Global model cache: Model tekrar yüklemelerini önlemek için kullanılır.
# _MODEL_CACHE = {}

# def get_sentence_transformer(model_env_var, default_model):
#     """
#     .env dosyasından model adını okur; eğer belirtilmemişse default modeli kullanır.
#     Yüklenen model, _MODEL_CACHE içinde saklanır.
#     """
#     model_name = os.getenv(model_env_var, default_model)
#     if model_name not in _MODEL_CACHE:
#         config.logger.info(f"SentenceTransformer modeli yükleniyor: {model_name}")
#         _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
#     return _MODEL_CACHE[model_name]

# def contriever_large_embed(text):
#     """
#     Contriever-large modelini kullanarak metni embedding'e dönüştürür.
#     Model: .env üzerinden "CONTRIEVER_LARGE_MODEL" değişkeni; varsayılan: "facebook/contriever-large"
#     Çıktı boyutu: 768
#     """
#     model = get_sentence_transformer("CONTRIEVER_LARGE_MODEL", "facebook/contriever-large")
#     embedding = model.encode(text)
#     return embedding.tolist()

# def specter_large_embed(text):
#     """
#     SPECTER-large modelini kullanarak metni embedding'e dönüştürür.
#     Model: .env üzerinden "SPECTER_LARGE_MODEL"; varsayılan: "allenai-specter-large"
#     Çıktı boyutu: 768
#     """
#     model = get_sentence_transformer("SPECTER_LARGE_MODEL", "allenai-specter-large")
#     embedding = model.encode(text)
#     return embedding.tolist()

# def specter_embed(text):
#     """
#     SPECTER modelini kullanarak metni embedding'e dönüştürür.
#     Model: .env üzerinden "SPECTER_MODEL"; varsayılan: "allenai-specter"
#     Çıktı boyutu: 768
#     """
#     model = get_sentence_transformer("SPECTER_MODEL", "allenai-specter")
#     embedding = model.encode(text)
#     return embedding.tolist()

# def all_mpnet_base_v2_embed(text):
#     """
#     all-mpnet-base-v2 modelini kullanarak metni embedding'e dönüştürür.
#     Model: .env üzerinden "ALL_MPNET_BASE_V2_MODEL"; varsayılan: "all-mpnet-base-v2"
#     Çıktı boyutu: 768
#     """
#     model = get_sentence_transformer("ALL_MPNET_BASE_V2_MODEL", "all-mpnet-base-v2")
#     embedding = model.encode(text)
#     return embedding.tolist()

# def paraphrase_mpnet_base_v2_embed(text):
#     """
#     paraphrase-mpnet-base-v2 modelini kullanarak metni embedding'e dönüştürür.
#     Model: .env üzerinden "PARAPHRASE_MPNET_BASE_V2_MODEL"; varsayılan: "paraphrase-mpnet-base-v2"
#     Çıktı boyutu: 768
#     """
#     model = get_sentence_transformer("PARAPHRASE_MPNET_BASE_V2_MODEL", "paraphrase-mpnet-base-v2")
#     embedding = model.encode(text)
#     return embedding.tolist()

# def stsb_roberta_large_embed(text):
#     """
#     stsb-roberta-large modelini kullanarak metni embedding'e dönüştürür.
#     Model: .env üzerinden "STSB_ROBERTA_LARGE_MODEL"; varsayılan: "stsb-roberta-large"
#     Çıktı boyutu: 1024
#     """
#     model = get_sentence_transformer("STSB_ROBERTA_LARGE_MODEL", "stsb-roberta-large")
#     embedding = model.encode(text)
#     return embedding.tolist()

# def labse_embed(text):
#     """
#     LaBSE modelini kullanarak metni embedding'e dönüştürür.
#     Model: .env üzerinden "LABSE_MODEL"; varsayılan: "LaBSE"
#     Çıktı boyutu: 768
#     """
#     model = get_sentence_transformer("LABSE_MODEL", "LaBSE")
#     embedding = model.encode(text)
#     return embedding.tolist()

# def universal_sentence_encoder_embed(text):
#     """
#     Universal Sentence Encoder modelini kullanarak metni embedding'e dönüştürür.
#     Model URL: .env üzerinden "USE_MODEL"; varsayılan: "https://tfhub.dev/google/universal-sentence-encoder/4"
#     Çıktı boyutu: 512
#     """
#     if "USE" not in _MODEL_CACHE:
#         model_url = os.getenv("USE_MODEL", "https://tfhub.dev/google/universal-sentence-encoder/4")
#         config.logger.info(f"Universal Sentence Encoder modeli yükleniyor: {model_url}")
#         _MODEL_CACHE["USE"] = hub.load(model_url)
#     model = _MODEL_CACHE["USE"]
#     embedding = model([text])
#     return embedding.numpy()[0].tolist()

# def universal_sentence_encoder_lite_embed(text):
#     """
#     Universal Sentence Encoder Lite modelini kullanarak metni embedding'e dönüştürür.
#     Model URL: .env üzerinden "USE_LITE_MODEL"; varsayılan: "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
#     Çıktı boyutu: 512
#     """
#     if "USE_LITE" not in _MODEL_CACHE:
#         model_url = os.getenv("USE_LITE_MODEL", "https://tfhub.dev/google/universal-sentence-encoder-lite/2")
#         config.logger.info(f"Universal Sentence Encoder Lite modeli yükleniyor: {model_url}")
#         _MODEL_CACHE["USE_LITE"] = hub.load(model_url)
#     model = _MODEL_CACHE["USE_LITE"]
#     embedding = model([text])
#     return embedding.numpy()[0].tolist()
