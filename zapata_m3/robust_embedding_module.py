import time
import json
from datetime import datetime
from pathlib import Path
from config_module import config
from alternative_embedding_module import (
    embed_text_with_retry,           # OpenAI API kullanan fonksiyon
    sentence_transformer_embed,      # SentenceTransformer tabanlı alternatif
    contriever_large_embed,
    specter_large_embed,
    specter_embed,
    all_mpnet_base_v2_embed,
    paraphrase_mpnet_base_v2_embed,
    stsb_roberta_large_embed,
    labse_embed,
    universal_sentence_encoder_embed,
    universal_sentence_encoder_lite_embed
)

# Model fonksiyonlarını içeren sözlük
_MODEL_FUNCTIONS = {
    "openai": embed_text_with_retry,
    "sentence": sentence_transformer_embed,
    "contriever_large": contriever_large_embed,
    "specter_large": specter_large_embed,
    "specter": specter_embed,
    "all_mpnet_base_v2": all_mpnet_base_v2_embed,
    "paraphrase_mpnet_base_v2": paraphrase_mpnet_base_v2_embed,
    "stsb_roberta_large": stsb_roberta_large_embed,
    "labse": labse_embed,
    "use": universal_sentence_encoder_embed,
    "use_lite": universal_sentence_encoder_lite_embed
}

# Yığın (cache) dosyası yolu
CACHE_FILE = Path("embedding_cache.json")

def load_embedding_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def update_embedding_cache(entry):
    cache = load_embedding_cache()
    updated = False
    for i, rec in enumerate(cache):
        if rec["pdf_id"] == entry["pdf_id"] and rec["chunk_no"] == entry["chunk_no"]:
            cache[i] = entry
            updated = True
            break
    if not updated:
        cache.append(entry)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def robust_embed_text(text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.0):
    """
    Robust embedding oluşturma fonksiyonu.
    
    Args:
        text (str): Embedding oluşturulacak metin.
        pdf_id (str): PDF'nin temel kimliği.
        chunk_index (int): İşlenen chunk numarası.
        total_chunks (int): PDF için toplam chunk sayısı.
        model_priority (list, optional): Kullanılacak model anahtarlarının sıralı listesi. 
                                         Varsayılan: ["openai", "sentence", "contriever_large", "specter_large", "specter", 
                                                      "all_mpnet_base_v2", "paraphrase_mpnet_base_v2", "stsb_roberta_large",
                                                      "labse", "use", "use_lite"].
        max_retries (int): Her model için maksimum deneme sayısı (varsayılan: 3).
        backoff_factor (float): Denemeler arası gecikme katsayısı.
    
    Returns:
        list veya None: Oluşturulan embedding vektörü (örneğin, 768 ya da 1536 boyutlu liste) veya tüm modeller başarısız olursa None.
    """
    if model_priority is None:
        model_priority = ["openai", "sentence", "contriever_large", "specter_large", "specter",
                          "all_mpnet_base_v2", "paraphrase_mpnet_base_v2", "stsb_roberta_large",
                          "labse", "use", "use_lite"]
    
    chosen_model = None
    embedding_result = None

    for model_key in model_priority:
        func = _MODEL_FUNCTIONS.get(model_key)
        if not func:
            config.logger.error(f"Model fonksiyonu bulunamadı: {model_key}")
            continue
        
        attempt = 0
        while attempt < max_retries:
            try:
                embedding = func(text)
                if embedding is not None:
                    chosen_model = model_key
                    embedding_result = embedding
                    break
            except Exception as e:
                config.logger.error(f"{model_key} modeli deneme {attempt+1} hatası: {e}")
            attempt += 1
            time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
        if embedding_result is not None:
            break
        else:
            config.logger.error(f"{model_key} modeli için {max_retries} deneme başarısız oldu. Alternatif modele geçiliyor.")
    
    # Cache güncellemesi: Hangi model kullanıldı, retry bilgileri ve durum.
    cache_entry = {
        "pdf_id": pdf_id,
        "chunk_no": chunk_index,
        "total_chunks": total_chunks,
        "used_model": chosen_model if chosen_model else "none",
        "status": "success" if embedding_result is not None else "failed",
        "timestamp": datetime.now().isoformat()
    }
    update_embedding_cache(cache_entry)
    
    if embedding_result is not None:
        config.logger.info(f"Embedding başarıyla oluşturuldu. Model: {chosen_model}, PDF ID: {pdf_id}, Chunk: {chunk_index}")
    else:
        config.logger.error(f"Tüm modeller denendi; embedding oluşturulamadı. PDF ID: {pdf_id}, Chunk: {chunk_index}")
    
    return embedding_result
