import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# .env dosyasını belirtilen tam yoldan yükleyin
load_dotenv("C:/Users/mete/Zotero/zotasistan/.env")

class Yapilandirma:
    def __init__(self):
        # Temel ortam değişkenleri
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR"))
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR"))
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
        self.ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
        
        # Çıktı dizinleri
        self.TEMIZMETIN_DIR = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.getenv("TEMIZ_TABLO_DIZIN", "temiztablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.getenv("TEMIZ_KAYNAKCA_DIZIN", "temizkaynakca"))
        self.EMBEDDING_PARCA_DIR = Path(os.getenv("EMBEDDING_PARCA_DIZIN", "embedingparca"))
        
        # Citation Mapping için dizin (varsa .env'de CITATIONS_DIR, yoksa varsayılan "citations")
        self.CITATIONS_DIR = Path(os.getenv("CITATIONS_DIR", "citations"))
        
        # Loglama dizini
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
        self.STACK_DOSYASI = self.LOG_DIR / "islem.stack"
        
        # Gerekli tüm dizinleri oluştur
        self._dizinleri_hazirla()
        # Loglama sistemini yapılandır
        self._loglama_ayarla()
        
    def _dizinleri_hazirla(self):
        """Gerekli dizinlerin varlığını kontrol eder ve oluşturur."""
        for d in [self.STORAGE_DIR, self.SUCCESS_DIR, self.TEMIZMETIN_DIR, 
                  self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN, 
                  self.EMBEDDING_PARCA_DIR, self.CITATIONS_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)
            
    def _loglama_ayarla(self):
        """Merkezi loglama sistemini başlatır: Dosya ve konsol loglarını yapılandırır."""
        self.logger = logging.getLogger('ZoteroProcessor')
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Dosya log handler
        file_handler = logging.FileHandler(self.LOG_DIR / 'islem.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Konsol log handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

# Global yapılandırma nesnesi
config = Yapilandirma()

# ---------------------------------------------------------------------
# zotero_module.py
# ---------------------------------------------------------------------
import os
import re
import requests
import json
from config_module import config

class ZoteroEntegratoru:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
        self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

    def meta_veri_al(self, item_key):
        """
        Belirtilen item_key için Zotero API'den bibliyometrik veriyi çeker.
        """
        try:
            response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API hatası: {str(e)}")
            return None

    def referanslari_analiz_et(self, referans_listesi):
        """
        Verilen referans listesindeki her referans için, yazar bilgisini (ilk kelime) çıkararak analiz eder.
        Eğer referans bulunamazsa 'Bilinmeyen' döndürür.
        """
        try:
            analiz_sonuc = []
            for referans in referans_listesi:
                yazar = re.search(r'^([A-Za-z]+)', referans)
                analiz_sonuc.append({
                    'orijinal': referans,
                    'yazar': yazar.group(1) if yazar else 'Bilinmeyen'
                })
            return analiz_sonuc
        except Exception as e:
            config.logger.error(f"Referans analiz hatası: {str(e)}")
            return []

def dokuman_id_al(dosya_adi):
    """
    Dosya adından, örneğin 'ABCD1234.pdf' şeklinde bir isimden, temel dosya kimliğini (ID) çıkarır.
    """
    m = re.search(r"^([A-Z0-9]+)\..*", dosya_adi)
    return m.group(1) if m else None

def fetch_zotero_metadata(item_key):
    """
    Zotero API'den belirtilen item_key için bibliyometrik veriyi çeker.
    """
    headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    try:
        response = requests.get(f"{config.base_url}/{item_key}", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
            return None
    except Exception as e:
        config.logger.error(f"Zotero API isteğinde hata: {e}")
        return None

def save_references_for_analysis(references, vosviewer_file, pajek_file):
    """
    Kaynakça verilerini, VOSviewer ve Pajek formatlarında kaydeder.
    
    - VOSviewer dosyası: İlk satır 'label' içermeli, ardından her referans ayrı satırda yer alır.
    - Pajek dosyası: İlk satır "*Vertices <sayı>" şeklinde, sonrasında her referans numaralandırılarak listelenir.
    """
    try:
        with open(vosviewer_file, 'w', encoding='utf-8') as vos_file:
            vos_file.write("label\n")
            for ref in references:
                vos_file.write(f"{ref}\n")
        config.logger.info(f"VOSviewer formatında referanslar kaydedildi: {vosviewer_file}")
        
        with open(pajek_file, 'w', encoding='utf-8') as pajek_f:
            pajek_f.write(f"*Vertices {len(references)}\n")
            for i, ref in enumerate(references, 1):
                pajek_f.write(f'{i} "{ref}"\n')
        config.logger.info(f"Pajek formatında referanslar kaydedildi: {pajek_file}")
    except Exception as e:
        config.logger.error(f"Referanslar analiz formatlarına kaydedilemedi: {e}")
# ---------------------------------------------------------------------
# pdf_processing.py
# ---------------------------------------------------------------------

import os
import re
import requests
import json
from config_module import config

class ZoteroEntegratoru:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
        self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

    def meta_veri_al(self, item_key):
        """
        Belirtilen item_key için Zotero API'den bibliyometrik veriyi çeker.
        """
        try:
            response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API hatası: {str(e)}")
            return None

    def referanslari_analiz_et(self, referans_listesi):
        """
        Verilen referans listesindeki her referans için, yazar bilgisini (ilk kelime) çıkararak analiz eder.
        Eğer referans bulunamazsa 'Bilinmeyen' döndürür.
        """
        try:
            analiz_sonuc = []
            for referans in referans_listesi:
                yazar = re.search(r'^([A-Za-z]+)', referans)
                analiz_sonuc.append({
                    'orijinal': referans,
                    'yazar': yazar.group(1) if yazar else 'Bilinmeyen'
                })
            return analiz_sonuc
        except Exception as e:
            config.logger.error(f"Referans analiz hatası: {str(e)}")
            return []

def dokuman_id_al(dosya_adi):
    """
    Dosya adından, örneğin 'ABCD1234.pdf' şeklinde bir isimden, temel dosya kimliğini (ID) çıkarır.
    """
    m = re.search(r"^([A-Z0-9]+)\..*", dosya_adi)
    return m.group(1) if m else None

def fetch_zotero_metadata(item_key):
    """
    Zotero API'den belirtilen item_key için bibliyometrik veriyi çeker.
    """
    headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    try:
        response = requests.get(f"{config.base_url}/{item_key}", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
            return None
    except Exception as e:
        config.logger.error(f"Zotero API isteğinde hata: {e}")
        return None

def save_references_for_analysis(references, vosviewer_file, pajek_file):
    """
    Kaynakça verilerini, VOSviewer ve Pajek formatlarında kaydeder.
    
    - VOSviewer dosyası: İlk satır 'label' içermeli, ardından her referans ayrı satırda yer alır.
    - Pajek dosyası: İlk satır "*Vertices <sayı>" şeklinde, sonrasında her referans numaralandırılarak listelenir.
    """
    try:
        with open(vosviewer_file, 'w', encoding='utf-8') as vos_file:
            vos_file.write("label\n")
            for ref in references:
                vos_file.write(f"{ref}\n")
        config.logger.info(f"VOSviewer formatında referanslar kaydedildi: {vosviewer_file}")
        
        with open(pajek_file, 'w', encoding='utf-8') as pajek_f:
            pajek_f.write(f"*Vertices {len(references)}\n")
            for i, ref in enumerate(references, 1):
                pajek_f.write(f'{i} "{ref}"\n')
        config.logger.info(f"Pajek formatında referanslar kaydedildi: {pajek_file}")
    except Exception as e:
        config.logger.error(f"Referanslar analiz formatlarına kaydedilemedi: {e}")


# ---------------------------------------------------------------------
# embedding_module.py (alternative_embedding_module.py)
# ---------------------------------------------------------------------
import os
import pandas as pd
from config_module import config
from openai import OpenAI
from transformers import LlamaTokenizer

def split_text(text, chunk_size=256):
    """
    Metni, her biri chunk_size kelime içeren parçalara böler.
    
    Args:
        text (str): İşlenecek metin.
        chunk_size (int): Her parçanın içerdiği maksimum kelime sayısı (varsayılan: 256).
    
    Returns:
        list: Parçalara ayrılmış metin parçalarının listesi.
    """
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text):
    """
    OpenAI API kullanarak verilen metin için embedding oluşturur.
    Model: "text-embedding-ada-002"
    
    Args:
        text (str): Embedding oluşturulacak metin.
    
    Returns:
        list veya None: Oluşturulan embedding vektörü (örneğin, 1536 boyutlu liste) veya hata durumunda None.
    """
    try:
        client_instance = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client_instance.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        config.logger.error(f"Embedding oluşturulamadı: {e}")
        return None

def fine_tuning_preparation():
    """
    Fine-tuning için veri hazırlığı yapar.
    Eğer "fine_tuning_dataset.csv" mevcutsa, buradan metin verileri alınır,
    aksi halde örnek metinlerle çalışılır.
    
    Her metin, LLaMA Tokenizer kullanılarak tokenize edilir ve token sayıları hesaplanır.
    
    Returns:
        tuple: (summary, tokenized_data)
            - summary: Token sayıları ve istatistiklerini içeren özet metin.
            - tokenized_data: Her metnin tokenize edilmiş verilerinin listesi.
    """
    try:
        if os.path.exists("fine_tuning_dataset.csv"):
            df = pd.read_csv("fine_tuning_dataset.csv")
            texts = df["icerik"].tolist()
        else:
            texts = ["Örnek metin 1", "Örnek metin 2"]
        if not texts:
            config.logger.error("Fine-tuning için yeterli veri bulunamadı.")
            return None
        
        # LLaMA Tokenizer'ı yükle
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        tokenized_data = [tokenizer(text, truncation=True, max_length=512) for text in texts]
        token_counts = [len(item["input_ids"]) for item in tokenized_data]
        summary = (
            f"Toplam kayıt: {len(tokenized_data)}\n"
            f"Ortalama token sayısı: {sum(token_counts)/len(token_counts):.2f}\n"
            f"En fazla token: {max(token_counts)}\n"
            f"En az token: {min(token_counts)}\n"
        )
        return summary, tokenized_data
    except Exception as e:
        config.logger.error(f"Fine-tuning hazırlık hatası: {e}")
        return None


# ---------------------------------------------------------------------
# alternative_embedding_module.py
# ---------------------------------------------------------------------

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from config_module import config

# Global model cache: Model tekrar yüklemelerini önlemek için kullanılır.
_MODEL_CACHE = {}

def get_sentence_transformer(model_env_var, default_model):
    """
    .env dosyasından model adını okur; eğer belirtilmemişse default modeli kullanır.
    Yüklenen model, _MODEL_CACHE içinde saklanır.
    """
    model_name = os.getenv(model_env_var, default_model)
    if model_name not in _MODEL_CACHE:
        config.logger.info(f"SentenceTransformer modeli yükleniyor: {model_name}")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]

def contriever_large_embed(text):
    """
    Contriever-large modelini kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "CONTRIEVER_LARGE_MODEL" değişkeni; varsayılan: "facebook/contriever-large"
    Çıktı boyutu: 768
    """
    model = get_sentence_transformer("CONTRIEVER_LARGE_MODEL", "facebook/contriever-large")
    embedding = model.encode(text)
    return embedding.tolist()

def specter_large_embed(text):
    """
    SPECTER-large modelini kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "SPECTER_LARGE_MODEL"; varsayılan: "allenai-specter-large"
    Çıktı boyutu: 768
    """
    model = get_sentence_transformer("SPECTER_LARGE_MODEL", "allenai-specter-large")
    embedding = model.encode(text)
    return embedding.tolist()

def specter_embed(text):
    """
    SPECTER modelini kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "SPECTER_MODEL"; varsayılan: "allenai-specter"
    Çıktı boyutu: 768
    """
    model = get_sentence_transformer("SPECTER_MODEL", "allenai-specter")
    embedding = model.encode(text)
    return embedding.tolist()

def all_mpnet_base_v2_embed(text):
    """
    all-mpnet-base-v2 modelini kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "ALL_MPNET_BASE_V2_MODEL"; varsayılan: "all-mpnet-base-v2"
    Çıktı boyutu: 768
    """
    model = get_sentence_transformer("ALL_MPNET_BASE_V2_MODEL", "all-mpnet-base-v2")
    embedding = model.encode(text)
    return embedding.tolist()

def paraphrase_mpnet_base_v2_embed(text):
    """
    paraphrase-mpnet-base-v2 modelini kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "PARAPHRASE_MPNET_BASE_V2_MODEL"; varsayılan: "paraphrase-mpnet-base-v2"
    Çıktı boyutu: 768
    """
    model = get_sentence_transformer("PARAPHRASE_MPNET_BASE_V2_MODEL", "paraphrase-mpnet-base-v2")
    embedding = model.encode(text)
    return embedding.tolist()

def stsb_roberta_large_embed(text):
    """
    stsb-roberta-large modelini kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "STSB_ROBERTA_LARGE_MODEL"; varsayılan: "stsb-roberta-large"
    Çıktı boyutu: 1024
    """
    model = get_sentence_transformer("STSB_ROBERTA_LARGE_MODEL", "stsb-roberta-large")
    embedding = model.encode(text)
    return embedding.tolist()

def labse_embed(text):
    """
    LaBSE modelini kullanarak metni embedding'e dönüştürür.
    Model: .env üzerinden "LABSE_MODEL"; varsayılan: "LaBSE"
    Çıktı boyutu: 768
    """
    model = get_sentence_transformer("LABSE_MODEL", "LaBSE")
    embedding = model.encode(text)
    return embedding.tolist()

def universal_sentence_encoder_embed(text):
    """
    Universal Sentence Encoder modelini kullanarak metni embedding'e dönüştürür.
    Model URL: .env üzerinden "USE_MODEL"; varsayılan: "https://tfhub.dev/google/universal-sentence-encoder/4"
    Çıktı boyutu: 512
    """
    if "USE" not in _MODEL_CACHE:
        model_url = os.getenv("USE_MODEL", "https://tfhub.dev/google/universal-sentence-encoder/4")
        config.logger.info(f"Universal Sentence Encoder modeli yükleniyor: {model_url}")
        _MODEL_CACHE["USE"] = hub.load(model_url)
    model = _MODEL_CACHE["USE"]
    embedding = model([text])
    return embedding.numpy()[0].tolist()

def universal_sentence_encoder_lite_embed(text):
    """
    Universal Sentence Encoder Lite modelini kullanarak metni embedding'e dönüştürür.
    Model URL: .env üzerinden "USE_LITE_MODEL"; varsayılan: "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
    Çıktı boyutu: 512
    """
    if "USE_LITE" not in _MODEL_CACHE:
        model_url = os.getenv("USE_LITE_MODEL", "https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        config.logger.info(f"Universal Sentence Encoder Lite modeli yükleniyor: {model_url}")
        _MODEL_CACHE["USE_LITE"] = hub.load(model_url)
    model = _MODEL_CACHE["USE_LITE"]
    embedding = model([text])
    return embedding.numpy()[0].tolist()

# ---------------------------------------------------------------------
# robust_embedding_module.py 
# ---------------------------------------------------------------------

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
# ---------------------------------------------------------------------
# helper_module.py
# ---------------------------------------------------------------------

import os
import psutil
import threading
import re
from config_module import config
from rapidfuzz import fuzz

def memory_usage():
    """
    Bellek kullanımını MB cinsinden hesaplar ve string olarak döndürür.
    Örnek: "123.45 MB"
    """
    return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"

def shorten_title(title, max_length=80):
    """
    Verilen başlığı, belirlenen maksimum uzunluğa göre kısaltır.
    Eğer title uzunluğu max_length'ten büyükse, ilk max_length karakterini döndürür;
    aksi halde orijinal title'ı döndürür.
    """
    return title[:max_length] if len(title) > max_length else title

# Global stack yönetimi için thread-safe kilit
stack_lock = threading.Lock()

def stack_yukle():
    """
    Stack dosyasını okur. Eğer dosya mevcut değilse boş bir set döndürür.
    
    Returns:
        set: Stack'te kayıtlı dosya adlarının seti.
    """
    with stack_lock:
        if os.path.exists(config.STACK_DOSYASI):
            with open(config.STACK_DOSYASI, "r", encoding="utf-8") as f:
                return set(f.read().splitlines())
        return set()

def stack_guncelle(dosya_adi, islem_tipi):
    """
    Stack güncelleme işlemini gerçekleştirir.
    
    Args:
        dosya_adi (str): İşlenen dosyanın adı.
        islem_tipi (str): 'ekle' veya 'sil'. 
            - 'ekle': dosya adını stack'e ekler.
            - 'sil': dosya adını stack'ten çıkarır.
    """
    with stack_lock:
        current_stack = stack_yukle()
        if islem_tipi == "ekle":
            current_stack.add(dosya_adi)
        elif islem_tipi == "sil":
            current_stack.discard(dosya_adi)
        with open(config.STACK_DOSYASI, "w", encoding="utf-8") as f:
            f.write("\n".join(current_stack))

def kalan_dosyalar_oku():
    """
    LOG dosyası (islem_logu.json) içindeki kayıtlı dosyaları okur ve,
    STORAGE_DIR içinde mevcut olan dosyalarla karşılaştırarak işlenmeyi bekleyen dosyaların listesini döndürür.
    
    Returns:
        list: Kalan dosya adlarının listesi.
    """
    log_file = config.LOG_DIR / "islem_logu.json"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            kayitli = f.read().splitlines()
    else:
        kayitli = []
    mevcut = set(os.listdir(config.STORAGE_DIR))
    return [dosya for dosya in kayitli if dosya in mevcut]

def clean_advanced_text(text):
    """
    Metni gelişmiş temizlik adımlarıyla temizler:
      - HTML etiketlerini kaldırır.
      - Markdown etiketlerini kaldırır.
      - Sayfa başı/sonu bilgilerini (örn. "Page 1", "Sayfa 1") temizler.
      - Fazla boşlukları ve gereksiz özel karakterleri temizler.
    
    Args:
        text (str): İşlenecek metin.
    
    Returns:
        str: Temizlenmiş metin.
    """
    # HTML etiketlerini kaldır
    text = re.sub(r'<[^>]+>', ' ', text)
    # Markdown link yapılarını kaldır: [text](url)
    text = re.sub(r'\[[^\]]+\]\([^)]+\)', ' ', text)
    # Sayfa bilgilerini temizle: "Page 1", "Sayfa 1"
    text = re.sub(r'(Page|Sayfa)\s*\d+', ' ', text, flags=re.IGNORECASE)
    # Fazla boşlukları tek boşluk yap
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def fuzzy_match(text1, text2):
    """
    İki metin arasındaki fuzzy matching skorunu hesaplar.
    Rapidfuzz kütüphanesi kullanılarak Levenshtein benzerlik oranını döndürür.
    
    Args:
        text1 (str): İlk metin.
        text2 (str): İkinci metin.
    
    Returns:
        float: Benzerlik skoru (0 ile 100 arasında).
    """
    return fuzz.ratio(text1, text2)
# ---------------------------------------------------------------------
# processing_manager.py
# ---------------------------------------------------------------------
import os
from pathlib import Path
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from config_module import config
from zotero_module import dokuman_id_al, fetch_zotero_metadata
from pdf_processing import extract_text_from_pdf, reflow_columns, map_scientific_sections_extended, detect_columns
from robust_embedding_module import robust_embed_text
from citation_mapping_module import map_citations  # Citation mapping modülündeki ana fonksiyon
from file_save_module import save_citation_file
from helper_module import shorten_title, stack_guncelle

def pdf_txt_isle(dosya_yolu):
    """
    PDF veya TXT dosyalarını işleyerek; 
      - Ham metni çıkarır,
      - reflow_columns ile temizler,
      - Bilimsel bölümleri (map_scientific_sections_extended) ve sütun yapısını (detect_columns) belirler,
      - Tablo ve kaynakça çıkarımını (örneğin, extract_references_enhanced) gerçekleştirir,
      - Robust embedding işlemlerini (robust_embed_text) uygular,
      - Citation Mapping'i (map_citations) çağırır ve elde edilen verileri sonuç sözlüğüne ekler.
    
    İş akışı:
      1. Dosya tipi belirlenir (PDF/TXT).
      2. Metin çıkarılır (PDF için extract_text_from_pdf, TXT için doğrudan okuma).
      3. reflow_columns ile metin temizlenir.
      4. map_scientific_sections_extended ile bilimsel bölümler haritalanır.
      5. detect_columns ile sütun yapısı belirlenir.
      6. (Varsa) tablo ve kaynakça çıkarım işlemleri yapılır.
      7. Temiz metin üzerinden robust embedding işlemi gerçekleştirilir.
      8. Citation Mapping, map_citations fonksiyonu ile çağrılır.
      9. Elde edilen tüm veriler, bir sonuç sözlüğü halinde döndürülür.
      10. İşlem başlangıcında dosya adı stack'e eklenip, işlem sonunda kaldırılır.
    
    Args:
        dosya_yolu (Path): İşlenecek dosyanın yolu.
    
    Returns:
        dict veya None: İşlem başarılı ise dosyaya ait tüm verileri içeren sözlük; hata durumunda None.
    """
    try:
        # Stack'e ekle
        stack_guncelle(dosya_yolu.name, "ekle")
        config.logger.info(f"{dosya_yolu.name} işleme başladı.")
        
        ext = dosya_yolu.suffix.lower()
        if ext == ".pdf":
            ham_metin = extract_text_from_pdf(dosya_yolu, method='pdfminer')
        elif ext == ".txt":
            with open(dosya_yolu, "r", encoding="utf-8") as f:
                ham_metin = f.read()
        else:
            config.logger.error(f"Desteklenmeyen dosya uzantısı: {dosya_yolu}")
            return None
        
        if not ham_metin:
            raise ValueError("Ham metin çıkarılamadı.")
        
        # Reflow ve gelişmiş temizlik (clean_advanced_text fonksiyonu helper_module'da mevcut olabilir)
        temiz_metin = reflow_columns(ham_metin)
        
        # Bilimsel bölümlerin haritalanması
        bolum_haritasi = map_scientific_sections_extended(ham_metin)
        
        # Sütun yapısının belirlenmesi
        sutun_bilgisi = detect_columns(ham_metin)
        
        # (Opsiyonel) Tablo ve kaynakça çıkarımı - bu adım için ilgili fonksiyonlar varsa çağrılabilir.
        # Örneğin: kaynakca = extract_references_enhanced(ham_metin)
        # Burada kaynakça çıkarımı modüler yapıya entegre edilmişse, onun çıktısı da alınabilir.
        kaynakca = []  # Varsayılan boş liste; gerçek çıkarım fonksiyonu eklenebilir.
        tablolar = []   # Benzer şekilde tablolar çıkarımı yapılabilir.
        
        # Robust embedding işlemleri: Temiz metni chunk'lara bölüp her chunk için embedding oluştur.
        # Örneğin, split_text fonksiyonu embedding_module.py içinde bulunuyor.
        from embedding_module import split_text  # İlgili modülden import ediyoruz.
        chunks = split_text(temiz_metin, chunk_size=256)
        embedding_list = []
        for idx, chunk in enumerate(chunks):
            emb = robust_embed_text(chunk, 
                                    pdf_id=shorten_title(dokuman_id_al(dosya_yolu.name) or dosya_yolu.stem, 80),
                                    chunk_index=idx,
                                    total_chunks=len(chunks))
            embedding_list.append(emb)
        
        # Citation Mapping: Temiz metin üzerinden cümle bazında atıf ifadeleri eşleştirmesi yapılır.
        citation_mapping = map_citations(temiz_metin, bibliography=kaynakca, section_info=bolum_haritasi)
        
        # Citation Mapping dosyası kaydedilir
        citation_file = save_citation_file(dosya_yolu.name, citation_mapping)
        
        # Zotero entegrasyonu: Dosya adı üzerinden temel ID alınır, bibliyometrik veriler çekilir.
        file_id = dokuman_id_al(dosya_yolu.name) or dosya_yolu.stem
        file_id = shorten_title(file_id, 80)
        zotero_meta = fetch_zotero_metadata(file_id)
        
        # Sonuç sözlüğü oluşturulur
        result = {
            "dosya": dosya_yolu.name,
            "ham_metin": ham_metin,
            "temiz_metin": temiz_metin,
            "bolum_haritasi": bolum_haritasi,
            "sutun_bilgisi": sutun_bilgisi,
            "tablolar": tablolar,
            "kaynakca": kaynakca,
            "embedding": embedding_list,
            "citation_mapping": citation_mapping,
            "citation_file": str(citation_file) if citation_file else None,
            "zotero_meta": zotero_meta,
            "islem_tarihi": datetime.now().isoformat()
        }
        
        # Stack'ten kaldır
        stack_guncelle(dosya_yolu.name, "sil")
        config.logger.info(f"{dosya_yolu.name} başarıyla işlendi.")
        
        return result
    except Exception as e:
        config.logger.error(f"{dosya_yolu.name} işlenirken hata: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Örnek test: STORAGE_DIR altındaki dosyaları işlemek
    from config_module import config
    import glob
    dosyalar = glob.glob(str(config.STORAGE_DIR / "*.pdf"))  # PDF dosyalarını örnek olarak işleyelim
    for dosya in dosyalar:
        result = pdf_txt_isle(Path(dosya))
        if result:
            print(f"İşlem tamamlandı: {result['dosya']}")
        else:
            print(f"Hata oluştu: {dosya}")

# ---------------------------------------------------------------------
# file_save_module.py
# ---------------------------------------------------------------------

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from config_module import config
from zotero_module import dokuman_id_al
from helper_module import shorten_title
from embedding_module import split_text  # Metni parçalara ayırmak için

# Genel dosya kaydetme fonksiyonları

def save_text_file(directory, filename, content):
    """
    Belirtilen dizine, dosya adıyla metni TXT formatında kaydeder.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

def save_json_file(directory, filename, content):
    """
    Belirtilen dizine, dosya adıyla içerik JSON formatında kaydeder.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    return filepath

# Temiz metin dosyaları kaydetme

def save_clean_text_files(original_filename, clean_text, bib_info):
    """
    Temiz metin içeriğini bibliyometrik bilgi ile birlikte kaydeder.
    TXT dosyası: TEMIZMETIN_DIR/txt/ID.temizmetin.txt
    JSON dosyası: TEMIZMETIN_DIR/json/ID.temizmetin.json
    Bibliyometrik bilgiler, TXT dosyasında "# " yorum satırları olarak, JSON dosyasında "bibinfo" alanı altında eklenir.
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    
    # Bibliyometrik bilgi bloğu oluştur (TXT için)
    bib_lines = []
    if bib_info:
        bib_lines.append("# Bibliyometrik Bilgiler:")
        for key, value in bib_info.items():
            if isinstance(value, list):
                value = "; ".join(value)
            bib_lines.append(f"# {key}: {value}")
        bib_lines.append("")  # Boş satır ekle
    bib_block = "\n".join(bib_lines)
    
    # TXT olarak kaydet
    txt_dir = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin")) / "txt"
    txt_filename = f"{file_id}.temizmetin.txt"
    txt_content = bib_block + "\n" + clean_text
    txt_path = save_text_file(txt_dir, txt_filename, txt_content)
    
    # JSON olarak kaydet
    json_dir = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin")) / "json"
    json_filename = f"{file_id}.temizmetin.json"
    json_content = {
        "bibinfo": bib_info,
        "content": clean_text,
        "olusturma_tarihi": datetime.now().isoformat()
    }
    json_path = save_json_file(json_dir, json_filename, json_content)
    
    return txt_path, json_path

# Kaynakça dosyaları kaydetme

def save_references_files(original_filename, references, bib_info):
    """
    Kaynakça verilerini, bibliyometrik bilgi ile birlikte aşağıdaki formatlarda kaydeder:
      - TXT: TEMIZ_KAYNAKCA_DIZIN/txt/ID.kaynakca.txt
      - JSON: TEMIZ_KAYNAKCA_DIZIN/json/ID.kaynakca.json
      - VOSviewer: TEMIZ_KAYNAKCA_DIZIN/vos/ID.vos.kaynak.txt
      - Pajek: TEMIZ_KAYNAKCA_DIZIN/paj/ID.pjk.kaynak.paj
      - CSV: TEMIZ_KAYNAKCA_DIZIN/csv/ID.kaynakca.csv
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    
    base_dir = Path(os.getenv("TEMIZ_KAYNAKCA_DIZIN", "temizkaynakca"))
    
    # TXT kaydı
    txt_dir = base_dir / "txt"
    txt_filename = f"{file_id}.kaynakca.txt"
    bib_lines = []
    if bib_info:
        bib_lines.append("# Bibliyometrik Bilgiler:")
        for key, value in bib_info.items():
            if isinstance(value, list):
                value = "; ".join(value)
            bib_lines.append(f"# {key}: {value}")
        bib_lines.append("")
    txt_content = "\n".join(bib_lines) + "\n" + "\n".join(references)
    txt_path = save_text_file(txt_dir, txt_filename, txt_content)
    
    # JSON kaydı
    json_dir = base_dir / "json"
    json_filename = f"{file_id}.kaynakca.json"
    json_content = {
        "bibinfo": bib_info,
        "references": references,
        "olusturma_tarihi": datetime.now().isoformat()
    }
    json_path = save_json_file(json_dir, json_filename, json_content)
    
    # VOSviewer formatı
    vos_dir = base_dir / "vos"
    vos_filename = f"{file_id}.vos.kaynak.txt"
    vos_content = "label\n" + "\n".join(references)
    vos_path = save_text_file(vos_dir, vos_filename, vos_content)
    
    # Pajek formatı
    paj_dir = base_dir / "paj"
    paj_filename = f"{file_id}.pjk.kaynak.paj"
    paj_content = f"*Vertices {len(references)}\n"
    for i, ref in enumerate(references, 1):
        paj_content += f'{i} "{ref}"\n'
    paj_path = save_text_file(paj_dir, paj_filename, paj_content)
    
    # CSV formatı
    csv_dir = base_dir / "csv"
    csv_filename = f"{file_id}.kaynakca.csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / csv_filename
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Reference"])  # Başlık satırı
            for ref in references:
                writer.writerow([ref])
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {e}")
    
    return {
        "txt": txt_path,
        "json": json_path,
        "vos": vos_path,
        "paj": paj_path,
        "csv": str(csv_path)
    }

# Tablo dosyaları kaydetme

def save_table_files(original_filename, table_data_list):
    """
    Bir yayına ait tabloları, belirlenen formatlarda kaydeder:
      - JSON: TEMIZ_TABLO_DIZIN/json/ID_tabloX.json
      - CSV: TEMIZ_TABLO_DIZIN/csv/ID_tabloX.csv
      - Excel: TEMIZ_TABLO_DIZIN/excel/ID_tabloX.xlsx
    Her tablo, numaralandırılarak kaydedilir.
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    
    base_dir = Path(os.getenv("TEMIZ_TABLO_DIZIN", "temiztablo"))
    results = []
    for i, table_data in enumerate(table_data_list, start=1):
        # JSON olarak kaydet
        json_dir = base_dir / "json"
        json_filename = f"{file_id}_tablo{i}.json"
        json_path = save_json_file(json_dir, json_filename, {"table": table_data, "olusturma_tarihi": datetime.now().isoformat()})
        
        # CSV olarak kaydet
        csv_dir = base_dir / "csv"
        csv_filename = f"{file_id}_tablo{i}.csv"
        csv_path = csv_dir / csv_filename
        csv_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in table_data:
                    writer.writerow(row)
        except Exception as e:
            config.logger.error(f"CSV dosyası kaydedilemedi: {e}")
        
        # Excel olarak kaydet
        excel_dir = base_dir / "excel"
        excel_filename = f"{file_id}_tablo{i}.xlsx"
        excel_path = excel_dir / excel_filename
        excel_dir.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd
            if len(table_data) > 1:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
            else:
                df = pd.DataFrame(table_data)
            df.to_excel(excel_path, index=False)
        except Exception as e:
            config.logger.error(f"Excel dosyası kaydedilemedi: {e}")
        
        results.append({
            "json": json_path,
            "csv": str(csv_path),
            "excel": str(excel_path)
        })
    return results

# Embedding dosyası kaydetme

def save_embedding_file(original_filename, embedding_text, chunk_index):
    """
    Her PDF/TXT dosyasının embedding verilerini, belirlenen formatta kaydeder.
    Dosya adı: ID_{chunk_index}.embed.txt
    Kaydetme yeri: EMBEDDING_PARCA_DIR altında.
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    base_dir = Path(os.getenv("EMBEDDING_PARCA_DIZIN", "embedingparca"))
    filename = f"{file_id}_{chunk_index}.embed.txt"
    filepath = save_text_file(base_dir, filename, embedding_text)
    return filepath

# Chunk'lanmış ham metin dosyaları kaydetme

def save_chunked_text_files(original_filename, full_text, chunk_size=256):
    """
    Büyük metinler, chunk_size kadar kelimeye bölünür ve her parça ayrı .txt dosyası olarak kaydedilir.
    Dosya adı: ID_<chunk_index>.txt
    Kaydetme yeri: TEMIZMETIN_DIR/chunks
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    chunks = split_text(full_text, chunk_size=chunk_size)
    base_dir = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin")) / "chunks"
    saved_files = []
    for i, chunk in enumerate(chunks):
        filename = f"{file_id}_{i}.txt"
        path = save_text_file(base_dir, filename, chunk)
        saved_files.append(str(path))
    return saved_files

# ---------------------------------------------------------------------
# citation_mapping_module.py
# ---------------------------------------------------------------------
import re
import json
from datetime import datetime
from config_module import config
from rapidfuzz import fuzz
import spacy
from helper_module import shorten_title

# SpaCy modelini yükle (en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    config.logger.error("SpaCy modeli yüklenemedi. Lütfen 'python -m spacy download en_core_web_sm' komutunu çalıştırın.")
    raise

def split_into_sentences(text):
    """
    Metni cümlelere böler ve her cümleye sıra numarası ekler.
    
    Args:
        text (str): İşlenecek metin.
        
    Returns:
        list: Her biri (cümle_no, cümle) tuple'su içeren liste.
    """
    # Basit cümle bölme: nokta, ünlem veya soru işareti sonrası boşluk
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [(i, s.strip()) for i, s in enumerate(sentences) if s.strip()]

def extract_citations_from_sentence(sentence):
    """
    Verilen cümledeki atıf ifadelerini regex, fuzzy matching ve spaCy NER kullanarak tespit eder.
    
    Args:
        sentence (str): İşlenecek cümle.
        
    Returns:
        list: Tespit edilen atıf ifadelerinin listesi (örn. ["(Smith et al., 2020)", "[2]"]).
    """
    citations = []
    # Regex: Parantez içi atıflar örn. (Smith et al., 2020)
    paren_pattern = r"\(([A-Za-z\s\.,&\-]+?\d{4}[a-zA-Z]?)\)"
    paren_candidates = re.findall(paren_pattern, sentence)
    
    for candidate in paren_candidates:
        candidate = candidate.strip()
        doc = nlp(candidate)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        # Fuzzy matching: örneğin "et al.," ifadesine yakınlık kontrolü
        if persons and dates:
            similarity = fuzz.partial_ratio(candidate.lower(), "et al.,")
            if similarity >= 50:
                citations.append(f"({candidate})")
            else:
                citations.append(f"({candidate})")
    
    # Regex: Köşeli parantez içi atıflar örn. [2] veya [1, 2, 3]
    bracket_pattern = r"\[([\d,\s]+)\]"
    bracket_candidates = re.findall(bracket_pattern, sentence)
    for candidate in bracket_candidates:
        candidate = candidate.strip()
        if re.fullmatch(r"[\d,\s]+", candidate):
            citations.append(f"[{candidate}]")
    return citations

def match_citation_with_references(citation_marker, references):
    """
    Tespit edilen atıf ifadesini, PDF'nin kaynakça listesindeki referanslarla eşleştirmeye çalışır.
    İlk aşamada tam eşleşme, bulunamazsa fuzzy matching ile benzerlik hesaplaması yapılır.
    
    Args:
        citation_marker (str): Örneğin, "(Smith et al., 2020)" veya "[2]".
        references (list): Kaynakça referanslarının listesi (her biri bir string).
        
    Returns:
        dict veya None: Eşleşen referansın detayları (örneğin, {'ref_no': 1, 'text': ...}) veya None.
    """
    # İlk aşama: tam eşleşme
    normalized_marker = citation_marker.strip("()[]").lower()
    for idx, ref in enumerate(references, start=1):
        if normalized_marker in ref.lower():
            return {"ref_no": idx, "text": ref}
    # İkinci aşama: fuzzy matching
    best_score = 0
    best_match = None
    for idx, ref in enumerate(references, start=1):
        score = fuzz.ratio(normalized_marker, ref.lower())
        if score > best_score:
            best_score = score
            best_match = {"ref_no": idx, "text": ref}
    if best_score >= 60:
        return best_match
    return None

def map_citations(clean_text, bibliography, section_info):
    """
    Temiz metin içindeki tüm cümleleri işleyerek, cümle numarası, cümle metni,
    tespit edilen atıf ifadesi, eşleşen referans ve bilimsel bölüm bilgisini içeren bir yapı oluşturur.
    
    Args:
        clean_text (str): Temiz metin.
        bibliography (list): PDF'nin kaynakça referansları.
        section_info (dict): Bölümlerin haritalanması bilgisi (örn. {"Introduction": (start, end), ...}).
    
    Returns:
        dict: Citation mapping verisini içeren sözlük.
    """
    mapping = {"citations": []}
    sentences = split_into_sentences(clean_text)
    for sent_no, sentence in sentences:
        citations = extract_citations_from_sentence(sentence)
        for citation in citations:
            matched = match_citation_with_references(citation, bibliography)
            # Bölüm bilgisini belirlemek için: Eğer cümle numarası, haritalandığı bölüme denk geliyorsa ekle
            section = None
            for sec, span in section_info.items():
                if span and span[0] <= sent_no <= span[1]:
                    section = sec
                    break
            mapping["citations"].append({
                "sentence_no": sent_no,
                "sentence_text": sentence,
                "citation_marker": citation,
                "matched_reference": matched,
                "section": section
            })
    return mapping

def save_citation_mapping(pdf_id, citation_mapping):
    """
    Oluşturulan Citation Mapping verilerini, bibliyometrik bilgilerle birlikte "pdf_id.citation.json"
    dosyası olarak, CITATIONS_DIR altındaki klasörde saklar.
    
    Args:
        pdf_id (str): PDF'nin temel kimliği.
        citation_mapping (dict): Elde edilen citation mapping verisi.
    
    Returns:
        Path veya None: Oluşturulan dosyanın yolu.
    """
    file_id = shorten_title(pdf_id, max_length=80)
    target_dir = config.CITATIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{file_id}.citation.json"
    target_path = target_dir / filename
    citation_mapping["olusturma_tarihi"] = datetime.now().isoformat()
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(citation_mapping, f, ensure_ascii=False, indent=2)
        config.logger.info(f"Citation mapping dosyası kaydedildi: {target_path}")
        return target_path
    except Exception as e:
        config.logger.error(f"Citation mapping dosyası kaydedilirken hata: {e}")
        return None

# Test bölümü (modülü direkt çalıştırmak için)
if __name__ == "__main__":
    test_text = (
        "This study utilized (Smith et al., 2020) method which has been cited. "
        "Other studies, such as [2], support these findings. Additional details in (Doe et al., 2019) are provided."
    )
    test_bibliography = [
        "Smith, J., et al. (2020). Title of the paper. Journal Name. DOI:10.xxxx/xxxxxx",
        "Doe, J., et al. (2019). Another paper title. Another Journal. DOI:10.xxxx/yyyyyy"
    ]
    test_section_info = {
        "Introduction": (0, 2),
        "Methods": (3, 5),
        "Results": (6, 10)
    }
    mapping = map_citations(test_text, test_bibliography, test_section_info)
    print(json.dumps(mapping, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------
# gui_module.py
# ---------------------------------------------------------------------

import os
import json
import pandas as pd
import customtkinter as ctk
import tkinter.messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaTokenizer
from config_module import config
from helper_module import shorten_title
from file_save_module import save_citation_file
from citation_mapping_module import load_citation_mapping  # Fonksiyon, CITATION dosyasını yüklemek için

# Örnek: Kümeleme analizi fonksiyonunun basit bir implementasyonu
def cluster_analysis_from_chromadb(kume_sonuclari, n_clusters=5, output_dir="cluster_results"):
    try:
        vectorizer = TfidfVectorizer(max_features=1000)
        texts = [record["icerik"] for record in kume_sonuclari]
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        for i, cluster in enumerate(kmeans.labels_):
            kume_sonuclari[i]['cluster'] = int(cluster)
        os.makedirs(output_dir, exist_ok=True)
        cluster_data = pd.DataFrame({
            "Index": list(range(len(texts))),
            "Cluster": kmeans.labels_,
            "Summary": [text[:100] + "..." for text in texts]
        })
        cluster_file = os.path.join(output_dir, "cluster_results.csv")
        cluster_data.to_csv(cluster_file, index=False, encoding='utf-8')
        config.logger.info(f"Kümeleme analizi tamamlandı, sonuçlar {cluster_file} dosyasına kaydedildi.")
    except Exception as e:
        config.logger.error(f"Kümeleme analizi hatası: {str(e)}")

# Ek Özellikler paneli
class AnalizPaneli(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master  # master, AnaArayuz nesnesi
        self.kume_btn = ctk.CTkButton(self, text="Kümeleme Analizi Yap", command=self._kumeleme_baslat)
        self.kume_btn.pack(pady=10)

    def _kumeleme_baslat(self):
        try:
            if not self.master.kume_sonuclari:
                ctk.messagebox.showwarning("Uyarı", "Kümeleme için yeterli veri mevcut değil.")
                return
            n_input = self.klm_entry.get() if hasattr(self, "klm_entry") else "5"
            n_clusters = int(n_input) if n_input.isdigit() else 5
            cluster_analysis_from_chromadb(self.master.kume_sonuclari, n_clusters=n_clusters, output_dir="cluster_results")
            self.master.sonuc_text.delete("1.0", "end")
            self.master.sonuc_text.insert("1.0", f"Kümeleme analizi tamamlandı. Sonuçlar 'cluster_results/cluster_results.csv' dosyasına kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Kümeleme analizi sırasında hata: {e}")

# Ek özellikler için GUI penceresi
class AdditionalFeaturesGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ek Özellikler")
        self.geometry("950x800")
        self._arayuzu_hazirla()

    def _arayuzu_hazirla(self):
        # Embedding Arama Bölümü
        self.ara_label = ctk.CTkLabel(self, text="Embedding Arama", font=("Arial", 16))
        self.ara_label.pack(pady=10)
        self.ara_entry = ctk.CTkEntry(self, placeholder_text="Arama sorgusunu girin...")
        self.ara_entry.pack(pady=5, padx=20, fill="x")
        self.ara_button = ctk.CTkButton(self, text="Ara", command=self.embedding_arama)
        self.ara_button.pack(pady=5)

        # Kümeleme Analizi Yeniden Çalıştırma
        self.klm_label = ctk.CTkLabel(self, text="Kümeleme Analizi Yeniden Çalıştırma", font=("Arial", 16))
        self.klm_label.pack(pady=20)
        self.klm_entry = ctk.CTkEntry(self, placeholder_text="Küme sayısı (varsayılan: 5)")
        self.klm_entry.pack(pady=5, padx=20, fill="x")
        self.klm_button = ctk.CTkButton(self, text="Analizi Başlat", command=self.kumelenme_analizi)
        self.klm_button.pack(pady=5)

        # Fine-Tuning Veri Seti Dışa Aktarma
        self.ft_label = ctk.CTkLabel(self, text="Fine-Tuning Veri Seti Dışa Aktar", font=("Arial", 16))
        self.ft_label.pack(pady=20)
        self.ft_button = ctk.CTkButton(self, text="Veri Setini Dışa Aktar", command=self.export_fine_tuning_dataset)
        self.ft_button.pack(pady=5)

        # Fine-Tuning Hazırlık (LLaMA Tokenization)
        self.ft_prep_label = ctk.CTkLabel(self, text="Fine-Tuning Hazırlık (LLaMA Tokenization)", font=("Arial", 16))
        self.ft_prep_label.pack(pady=20)
        self.ft_prep_button = ctk.CTkButton(self, text="Fine-Tuning Başlat", command=self.fine_tuning_preparation_view)
        self.ft_prep_button.pack(pady=5)

        # Atıf Zinciri Görüntüleme
        self.citation_label = ctk.CTkLabel(self, text="Atıf Zinciri Görüntüleme", font=("Arial", 16))
        self.citation_label.pack(pady=20)
        self.citation_entry = ctk.CTkEntry(self, placeholder_text="Atıf zinciri için PDF dosya adını girin...")
        self.citation_entry.pack(pady=5, padx=20, fill="x")
        self.citation_button = ctk.CTkButton(self, text="Atıf Zincirini Görüntüle", command=self.citation_chain_view)
        self.citation_button.pack(pady=5)

        # Veri Tarama ve Sorgulama
        self.search_label = ctk.CTkLabel(self, text="Veri Tarama ve Sorgulama", font=("Arial", 16))
        self.search_label.pack(pady=20)
        self.search_entry = ctk.CTkEntry(self, placeholder_text="Sorgu girin...")
        self.search_entry.pack(pady=5, padx=20, fill="x")
        self.search_button = ctk.CTkButton(self, text="Sorgulama Yap", command=self.data_search)
        self.search_button.pack(pady=5)

        # Sonuç Görselleştirme Alanı
        self.sonuc_text = ctk.CTkTextbox(self, width=900, height=300)
        self.sonuc_text.pack(pady=20, padx=20)

    def embedding_arama(self):
        query = self.ara_entry.get()
        if not query:
            ctk.messagebox.showwarning("Uyarı", "Lütfen bir arama sorgusu girin.")
            return
        if not self.master.kume_sonuclari:
            ctk.messagebox.showwarning("Uyarı", "Arama için yeterli veri mevcut değil.")
            return
        try:
            texts = [record["icerik"] for record in self.master.kume_sonuclari]
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            sims = cosine_similarity(X, query_vec).flatten()
            top_indices = sims.argsort()[-10:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    "Index": idx,
                    "Benzerlik": round(sims[idx], 4),
                    "Özet": texts[idx][:100] + "..."
                })
            df_results = pd.DataFrame(results)
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", df_results.to_string(index=False))
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Embedding arama sırasında hata: {e}")

    def kumelenme_analizi(self):
        n = self.klm_entry.get()
        n_clusters = int(n) if n.isdigit() else 5
        try:
            from processing_manager import cluster_analysis_from_chromadb
            cluster_analysis_from_chromadb(self.master.kume_sonuclari, n_clusters=n_clusters, output_dir="cluster_results")
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", "Kümeleme analizi tamamlandı. Sonuçlar 'cluster_results/cluster_results.csv' dosyasına kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Kümeleme analizi sırasında hata: {e}")

    def export_fine_tuning_dataset(self):
        if not self.master.kume_sonuclari:
            ctk.messagebox.showwarning("Uyarı", "Fine-tuning için veri seti oluşturulacak veri bulunamadı.")
            return
        try:
            df = pd.DataFrame(self.master.kume_sonuclari)
            output_file = "fine_tuning_dataset.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", f"Fine-tuning veri seti '{output_file}' olarak kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Veri seti dışa aktarılırken hata: {e}")

    def fine_tuning_preparation_view(self):
        try:
            if os.path.exists("fine_tuning_dataset.csv"):
                df = pd.read_csv("fine_tuning_dataset.csv")
                texts = df["icerik"].tolist()
            else:
                texts = [record["icerik"] for record in self.master.kume_sonuclari if "icerik" in record]
            if not texts:
                ctk.messagebox.showwarning("Uyarı", "Fine-tuning için yeterli veri bulunamadı.")
                return
            tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
            tokenized_data = [tokenizer(text, truncation=True, max_length=512) for text in texts]
            token_counts = [len(item["input_ids"]) for item in tokenized_data]
            summary = (
                f"Toplam kayıt: {len(tokenized_data)}\n"
                f"Ortalama token sayısı: {sum(token_counts)/len(token_counts):.2f}\n"
                f"En fazla token: {max(token_counts)}\n"
                f"En az token: {min(token_counts)}\n"
            )
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", "Fine-Tuning Hazırlık Sonuçları:\n" + summary)
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Fine-tuning hazırlık sırasında hata: {e}")

    def citation_chain_view(self):
        """
        Kullanıcının girdiği PDF dosya adından, ilgili Citation Mapping dosyasını okur ve içeriği GUI'de görüntüler.
        """
        pdf_name = self.citation_entry.get().strip()
        if not pdf_name:
            ctk.messagebox.showwarning("Uyarı", "Lütfen görüntülenecek PDF dosya adını girin.")
            return
        # Temel dosya ID'sini belirleyin (dokuman_id_al ve shorten_title kullanılarak)
        from zotero_module import dokuman_id_al
        file_id = dokuman_id_al(pdf_name)
        if not file_id:
            file_id = pdf_name.split('.')[0]
        file_id = shorten_title(file_id, 80)
        citation_file = f"{file_id}.citation.json"
        citation_path = config.CITATIONS_DIR / citation_file
        if not citation_path.exists():
            ctk.messagebox.showwarning("Uyarı", f"Citation Mapping dosyası bulunamadı: {citation_path}")
            return
        try:
            with open(citation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # JSON verisini düzenli bir biçimde görüntüleyelim
            pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", pretty_json)
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Citation Mapping görüntülenirken hata: {e}")

    def data_search(self):
        query = self.search_entry.get()
        if not query:
            ctk.messagebox.showwarning("Uyarı", "Lütfen bir sorgu girin.")
            return
        try:
            texts = [result["icerik"] for result in self.master.kume_sonuclari]
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            sims = cosine_similarity(X, query_vec).flatten()
            top_indices = sims.argsort()[-10:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    "Index": idx,
                    "Benzerlik": round(sims[idx], 4),
                    "Özet": texts[idx][:100] + "..."
                })
            df_results = pd.DataFrame(results)
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", df_results.to_string(index=False))
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Veri tarama/sorgulama sırasında hata: {e}")

# Ana Arayüz
class AnaArayuz(ctk.CTk):
    def __init__(self, islem_yoneticisi):
        super().__init__()
        self.islem_yoneticisi = islem_yoneticisi
        self.kume_sonuclari = []  # İşlenen dosya sonuçlarının saklanacağı liste
        self.title("Zotero Entegre PDF/TXT İşleyici")
        self.geometry("1200x800")
        self._arayuzu_hazirla()

    def _arayuzu_hazirla(self):
        # Sol tarafta dosya listesi (opsiyonel)
        self.dosya_listesi = ctk.CTkListbox(self, width=400)
        self.dosya_listesi.pack(side="left", fill="both", padx=10, pady=10)
        # Üst kısımda işlem başlatma butonu
        self.baslat_btn = ctk.CTkButton(self, text="İşlemi Başlat", command=self._islem_baslat)
        self.baslat_btn.pack(side="top", padx=5, pady=5)
        # Sağ tarafta ek özellikler paneli
        self.ek_panel = AnalizPaneli(self)
        self.ek_panel.pack(side="right", fill="both", padx=10, pady=10)
        # Alt tarafta sonuçların görüntüleneceği metin kutusu
        self.sonuc_text = ctk.CTkTextbox(self, width=900, height=300)
        self.sonuc_text.pack(pady=20, padx=20)

    def _islem_baslat(self):
        from processing_manager import pdf_txt_isle
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        try:
            dosyalar = [os.path.join(config.STORAGE_DIR, f) for f in os.listdir(config.STORAGE_DIR)]
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = {executor.submit(pdf_txt_isle, Path(dosya)): dosya for dosya in dosyalar}
                for future in as_completed(futures):
                    dosya = futures[future]
                    try:
                        result = future.result()
                        if result:
                            config.logger.info(f"{dosya} başarıyla işlendi")
                            self.kume_sonuclari.append(result)
                    except Exception as e:
                        config.logger.error(f"{dosya} işlenirken hata: {str(e)}")
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", "Dosya işlemleri tamamlandı.\n")
        except Exception as e:
            config.logger.error(f"Ana iş akışı hatası: {str(e)}", exc_info=True)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from processing_manager import IslemYoneticisi
    islem_yoneticisi = IslemYoneticisi()
    # STORAGE_DIR içerisindeki toplam dosya sayısını güncelledik
    islem_yoneticisi.sayaçlar['toplam'] = len(os.listdir(config.STORAGE_DIR))
    
    try:
        arayuz = AnaArayuz(islem_yoneticisi)
        arayuz.mainloop()
    except Exception as e:
        config.logger.critical(f"Ana uygulama hatası: {str(e)}", exc_info=True)
    finally:
        print("\nİstatistikler:")
        print(f"Toplam Dosya: {islem_yoneticisi.sayaçlar['toplam']}")
        print(f"Başarılı: {islem_yoneticisi.sayaçlar['başarılı']}")
        print(f"Hatalı: {islem_yoneticisi.sayaçlar['hata']}")

# ---------------------------------------------------------------------
# main.py
# ---------------------------------------------------------------------

import os
import multiprocessing
from config_module import config
from processing_manager import IslemYoneticisi
from gui_module import AnaArayuz

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows için gerekli
    # İşlem yöneticisini oluştur ve STORAGE_DIR içerisindeki toplam dosya sayısını sayaçlara aktar
    islem_yoneticisi = IslemYoneticisi()
    islem_yoneticisi.sayaçlar['toplam'] = len(os.listdir(config.STORAGE_DIR))
    
    try:
        # Ana GUI'yi başlat
        arayuz = AnaArayuz(islem_yoneticisi)
        arayuz.mainloop()
    except Exception as e:
        config.logger.critical(f"Ana uygulama hatası: {e}", exc_info=True)
    finally:
        print("\nİstatistikler:")
        print(f"Toplam Dosya: {islem_yoneticisi.sayaçlar.get('toplam', 0)}")
        print(f"Başarılı: {islem_yoneticisi.sayaçlar.get('başarılı', 0)}")
        print(f"Hatalı: {islem_yoneticisi.sayaçlar.get('hata', 0)}")
