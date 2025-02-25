# İşte güncellenmiş ve eksiksiz config_module.py:

# 🔹 Tüm çevresel değişkenler .env'den yükleniyor.
# 🔹 Loglama sistemi tam entegre.
# 🔹 Gerekli dizinler otomatik oluşturuluyor.
# 🔹 PDF işleme yöntemi (pdfplumber, pdfminer) .env'den okunuyor.
# 🔹 Bellek yönetimi ve hata ayıklama desteği var.
# Bu Güncellenmiş Versiyonda Neler Değişti?
# ✔ Eksik dizinler eklendi: CITATIONS_DIR, TEMP_DIR
# ✔ PDF metin çıkarma yöntemi artık .env'den okunuyor.
# ✔ Bellek kullanımını hesaplayan bellek_kullanimi() fonksiyonu eklendi.
# ✔ Tüm hata logları islem.log dosyasına kaydediliyor.
# ✔ Kritik işlemler konsola ve log dosyasına yazdırılıyor.


import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Ortam değişkenlerini yükle
load_dotenv("C:/Users/mete/Zotero/zotasistan/.env")

class Yapilandirma:
    def __init__(self):
        # 📂 Dizin Yapılandırmaları
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", "success"))
        self.CITATIONS_DIR = Path(os.getenv("CITATIONS_DIR", "citations"))
        self.TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

        # 📝 OpenAI API Anahtarı
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # 🔗 Zotero API Anahtarları
        self.ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
        self.ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")

        # 🗂️ Varsayılan PDF Metin Çıkarma Yöntemi
        self.PDF_TEXT_EXTRACTION_METHOD = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()

        # 📊 Loglama Sistemi
        self._dizinleri_hazirla()
        self._loglama_ayarla()

    def _dizinleri_hazirla(self):
        """📂 Gerekli dizinleri oluşturur."""
        for d in [self.STORAGE_DIR, self.SUCCESS_DIR, self.CITATIONS_DIR, self.TEMP_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _loglama_ayarla(self):
        """📊 Loglama sistemini başlatır."""
        self.logger = logging.getLogger("ZoteroProcessor")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # 📄 Logları dosyaya kaydet
        file_handler = logging.FileHandler(self.LOG_DIR / "islem.log", encoding="utf-8")
        file_handler.setFormatter(formatter)

        # 🖥️ Logları konsola yazdır
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def bellek_kullanimi(self):
        """💾 Bellek kullanımını MB cinsinden döndürür."""
        import psutil
        return f"Bellek Kullanımı: {psutil.virtual_memory().used / (1024 ** 2):.2f} MB"

# 🌍 Yapılandırmayı başlat
config = Yapilandirma()


# # Gerekli kütüphaneleri içe aktar
# import os
# import re
# import json
# import fitz  # PyMuPDF
# import shutil
# import chromadb
# import logging
# import threading
# import requests
# import pandas as pd
# import multiprocessing
# import customtkinter as ctk
# from openai import OpenAI
# from dotenv import load_dotenv
# from pathlib import Path
# from tqdm import tqdm
# from datetime import datetime
# from transformers import LlamaTokenizer
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity
# import psutil

# # Ortam değişkenlerini yükle
# load_dotenv("C:/Users/mete/Zotero/zotasistan/.env")

# class Yapilandirma:
#     def __init__(self):
#         self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR"))
#         self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR"))
#         self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#         self.ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
#         self.ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
#         self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
#         self.STACK_DOSYASI = self.LOG_DIR / "islem.stack"
        
#         self._dizinleri_hazirla()
#         self._loglama_ayarla()
        
#     def _dizinleri_hazirla(self):
#         """Gerekli dizinleri oluştur"""
#         for d in [self.STORAGE_DIR, self.SUCCESS_DIR, self.LOG_DIR]:
#             d.mkdir(parents=True, exist_ok=True)
            
#     def _loglama_ayarla(self):
#         """Loglama sistemini başlat"""
#         self.logger = logging.getLogger('ZoteroProcessor')
#         self.logger.setLevel(logging.DEBUG)
        
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
#         # Dosya log handler
#         file_handler = logging.FileHandler(self.LOG_DIR / 'islem.log', encoding='utf-8')
#         file_handler.setFormatter(formatter)
        
#         # Konsol log handler
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
        
#         self.logger.addHandler(file_handler)
#         self.logger.addHandler(console_handler)

# config = Yapilandirma()

# class ZoteroEntegratoru:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
#         self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    
#     def meta_veri_al(self, item_key):
#         try:
#             response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
#             return response.json() if response.status_code == 200 else None
#         except Exception as e:
#             config.logger.error(f"Zotero API hatası: {str(e)}")
#             return None

#     def referanslari_analiz_et(self, referans_listesi):
#         try:
#             analiz_sonuc = []
#             for referans in referans_listesi:
#                 yazar = re.search(r'^([A-Za-z]+)', referans)
#                 analiz_sonuc.append({
#                     'orijinal': referans,
#                     'yazar': yazar.group(1) if yazar else 'Bilinmeyen'
#                 })
#             return analiz_sonuc
#         except Exception as e:
#             config.logger.error(f"Referans analiz hatası: {str(e)}")
#             return []

# class IslemYoneticisi:
#     def __init__(self):
#         self.stack_lock = threading.Lock()
#         self.kume_sonuclari = []
#         self.sayaçlar = {
#             'toplam': 0,
#             'başarılı': 0,
#             'hata': 0
#         }
#         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
#         self.koleksiyon = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
#         self.zotero_koleksiyon = self.chroma_client.get_or_create_collection(name="zotero_meta")
#         self.zotero = ZoteroEntegratoru()
    
#     def pdf_isle(self, dosya_yolu):
#         try:
#             self.stack_guncelle(dosya_yolu.name, "ekle")
#             ham_metin = self._pdf_metin_cikar(dosya_yolu)
#             if not ham_metin:
#                 raise ValueError("Metin çıkarılamadı")
#             temiz_metin = self._metni_temizle(ham_metin)
#             embeddingler = self._embedding_olustur(temiz_metin)
#             self._veritabanina_kaydet(dosya_yolu.name, embeddingler, temiz_metin)
#             self.sayaçlar['başarılı'] += 1
#             self.stack_guncelle(dosya_yolu.name, "sil")
#             return True
#         except Exception as e:
#             self.sayaçlar['hata'] += 1
#             config.logger.error(f"{dosya_yolu.name} işlenemedi: {str(e)}")
#             return False
    
#     def _pdf_metin_cikar(self, dosya_yolu):
#         try:
#             with fitz.open(dosya_yolu) as doc:
#                 return "\n\n".join([page.get_text("text", sort=True) for page in doc])
#         except Exception as e:
#             config.logger.error(f"PDF okuma hatası: {dosya_yolu}", exc_info=True)
#             return None

# class AnaArayuz(ctk.CTk):
#     def __init__(self, islem_yoneticisi):
#         super().__init__()
#         self.islem_yoneticisi = islem_yoneticisi
#         self.title("Zotero Entegre PDF İşleyici")
#         self.geometry("1200x800")
#         self._arayuzu_hazirla()
    
#     def _arayuzu_hazirla(self):
#         self.dosya_listesi = ctk.CTkListbox(self, width=400)
#         self.dosya_listesi.pack(side="left", fill="both", padx=10, pady=10)
#         self.baslat_btn = ctk.CTkButton(self, text="İşlemi Başlat", command=self._islem_baslat)
#         self.baslat_btn.pack(side="top", padx=5, pady=5)
    
#     def _islem_baslat(self):
#         with ProcessPoolExecutor() as executor:
#             dosyalar = [config.STORAGE_DIR / f for f in os.listdir(config.STORAGE_DIR)]
#             futures = {executor.submit(self.islem_yoneticisi.pdf_isle, dosya): dosya for dosya in dosyalar}
#             for future in tqdm(as_completed(futures), total=len(futures)):
#                 dosya = futures[future]
#                 try:
#                     sonuc = future.result()
#                     if sonuc:
#                         config.logger.info(f"{dosya.name} başarıyla işlendi")
#                 except Exception as e:
#                     config.logger.error(f"{dosya.name} işlenirken hata: {str(e)}")

# if __name__ == '__main__':
#     islem_yoneticisi = IslemYoneticisi()
#     arayuz = AnaArayuz(islem_yoneticisi)
#     arayuz.mainloop()


# Bu modülde yapılan güncellemeler şunlardır:

# PDF Metin Çıkarma Yöntemi: .env dosyasından PDF_TEXT_EXTRACTION_METHOD değişkenini okur (pdfplumber varsayılan).
# Alternatif Embedding Modelleri: .env üzerinden farklı embedding modelleri seçilebilir hale getirildi.
# Büyük Dosya İşleme: LARGE_FILE_SPLIT_SIZE tanımlandı.
# Regex Güncellemeleri: Bilimsel bölümleri ve tablo tespitini daha sağlam hale getiren regex desenleri eklendi.
# Paralel İşlem ve Hata Yönetimi: MAX_RETRIES ve BACKOFF_FACTOR hata yönetimi için eklendi.







# # Aşağıda, "config_module.py" dosyasının güncellenmiş halini bulabilirsiniz. 
# # Bu modül, ortam değişkenlerini (ör. STORAGE_DIR, SUCCESS_DIR, TEMIZMETIN_DIR, 
# #                                 TEMIZ_TABLO_DIZIN, TEMIZ_KAYNAKCA_DIZIN, EMBEDDING_PARCA_DIZIN, 
# #                                 CITATIONS_DIR, LOG_DIR ve API anahtarları) yükler,
# # gerekli dizinleri oluşturur ve merkezi loglama yapılandırmasını gerçekleştirir.


# import os
# import logging
# from dotenv import load_dotenv
# from pathlib import Path

# # .env dosyasını belirtilen tam yoldan yükleyin
# load_dotenv("C:/Users/mete/Zotero/zotasistan/.env")

# class Yapilandirma:
#     def __init__(self):
#         # Temel ortam değişkenleri
#         self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR"))
#         self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR"))
#         self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#         self.ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
#         self.ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
        
#         # Çıktı dizinleri
#         self.TEMIZMETIN_DIR = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin"))
#         self.TEMIZ_TABLO_DIZIN = Path(os.getenv("TEMIZ_TABLO_DIZIN", "temiztablo"))
#         self.TEMIZ_KAYNAKCA_DIZIN = Path(os.getenv("TEMIZ_KAYNAKCA_DIZIN", "temizkaynakca"))
#         self.EMBEDDING_PARCA_DIZIN = Path(os.getenv("EMBEDDING_PARCA_DIZIN", "embedingparca"))
        
#         # Citation Mapping için dizin
#         self.CITATIONS_DIR = Path(os.getenv("CITATIONS_DIR", "citations"))
        
#         # Log dosyalarının saklanacağı dizin
#         self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
#         self.STACK_DOSYASI = self.LOG_DIR / "islem.stack"
        
#         # Gerekli tüm dizinleri oluştur
#         self._dizinleri_hazirla()
#         # Merkezi loglama sistemini yapılandır
#         self._loglama_ayarla()
        
#     def _dizinleri_hazirla(self):
#         """Gerekli dizinlerin varlığını kontrol eder ve oluşturur."""
#         for d in [self.STORAGE_DIR, self.SUCCESS_DIR, self.TEMIZMETIN_DIR, 
#                   self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN, 
#                   self.EMBEDDING_PARCA_DIZIN, self.CITATIONS_DIR, self.LOG_DIR]:
#             d.mkdir(parents=True, exist_ok=True)
            
#     def _loglama_ayarla(self):
#         """Merkezi loglama sistemini başlatır: Dosya ve konsol loglarını yapılandırır."""
#         self.logger = logging.getLogger('ZoteroProcessor')
#         self.logger.setLevel(logging.DEBUG)
        
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
#         # Dosya log handler
#         file_handler = logging.FileHandler(self.LOG_DIR / 'islem.log', encoding='utf-8')
#         file_handler.setFormatter(formatter)
        
#         # Konsol log handler
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
        
#         self.logger.addHandler(file_handler)
#         self.logger.addHandler(console_handler)

# # Global yapılandırma nesnesi
# config = Yapilandirma()


# ### Açıklama

# # - **load_dotenv:** Belirtilen .env dosyasından ortam değişkenlerini yükler.
# # - **Ortam Değişkenleri:**  
# #   - `STORAGE_DIR`, `SUCCESS_DIR`, `TEMIZMETIN_DIR`, `TEMIZ_TABLO_DIZIN`, `TEMIZ_KAYNAKCA_DIZIN`, `EMBEDDING_PARCA_DIZIN`, `CITATIONS_DIR` gibi dizin yolları ve API anahtarları okunur.
# # - **Dizin Oluşturma:**  
# #   `_dizinleri_hazirla()` fonksiyonu, tüm gerekli dizinlerin (varsa eksik olanların) oluşturulmasını sağlar.
# # - **Loglama:**  
# #   `_loglama_ayarla()` fonksiyonu, merkezi loglama sistemi kurar; loglar hem dosya (LOG_DIR içindeki islem.log) hem de konsol üzerinden görüntülenir.
# # - **Global config:**  
# #   En sonunda, `config` adında global bir yapılandırma nesnesi oluşturulur. Diğer modüller bu nesneyi `from config_module import config` şeklinde kullanır.

# # Bu modül, projenin diğer bölümlerinde kullanılacak temel yapılandırmayı sağlar. Eğer eklemeler veya değişiklikler gerekirse lütfen bildirin.

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

class Config:
    """
    Uygulama genelinde kullanılacak yapılandırma ayarları.
    """
    # 📌 Dizin Yapıları
    BASE_DIR = Path(__file__).resolve().parent.parent
    STORAGE_DIR = BASE_DIR / "storage"
    SUCCESS_DIR = BASE_DIR / "processed/success"
    FAILED_DIR = BASE_DIR / "processed/failed"
    CLEAN_TEXT_DIR = BASE_DIR / "processed/clean_text"
    TABLES_DIR = BASE_DIR / "processed/tables"
    REFERENCES_DIR = BASE_DIR / "processed/references"
    EMBEDDINGS_DIR = BASE_DIR / "processed/embeddings"
    CITATIONS_DIR = BASE_DIR / "processed/citations"
    LOGS_DIR = BASE_DIR / "logs"
    
    # 📌 Dizinleri oluştur
    for directory in [STORAGE_DIR, SUCCESS_DIR, FAILED_DIR, CLEAN_TEXT_DIR, TABLES_DIR, 
                      REFERENCES_DIR, EMBEDDINGS_DIR, CITATIONS_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # 📌 Loglama Ayarları
    LOG_FILE = LOGS_DIR / "app.log"
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    logger = logging.getLogger(__name__)
    
    # 📌 PDF Metin Çıkarma Yöntemi (.env’den alınır)
    PDF_TEXT_EXTRACTION_METHOD = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()  # pdfminer, pdfplumber
    
    # 📌 Zotero API Ayarları
    ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
    ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
    ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE", "user")  # user, group
    
    # 📌 OpenAI API Ayarları
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # 📌 Alternatif Embedding Modelleri
    EMBEDDING_MODELS = {
        "contriever_large": os.getenv("CONTRIEVER_LARGE_MODEL", "facebook/contriever-large"),
        "specter_large": os.getenv("SPECTER_LARGE_MODEL", "allenai-specter-large"),
        "stsb_roberta_large": os.getenv("STSB_ROBERTA_LARGE_MODEL", "stsb-roberta-large"),
        "labse": os.getenv("LABSE_MODEL", "LaBSE"),
        "universal_sentence_encoder": os.getenv("USE_MODEL", "universal-sentence-encoder"),
    }
    
    # 📌 Chunk ve Büyük Dosya İşleme Ayarları
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))  # Varsayılan: 256 token
    LARGE_FILE_SPLIT_SIZE = int(os.getenv("LARGE_FILE_SPLIT_SIZE", 10000))  # Büyük PDF'ler için bölme limiti
    
    # 📌 NLP ve Regex Ayarları
    REGEX_SECTION_PATTERNS = {
        "Abstract": r"(?:^|\n)(Abstract|Özet)(?::)?\s*\n",
        "Introduction": r"(?:^|\n)(Introduction|Giriş)(?::)?\s*\n",
        "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)(?::)?\s*\n",
        "Results": r"(?:^|\n)(Results|Bulgular)(?::)?\s*\n",
        "Discussion": r"(?:^|\n)(Discussion|Tartışma)(?::)?\s*\n",
        "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)(?::)?\s*\n"
    }
    
    TABLE_DETECTION_PATTERNS = [
        r"(?:^|\n)(Tablo|Table|Çizelge|Chart)\s*\d+",  # Tablo 1, Table 2, Çizelge 3 gibi ifadeleri tespit eder.
        r"(?:^|\n)(Şekil|Figure|Fig)\s*\d+"  # Şekil 1, Figure 2 gibi ifadeleri tespit eder.
    ]
    
    # 📌 Paralel İşlem ve Hata Yönetimi
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))  # API çağrıları için maksimum tekrar sayısı
    BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", 1.0))  # Exponential backoff katsayısı
    
    # 📌 GUI Ayarları
    GUI_THEME = os.getenv("GUI_THEME", "light")  # Arayüz temasını ayarlar
    
# Konfigürasyon nesnesi oluştur
config = Config()
