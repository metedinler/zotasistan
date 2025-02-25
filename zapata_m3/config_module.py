# Modüllerin İsimleri
# config_module.py
# zotero_module.py
# pdf_processing.py
# embedding_module.py
# alternative_embedding_module.py
# robust_embedding_module.py
# helper_module.py
# processing_manager.py
# file_save_module.py
# citation_mapping_module.py
# gui_module.py
# main.py
# .env (ortam değişkenlerini içeren dosya)

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
