# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")
