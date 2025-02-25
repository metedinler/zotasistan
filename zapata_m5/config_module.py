import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını belirtilen yoldan yükle
load_dotenv(".env") # C:\\Users\\mete\\Zotero\\zotasistan\\zapata_m5

class Config:
    """
    Uygulama genelinde kullanılacak yapılandırma ayarlarını yönetir.
    - Ortam değişkenlerini (.env) yükler.
    - Gerekli dizinleri oluşturur.
    - Merkezi loglama sistemini yapılandırır.
    """
    # Base directory (proje kök dizini olarak kullanılabilir)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Dizin Yapıları (Varsayılan değerler .env üzerinden de ayarlanabilir)
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
    SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", BASE_DIR / "processed" / "success"))
    CITATIONS_DIR = Path(os.getenv("CITATIONS_DIR", BASE_DIR / "processed" / "citations"))
    TABLES_DIR = Path(os.getenv("TABLES_DIR", BASE_DIR / "processed" / "tables"))
    CLEAN_TEXT_DIR = Path(os.getenv("TEMIZMETIN_DIR", BASE_DIR / "processed" / "clean_text"))
    EMBEDDINGS_DIR = Path(os.getenv("EMBEDDING_PARCA_DIZIN", BASE_DIR / "processed" / "embeddings"))
    TEMP_DIR = Path(os.getenv("TEMP_DIR", BASE_DIR / "temp"))
    LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))

    # Gerekli dizinlerin oluşturulması
    for directory in [STORAGE_DIR, SUCCESS_DIR, CITATIONS_DIR, TABLES_DIR, CLEAN_TEXT_DIR, EMBEDDINGS_DIR, TEMP_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Log Dosyası
    LOG_FILE = LOG_DIR / "app.log"

    # Loglama yapılandırması
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.DEBUG,  # İsteğe bağlı olarak os.getenv("LOG_LEVEL", "DEBUG") kullanılabilir
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    logger = logging.getLogger(__name__)

    # PDF Metin Çıkarma Yöntemi (.env’den okunur)
    PDF_TEXT_EXTRACTION_METHOD = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()

    # Zotero API Ayarları
    ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
    ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
    ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE", "user")  # user veya group

    # OpenAI API ve Embedding Ayarları
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Alternatif Embedding Modelleri (env üzerinden ayarlanabilir)
    EMBEDDING_MODELS = {
        "contriever_large": os.getenv("CONTRIEVER_LARGE_MODEL", "facebook/contriever-large"),
        "specter_large": os.getenv("SPECTER_LARGE_MODEL", "allenai-specter-large"),
        "stsb_roberta_large": os.getenv("STSB_ROBERTA_LARGE_MODEL", "stsb-roberta-large"),
        "labse": os.getenv("LABSE_MODEL", "LaBSE"),
        "universal_sentence_encoder": os.getenv("USE_MODEL", "universal-sentence-encoder"),
    }

    # Chunk ve Büyük Dosya İşleme Ayarları
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
    LARGE_FILE_SPLIT_SIZE = int(os.getenv("LARGE_FILE_SPLIT_SIZE", 10000))

    # NLP ve Regex Ayarları (Bilimsel bölümler için)
    REGEX_SECTION_PATTERNS = {
        "Abstract": r"(?:^|\n)(Abstract|Özet)(?::)?\s*\n",
        "Introduction": r"(?:^|\n)(Introduction|Giriş)(?::)?\s*\n",
        "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)(?::)?\s*\n",
        "Results": r"(?:^|\n)(Results|Bulgular)(?::)?\s*\n",
        "Discussion": r"(?:^|\n)(Discussion|Tartışma)(?::)?\s*\n",
        "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)(?::)?\s*\n"
    }

    # Tablo tespiti için regex desenleri
    TABLE_DETECTION_PATTERNS = [
        r"(?:^|\n)(Tablo|Table|Çizelge|Chart)\s*\d+",
        r"(?:^|\n)(Şekil|Figure|Fig)\s*\d+"
    ]

    # Paralel İşlem ve Hata Yönetimi Ayarları
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", 1.0))

    # GUI Ayarları
    GUI_THEME = os.getenv("GUI_THEME", "light")

config = Config()

# Bu final sürümü, tüm tartışmalarımızda belirtilen gereksinimler doğrultusunda ortam değişkenlerini yönetiyor, 
# gerekli dizinleri oluşturuyor ve merkezi loglama yapılandırmasını sağlıyor. Modüler sistemimizin diğer bölümleri 
# (zotero, pdf_processing, embedding, vb.) bu yapılandırma üzerinden ayarları kullanacak.

# Bu kod, sizin için final versiyon olarak hazırdır. Herhangi bir ekleme veya değişiklik isterseniz, lütfen belirtin.