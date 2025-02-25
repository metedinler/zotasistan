import os
from pathlib import Path
import logging

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('directory_management.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Dizin Yönetimi Fonksiyonları
# ----------------------------

def ensure_directories(directories):
    """
    Belirtilen dizinlerin varlığını kontrol eder ve yoksa oluşturur.
    Args:
        directories (list): Oluşturulacak dizinlerin listesi.
    """
    for directory in directories:
        try:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Dizin oluşturuldu: {directory}")
            else:
                logger.info(f"✅ Dizin zaten mevcut: {directory}")
        except Exception as e:
            logger.error(f"❌ Dizin oluşturulamadı: {directory}, Hata: {e}")


def init_dirs(base_dirs):
    """
    Programın çalışması için gerekli tüm temel dizinleri oluşturur.
    Args:
        base_dirs (dict): Dizin adları ve yollarını içeren bir sözlük.
    """
    try:
        # Ana dizinlerin oluşturulması
        ensure_directories(base_dirs.values())

        # Özel alt dizinlerin oluşturulması (örneğin, JSON, CSV, Excel için)
        special_formats = ['json', 'csv', 'excel']
        for fmt in special_formats:
            special_dir = base_dirs.get("TEMIZ_TABLO_DIZIN") / fmt
            ensure_directories([special_dir])

        logger.info("✅ Tüm gerekli dizinler başarıyla hazırlandı.")
    except Exception as e:
        logger.error(f"❌ Dizin başlatma sırasında hata: {e}")


def clean_directories(directories):
    """
    Belirtilen dizinleri temizler (içeriklerini siler).
    Args:
        directories (list): Temizlenecek dizinlerin listesi.
    """
    for directory in directories:
        try:
            if directory.exists():
                for file in directory.iterdir():
                    if file.is_file():
                        file.unlink()  # Dosyaları sil
                    elif file.is_dir():
                        shutil.rmtree(file)  # Alt dizinleri sil
                logger.info(f"✅ Dizin temizlendi: {directory}")
            else:
                logger.warning(f"⚠️ Dizin bulunamadı: {directory}")
        except Exception as e:
            logger.error(f"❌ Dizin temizlenirken hata: {directory}, Hata: {e}")


def get_directory_size(directory):
    """
    Belirtilen dizinin toplam boyutunu hesaplar.
    Args:
        directory (Path): Boyutu hesaplanacak dizin.
    Returns:
        int: Dizinin toplam boyutu (byte cinsinden).
    """
    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        logger.info(f"✅ Dizin boyutu hesaplandı: {directory}, Boyut: {total_size} bytes")
        return total_size
    except Exception as e:
        logger.error(f"❌ Dizin boyutu hesaplanırken hata: {directory}, Hata: {e}")
        return 0


# ----------------------------
# Örnek Kullanım
# ----------------------------

if __name__ == "__main__":
    # Ortam değişkenlerinden dizin yollarını çek
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
    SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", "success"))
    KAYNAK_DIZIN = Path(os.getenv("KAYNAK_DIZIN", "kaynak"))
    HEDEF_DIZIN = Path(os.getenv("HEDEF_DIZIN", "hedef"))
    TEMIZMETIN_DIR = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin"))
    TEMIZ_TABLO_DIZIN = Path(os.getenv("TEMIZ_TABLO_DIZIN", "temiztablo"))
    TEMIZ_KAYNAKCA_DIZIN = Path(os.getenv("TEMIZ_KAYNAKCA_DIZIN", "temizkaynakca"))
    PDF_DIR = Path(os.getenv("PDF_DIR", "pdf"))
    EMBEDDING_PARCA_DIR = Path(os.getenv("EMBEDDING_PARCA_DIR", "embeddingparca"))

    # Dizin yapılandırması
    base_dirs = {
        "STORAGE_DIR": STORAGE_DIR,
        "SUCCESS_DIR": SUCCESS_DIR,
        "KAYNAK_DIZIN": KAYNAK_DIZIN,
        "HEDEF_DIZIN": HEDEF_DIZIN,
        "TEMIZMETIN_DIR": TEMIZMETIN_DIR,
        "TEMIZ_TABLO_DIZIN": TEMIZ_TABLO_DIZIN,
        "TEMIZ_KAYNAKCA_DIZIN": TEMIZ_KAYNAKCA_DIZIN,
        "PDF_DIR": PDF_DIR,
        "EMBEDDING_PARCA_DIR": EMBEDDING_PARCA_DIR,
    }

    # Dizinleri başlat
    init_dirs(base_dirs)

    # Dizin boyutunu kontrol et
    print(f"PDF Dizini Boyutu: {get_directory_size(PDF_DIR)} bytes")

    # Dizinleri temizle (isteğe bağlı)
    # clean_directories([PDF_DIR, EMBEDDING_PARCA_DIR])
    
#     Modül Açıklaması
# ensure_directories :
# Verilen dizinlerin var olup olmadığını kontrol eder ve yoksa oluşturur.
# Her bir dizin için log kaydı tutar.
# init_dirs :
# Programın çalışması için gerekli tüm temel dizinleri oluşturur.
# Özel alt dizinler (örneğin, JSON, CSV, Excel) için de destek sağlar.
# clean_directories :
# Belirtilen dizinlerdeki tüm dosya ve alt dizinleri siler.
# Temizleme işlemi için log kaydı tutar.
# get_directory_size :
# Bir dizinin toplam boyutunu hesaplar ve byte cinsinden döndürür.
# Hesaplama sırasında oluşan hataları loglar.
# Özellikler
# Güvenlik : Dizin işlemleri sırasında hata oluşursa, detaylı log kaydı tutulur.
# Esneklik : Dizin yolları .env dosyasından veya varsayılan değerlerden çekilebilir.
# Modülerlik : Diğer modüllerle entegre edilebilir ve bağımsız olarak test edilebilir.