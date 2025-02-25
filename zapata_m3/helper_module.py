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
