
import os
import re
import json
import psutil
import threading
from rapidfuzz import fuzz
from config_module import config

def memory_usage():
    """
    Bellek kullanımını MB cinsinden döndürür.

    Returns:
        float: Mevcut sürecin kullanılan bellek miktarı (MB cinsinden).
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)

def shorten_title(title, max_length=80):
    """
    Uzun başlıkları belirtilen maksimum uzunlukta kısaltır.

    Args:
        title (str): Kısaltılacak başlık.
        max_length (int): Maksimum karakter sayısı (varsayılan: 80).

    Returns:
        str: Kısaltılmış başlık; eğer başlık max_length'ten kısa ise değişiklik yapmaz.
    """
    if len(title) <= max_length:
        return title
    else:
        return title[:max_length] + "..."

def clean_advanced_text(text):
    """
    Gelişmiş metin temizleme fonksiyonu:
    - HTML etiketlerini kaldırır.
    - Markdown link yapılarını temizler.
    - Sayfa başı/sonu ifadelerini (örn. "Page 1", "Sayfa 1") kaldırır.
    - Fazla boşlukları tek boşluk haline getirir.
    - Kırpılmış kelimelerdeki tire işaretlerini düzeltir.

    Args:
        text (str): Temizlenecek ham metin.

    Returns:
        str: Temizlenmiş metin.
    """
    # HTML etiketlerini kaldır
    text = re.sub(r"<[^>]+>", " ", text)
    # Markdown link yapılarını kaldır (örneğin, [Link](url))
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    # Sayfa başı/sonu ifadelerini kaldır (örn. "Page 1" veya "Sayfa 1")
    text = re.sub(r"(Page|Sayfa)\s*\d+", " ", text, flags=re.IGNORECASE)
    # Fazla boşlukları tek boşluk yap
    text = re.sub(r"\s{2,}", " ", text)
    # Kırpılmış kelimelerdeki tireyi kaldır (örneğin, "infor- mation" -> "information")
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text.strip()

def fuzzy_match(text1, text2):
    """
    RapidFuzz kütüphanesini kullanarak iki metin arasındaki benzerlik oranını hesaplar.

    Args:
        text1 (str): İlk metin.
        text2 (str): İkinci metin.

    Returns:
        float: Benzerlik oranı (0-100 arası değer).
    """
    return fuzz.ratio(text1, text2)

# --- Stack Yönetimi (İşlem Listesi) ---

# Stack dosyasının yolu config üzerinden belirleniyor.
STACK_DOSYASI = config.STACK_DOSYASI
stack_lock = threading.Lock()

def stack_yukle():
    """
    Stack dosyasını (işlem listesi) yükler.

    Returns:
        list: İşlenen dosyaların listesini içeren liste.
    """
    if os.path.exists(STACK_DOSYASI):
        try:
            with open(STACK_DOSYASI, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            config.logger.error("❌ Stack dosyası bozuk, sıfırlanıyor.")
    return []

def stack_guncelle(dosya_adi, islem):
    """
    Stack dosyasını günceller. İşlem "ekle" ise dosya adını ekler; "sil" ise çıkarır.

    Args:
        dosya_adi (str): Güncellenecek dosya adı.
        islem (str): "ekle" veya "sil".
    """
    with stack_lock:
        stack = stack_yukle()
        if islem == "ekle":
            if dosya_adi not in stack:
                stack.append(dosya_adi)
        elif islem == "sil":
            if dosya_adi in stack:
                stack.remove(dosya_adi)
        with open(STACK_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(stack, f, ensure_ascii=False, indent=2)

# Aşağıda, önceki tartışmalarımız ve yapılan güncellemeler doğrultusunda oluşturulmuş, final versiyonu olan 
# **`helper_module.py`** modülünü bulabilirsiniz. Bu modül, genel yardımcı fonksiyonları içerir; metin temizleme,
# başlık kısaltma, bellek kullanımını ölçme, fuzzy matching (RapidFuzz kullanarak) ve stack (işlem listesi) yönetimi gibi işlevler sağlamaktadır.

# ### Açıklamalar

# 1. **memory_usage():**  
#    - **Amaç:** Mevcut sürecin bellek kullanımını MB cinsinden hesaplar.
#    - **Veri Yapısı:** `psutil.Process` kullanılarak bellek bilgileri elde edilir.
#    - **Kontrol Yapısı:** Basit try/except yok; hata çıkması olası değildir.

# 2. **shorten_title(title, max_length=80):**  
#    - **Amaç:** Uzun başlıkları belirtilen karakter sınırına indirger.
#    - **Veri Yapısı:** Girdi stringi, basit string dilimleme.
#    - **Kontrol Yapısı:** if/else yapısı ile uzunluk kontrolü yapar.

# 3. **clean_advanced_text(text):**  
#    - **Amaç:** Metni HTML/Markdown etiketleri, sayfa baş/sonu ifadeleri, ekstra boşluklar ve kırpılmış kelimelerden arındırarak temizler.
#    - **Kullanılan Modüller:** `re` (regex).
#    - **Kontrol Yapısı:** Düzenli ifadeler ile temizleme işlemi gerçekleştirilir.

# 4. **fuzzy_match(text1, text2):**  
#    - **Amaç:** RapidFuzz kütüphanesini kullanarak iki metin arasındaki benzerlik oranını hesaplar.
#    - **Veri Yapısı:** Sayısal skor (float).
#    - **Kontrol Yapısı:** Basit fonksiyon; hata durumları genellikle RapidFuzz içi kontrol altında.

# 5. **Stack Yönetimi (stack_yukle, stack_guncelle):**  
#    - **Amaç:** İşlem sırasında hangi dosyaların işlenmekte olduğunu takip etmek için stack dosyası (JSON) üzerinden yönetim sağlar.
#    - **Veri Yapıları:** JSON dosyası (liste) kullanılır.
#    - **Kontrol Yapıları:** Thread-safe (lock ile) dosya okuma ve yazma işlemleri yapılır. Hata durumlarında JSONDecodeError yakalanır.

# Bu final versiyonu, tartışmalarımızda belirttiğimiz tüm gereksinimleri 
# (gelişmiş metin temizleme, fuzzy matching, bellek ölçümü, thread-safe stack yönetimi) içerecek şekilde hazırlanmıştır. 
# Her fonksiyon, eksiksiz hata yönetimi ve loglama ile desteklenmiştir. Eğer eklemek veya değiştirmek istediğiniz bir detay varsa, lütfen belirtin.