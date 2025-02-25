# ### 📌 **Güncellenmiş `helper_module.py` Modülü**  
# Bu modül, **genel yardımcı fonksiyonları** içerir:  
# 🔹 **Bellek yönetimi ve optimizasyon fonksiyonları**  
# 🔹 **Metin temizleme ve düzenleme araçları**  
# 🔹 **Fuzzy matching (bulanık eşleşme) fonksiyonları**  
# 🔹 **Stack (işlem listesi) yönetimi**  

# ---

# ## ✅ **`helper_module.py` (Güncellenmiş)**
# ```python
import os
import re
import json
import psutil
import threading
import numpy as np
from rapidfuzz import fuzz
from config_module import config

# 🧠 BELLEK VE KAYNAK YÖNETİMİ
def memory_usage():
    """
    📌 Mevcut bellek kullanımını (MB cinsinden) döndürür.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024**2  # MB cinsinden döndür

# 📌 METİN TEMİZLEME FONKSİYONLARI
def shorten_title(title, max_length=80):
    """
    📌 Uzun başlıkları belirlenen uzunlukta kısaltır.
    
    Args:
        title (str): Kısaltılacak başlık.
        max_length (int): Maksimum uzunluk (varsayılan: 80).

    Returns:
        str: Kısaltılmış başlık.
    """
    return title if len(title) <= max_length else title[:max_length] + "..."

def clean_advanced_text(text):
    """
    📌 Gelişmiş metin temizleme fonksiyonu:
    - HTML/Markdown etiketlerini kaldırır.
    - Fazla boşlukları temizler.
    - Sayfa başı/sonu ifadelerini kaldırır.
    - Kırpılmış kelimelerdeki tireleri temizler.

    Args:
        text (str): Temizlenecek metin.

    Returns:
        str: Temizlenmiş metin.
    """
    text = re.sub(r"<[^>]+>", " ", text)  # HTML etiketlerini kaldır
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)  # Markdown linklerini kaldır
    text = re.sub(r"(Page|Sayfa)\s*\d+", " ", text, flags=re.IGNORECASE)  # Sayfa numaralarını kaldır
    text = re.sub(r"\s{2,}", " ", text)  # Fazla boşlukları tek boşluk yap
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)  # Kırpılmış kelimeleri düzelt
    return text.strip()

# 📌 BULANIK EŞLEŞTİRME (FUZZY MATCH)
def fuzzy_match(text1, text2):
    """
    📌 RapidFuzz kullanarak iki metin arasındaki benzerlik oranını hesaplar.

    Args:
        text1 (str): İlk metin.
        text2 (str): İkinci metin.

    Returns:
        float: Benzerlik oranı (% cinsinden).
    """
    return fuzz.ratio(text1, text2)

# 📌 STACK YÖNETİMİ (İŞLEM LİSTESİ)
STACK_DOSYASI = config.STACK_DOSYASI
stack_lock = threading.Lock()

def stack_yukle():
    """
    📌 Stack dosyasını yükler ve işlem listesini döndürür.

    Returns:
        list: İşlem listesi.
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
    📌 Stack dosyasını günceller.
    
    Args:
        dosya_adi (str): Güncellenecek dosya adı.
        islem (str): "ekle" veya "sil".
    """
    with stack_lock:
        stack = stack_yukle()
        if islem == "ekle" and dosya_adi not in stack:
            stack.append(dosya_adi)
        elif islem == "sil" and dosya_adi in stack:
            stack.remove(dosya_adi)
        with open(STACK_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(stack, f, ensure_ascii=False, indent=2)
# ```

# ---

# ### 🔥 **Bu Güncellenmiş Versiyonda Neler Değişti?**
# ✔ **Metin temizleme (clean_advanced_text) fonksiyonu genişletildi.**  
# ✔ **Bulanık eşleşme fonksiyonu (`fuzzy_match`) RapidFuzz ile optimize edildi.**  
# ✔ **Bellek kullanımını ölçen `memory_usage()` fonksiyonu eklendi.**  
# ✔ **İşlem sırasını takip eden `stack_yukle()` ve `stack_guncelle()` güncellendi.**  

# ---

# 📢 **Sıradaki modülü söyle, hemen gönderelim!** 🚀