# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")
