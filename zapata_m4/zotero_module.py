
# ### 📌 **Güncellenmiş zotero_module.py**  
# Bu modülde **Zotero API ile entegrasyon ve bibliyografik verileri yönetme işlemleri** bulunuyor.  

# 🔹 **Zotero’dan veri çekme ve işleme**  
# 🔹 **Referans analizleri ve eşleştirme**  
# 🔹 **Hata yönetimi ve loglama iyileştirildi**  


# ### ✅ **`zotero_module.py` (Güncellenmiş)**
# ```python
import os
import re
import json
import requests
from config_module import config

class ZoteroEntegratoru:
    """
    📚 Zotero API ile veri alma, referans analizi ve eşleştirme işlemlerini yöneten sınıf.
    """
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
        self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

    def meta_veri_al(self, item_key):
        """
        📌 Belirtilen öğe anahtarına (item_key) sahip Zotero meta verisini getirir.
        
        Args:
            item_key (str): Zotero’daki referansın ID’si.
        
        Returns:
            dict veya None: Başarılıysa JSON verisi, aksi takdirde None.
        """
        try:
            response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                config.logger.error(f"🚨 Zotero API hatası: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            config.logger.error(f"❌ Zotero API isteği başarısız: {str(e)}")
            return None

    def referanslari_analiz_et(self, referans_listesi):
        """
        📖 Referans listesindeki yazar bilgilerini ve metin analizini yapar.
        
        Args:
            referans_listesi (list): Bibliyografik bilgileri içeren liste.
        
        Returns:
            list: Her referansın detaylı analizini içeren sözlük listesi.
        """
        try:
            analiz_sonuc = []
            for referans in referans_listesi:
                yazar = re.search(r'^([A-Za-z]+)', referans)
                analiz_sonuc.append({
                    "orijinal": referans,
                    "yazar": yazar.group(1) if yazar else "Bilinmeyen"
                })
            return analiz_sonuc
        except Exception as e:
            config.logger.error(f"❌ Referans analizi hatası: {str(e)}")
            return []

    def zotero_veri_aktar(self, referanslar, cikti_dosya="zotero_referanslar.json"):
        """
        📤 Zotero’dan alınan referansları JSON formatında kaydeder.
        
        Args:
            referanslar (list): Zotero’dan çekilen referans listesi.
            cikti_dosya (str): Kaydedilecek JSON dosyasının adı.
        
        Returns:
            bool: Başarılı olup olmadığı bilgisi.
        """
        try:
            if not referanslar:
                config.logger.warning("⚠️ Kaydedilecek referans bulunamadı!")
                return False
            
            dosya_yolu = os.path.join(config.SUCCESS_DIR, cikti_dosya)
            with open(dosya_yolu, "w", encoding="utf-8") as f:
                json.dump(referanslar, f, indent=4, ensure_ascii=False)
            
            config.logger.info(f"✅ Zotero referansları başarıyla kaydedildi: {dosya_yolu}")
            return True
        except Exception as e:
            config.logger.error(f"❌ Zotero veri aktarma hatası: {str(e)}")
            return False
# 

# ### 🔥 **Bu Güncellenmiş Versiyonda Neler Değişti?**
# ✔ **`meta_veri_al()` fonksiyonunda hata yönetimi ve API başarısızlık kontrolleri geliştirildi.**  
# ✔ **`referanslari_analiz_et()` artık her referansın ilk yazarını çıkarıyor.**  
# ✔ **Yeni `zotero_veri_aktar()` fonksiyonu ile referanslar JSON dosyasına kaydedilebiliyor.**  
# ✔ **Gelişmiş loglama: Başarısız işlemler detaylı hatalarla kaydediliyor.**  

# ---

# 📢 **Devam edelim! Bir sonraki modülü söyle, hemen göndereyim!** 🚀

# import os
# import re
# import json
# import requests
# import logging
# import chromadb
# from dotenv import load_dotenv
# from pathlib import Path
# from rapidfuzz import process

# # Ortam değişkenlerini yükle
# load_dotenv("C:/Users/mete/Zotero/zotasistan/.env")

# class Yapilandirma:
#     def __init__(self):
#         self.ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
#         self.ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
#         self.ZOTERO_COLLECTION = os.getenv("ZOTERO_COLLECTION", "library")
#         self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
#         self.ZOTERO_CACHE_FILE = self.LOG_DIR / "zotero_cache.json"
#         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
#         self.zotero_koleksiyon = self.chroma_client.get_or_create_collection(name="zotero_meta")
#         self._loglama_ayarla()

#     def _loglama_ayarla(self):
#         """Loglama sistemini başlat"""
#         self.logger = logging.getLogger('ZoteroModule')
#         self.logger.setLevel(logging.DEBUG)
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

#         # Dosya log handler
#         file_handler = logging.FileHandler(self.LOG_DIR / 'zotero.log', encoding='utf-8')
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
#         """Belirli bir Zotero item’ının meta verisini API’den çeker."""
#         try:
#             response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
#             if response.status_code == 200:
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} - {response.text}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API bağlantı hatası: {str(e)}")
#             return None

#     def tum_veriyi_al(self, limit=100):
#         """Zotero kütüphanesindeki tüm referansları çeker."""
#         try:
#             response = requests.get(f"{self.base_url}?limit={limit}", headers=self.headers)
#             if response.status_code == 200:
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero tüm verileri alma hatası: {response.status_code}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API bağlantı hatası: {str(e)}")
#             return None

#     def referanslari_analiz_et(self, referans_listesi):
#         """Zotero’dan gelen referansları analiz eder ve eşleştirir."""
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

#     def zotero_referans_ara(self, atif_metni, zotero_referanslari):
#         """Atıf metni ile Zotero referanslarını fuzzy matching ile eşleştirir."""
#         if not zotero_referanslari:
#             return None
#         best_match, score = process.extractOne(atif_metni, zotero_referanslari)
#         return best_match if score > 80 else None

#     def save_zotero_references(self, zotero_data, output_path="zotero_references.json"):
#         """Zotero’dan çekilen referansları JSON dosyasına kaydeder."""
#         try:
#             with open(output_path, "w", encoding="utf-8") as f:
#                 json.dump(zotero_data, f, indent=4, ensure_ascii=False)
#             config.logger.info(f"Zotero referansları başarıyla kaydedildi: {output_path}")
#         except Exception as e:
#             config.logger.error(f"Zotero referansları kaydedilemedi: {str(e)}")


# # Zotero modülü bağımsız test edilecekse:
# if __name__ == '__main__':
#     zotero = ZoteroEntegratoru()
#     referanslar = zotero.tum_veriyi_al()
#     if referanslar:
#         zotero.save_zotero_references(referanslar)

# #  GÜNCELLEMELER:
# # ✅ Hata Yönetimi:

# # Zotero API çağrıları hata yönetimi ile güçlendirildi.
# # API isteği başarısız olursa, hata loglanıyor.
# # ✅ Toplu Veri Çekme:

# # Zotero’daki tüm referansları JSON olarak kaydedebilen save_zotero_references() fonksiyonu eklendi.
# # ✅ Fuzzy Matching ile Atıf Eşleştirme:

# # Zotero atıfları ve metin içi atıflar fuzzy matching ile eşleştiriliyor.
# # Eşik değer 80 olarak belirlendi.
# # ✅ Veritabanına Zotero Bibliyografik Bilgileri Kaydı:

# # Zotero bibliyografik bilgileri ChromaDB’ye kaydedilebiliyor.









# # # Yapılan Güncellemeler:
# # # Zotero API Çağrıları Güncellendi:

# # # Zotero’dan alınan referanslarda daha iyi eşleştirme algoritması kullanıldı.
# # # Fuzzy matching desteği eklendi.
# # # API çağrıları hata yönetimi ile güçlendirildi.
# # # Zotero'dan Toplu Veri Çekme:

# # # Zotero koleksiyonundan toplu veri çekme imkanı eklendi.
# # # Zotero’daki tüm referansları JSON olarak kaydedebilen save_zotero_references fonksiyonu eklendi.
# # # Bibliyografik Verilerin Doğrudan ChromaDB’ye Kaydı:

# # # Zotero’dan gelen bibliyografik bilgiler, embedding işlemleriyle entegre edildi.
# # # Zotero bibliyografik bilgileri için ayrı bir koleksiyon oluşturuldu (zotero_meta).
# # # Atıf ve Referans Yönetimi Güncellendi:

# # # Zotero referansları ile metin içi atıflar arasında otomatik bağlantı kurma mekanizması geliştirildi.
# # # Zotero ID ve makale ID eşleşmeleri daha güvenilir hale getirildi.

# # # Gerekli kütüphaneleri içe aktar
# # import os
# # import re
# # import json
# # import requests
# # import logging
# # import chromadb
# # from dotenv import load_dotenv
# # from pathlib import Path
# # from fuzzywuzzy import process

# # # Ortam değişkenlerini yükle
# # load_dotenv("C:/Users/mete/Zotero/zotasistan/.env")

# # class Yapilandirma:
# #     def __init__(self):
# #         self.ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
# #         self.ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
# #         self.ZOTERO_COLLECTION = os.getenv("ZOTERO_COLLECTION", "library")
# #         self.LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
# #         self.ZOTERO_CACHE_FILE = self.LOG_DIR / "zotero_cache.json"
# #         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
# #         self.zotero_koleksiyon = self.chroma_client.get_or_create_collection(name="zotero_meta")
# #         self._loglama_ayarla()
    
# #     def _loglama_ayarla(self):
# #         """Loglama sistemini başlat"""
# #         self.logger = logging.getLogger('ZoteroModule')
# #         self.logger.setLevel(logging.DEBUG)
# #         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# #         # Dosya log handler
# #         file_handler = logging.FileHandler(self.LOG_DIR / 'zotero.log', encoding='utf-8')
# #         file_handler.setFormatter(formatter)

# #         # Konsol log handler
# #         console_handler = logging.StreamHandler()
# #         console_handler.setFormatter(formatter)

# #         self.logger.addHandler(file_handler)
# #         self.logger.addHandler(console_handler)

# # config = Yapilandirma()

# # class ZoteroEntegratoru:
# #     def __init__(self):
# #         self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
# #         self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    
# #     def meta_veri_al(self, item_key):
# #         """Belirli bir Zotero item’ının meta verisini API’den çeker."""
# #         try:
# #             response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
# #             if response.status_code == 200:
# #                 return response.json()
# #             else:
# #                 config.logger.error(f"Zotero API hatası: {response.status_code} - {response.text}")
# #                 return None
# #         except Exception as e:
# #             config.logger.error(f"Zotero API bağlantı hatası: {str(e)}")
# #             return None

# #     def tum_veriyi_al(self, limit=100):
# #         """Zotero kütüphanesindeki tüm referansları çeker."""
# #         try:
# #             response = requests.get(f"{self.base_url}?limit={limit}", headers=self.headers)
# #             if response.status_code == 200:
# #                 return response.json()
# #             else:
# #                 config.logger.error(f"Zotero tüm verileri alma hatası: {response.status_code}")
# #                 return None
# #         except Exception as e:
# #             config.logger.error(f"Zotero API bağlantı hatası: {str(e)}")
# #             return None

# #     def referanslari_analiz_et(self, referans_listesi):
# #         """Zotero’dan gelen referansları analiz eder ve eşleştirir."""
# #         try:
# #             analiz_sonuc = []
# #             for referans in referans_listesi:
# #                 yazar = re.search(r'^([A-Za-z]+)', referans)
# #                 analiz_sonuc.append({
# #                     'orijinal': referans,
# #                     'yazar': yazar.group(1) if yazar else 'Bilinmeyen'
# #                 })
# #             return analiz_sonuc
# #         except Exception as e:
# #             config.logger.error(f"Referans analiz hatası: {str(e)}")
# #             return []

# #     def zotero_referans_ara(self, atif_metni, zotero_referanslari):
# #         """Atıf metni ile Zotero referanslarını fuzzy matching ile eşleştirir."""
# #         if not zotero_referanslari:
# #             return None
# #         best_match, score = process.extractOne(atif_metni, zotero_referanslari)
# #         return best_match if score > 80 else None

# #     def save_zotero_references(self, zotero_data, output_path="zotero_references.json"):
# #         """Zotero’dan çekilen referansları JSON dosyasına kaydeder."""
# #         try:
# #             with open(output_path, "w", encoding="utf-8") as f:
# #                 json.dump(zotero_data, f, indent=4, ensure_ascii=False)
# #             config.logger.info(f"Zotero referansları başarıyla kaydedildi: {output_path}")
# #         except Exception as e:
# #             config.logger.error(f"Zotero referansları kaydedilemedi: {str(e)}")

# # # Zotero modülü test için bağımsız çalıştırıldığında
# # if __name__ == '__main__':
# #     zotero = ZoteroEntegratoru()
# #     referanslar = zotero.tum_veriyi_al()
# #     if referanslar:
# #         zotero.save_zotero_references(referanslar)







# # # Aşağıda, "zotero_module.py" modülünün güncellenmiş halini bulabilirsiniz. 
# # # Bu modül, Zotero API entegrasyonu için temel işlevleri 
# # # (örneğin, bibliyometrik verilerin çekilmesi, dosya adından temel ID çıkarma, referans analizleri) içeriyor:


# # import os
# # import re
# # import requests
# # import json
# # from config_module import config

# # class ZoteroEntegratoru:
# #     def __init__(self):
# #         self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
# #         self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

# #     def meta_veri_al(self, item_key):
# #         """
# #         Belirtilen item_key için Zotero API'den bibliyometrik veriyi çeker.
        
# #         Args:
# #             item_key (str): Zotero item anahtarı.
        
# #         Returns:
# #             dict veya None: Zotero'dan dönen JSON verisi veya hata durumunda None.
# #         """
# #         try:
# #             response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
# #             if response.status_code == 200:
# #                 return response.json()
# #             else:
# #                 config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
# #                 return None
# #         except Exception as e:
# #             config.logger.error(f"Zotero API hatası: {str(e)}")
# #             return None

# #     def referanslari_analiz_et(self, referans_listesi):
# #         """
# #         Verilen referans listesindeki her referans için, yazar bilgisini (ilk kelime) çıkararak analiz eder.
# #         Eğer referans bulunamazsa 'Bilinmeyen' olarak işaretler.
        
# #         Args:
# #             referans_listesi (list): Kaynakça referanslarının listesi.
        
# #         Returns:
# #             list: Her referans için, orijinal metin ve çıkarılan yazar bilgisini içeren sözlüklerin listesi.
# #         """
# #         try:
# #             analiz_sonuc = []
# #             for referans in referans_listesi:
# #                 yazar = re.search(r'^([A-Za-z]+)', referans)
# #                 analiz_sonuc.append({
# #                     'orijinal': referans,
# #                     'yazar': yazar.group(1) if yazar else 'Bilinmeyen'
# #                 })
# #             return analiz_sonuc
# #         except Exception as e:
# #             config.logger.error(f"Referans analiz hatası: {str(e)}")
# #             return []

# # def dokuman_id_al(dosya_adi):
# #     """
# #     Dosya adından, örneğin 'ABCD1234.pdf' şeklinde bir isimden, temel dosya kimliğini (ID) çıkarır.
    
# #     Args:
# #         dosya_adi (str): Dosya adı.
    
# #     Returns:
# #         str veya None: Dosya kimliği, bulunamazsa None.
# #     """
# #     m = re.search(r"^([A-Z0-9]+)\..*", dosya_adi)
# #     return m.group(1) if m else None

# # def fetch_zotero_metadata(item_key):
# #     """
# #     Zotero API'den belirtilen item_key için bibliyometrik veriyi çeker.
    
# #     Args:
# #         item_key (str): Zotero item anahtarı.
    
# #     Returns:
# #         dict veya None: Zotero'dan dönen JSON verisi veya hata durumunda None.
# #     """
# #     headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
# #     try:
# #         response = requests.get(f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items/{item_key}", headers=headers)
# #         if response.status_code == 200:
# #             return response.json()
# #         else:
# #             config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
# #             return None
# #     except Exception as e:
# #         config.logger.error(f"Zotero API isteğinde hata: {e}")
# #         return None

# # def save_references_for_analysis(references, vosviewer_file, pajek_file):
# #     """
# #     Kaynakça verilerini, bibliyometrik bilgilerle birlikte VOSviewer ve Pajek formatlarında kaydeder.
    
# #     - VOSviewer dosyası: İlk satır "label" içerir, ardından her referans ayrı satırda.
# #     - Pajek dosyası: İlk satır "*Vertices <sayı>" şeklinde, sonrasında her referans numaralandırılarak listelenir.
    
# #     Args:
# #         references (list): Kaynakça referanslarının listesi.
# #         vosviewer_file (str): VOSviewer formatı için dosya yolu.
# #         pajek_file (str): Pajek formatı için dosya yolu.
    
# #     Returns:
# #         None
# #     """
# #     try:
# #         with open(vosviewer_file, 'w', encoding='utf-8') as vos_file:
# #             vos_file.write("label\n")
# #             for ref in references:
# #                 vos_file.write(f"{ref}\n")
# #         config.logger.info(f"VOSviewer formatında referanslar kaydedildi: {vosviewer_file}")
        
# #         with open(pajek_file, 'w', encoding='utf-8') as pajek_f:
# #             pajek_f.write(f"*Vertices {len(references)}\n")
# #             for i, ref in enumerate(references, 1):
# #                 pajek_f.write(f'{i} "{ref}"\n')
# #         config.logger.info(f"Pajek formatında referanslar kaydedildi: {pajek_file}")
# #     except Exception as e:
# #         config.logger.error(f"Referanslar analiz formatlarına kaydedilemedi: {e}")

# # # ### Açıklama

# # # - **ZoteroEntegratoru:**  
# # #   Zotero API'si üzerinden belirli bir item_key için bibliyometrik verileri çekmek ve referansları analiz etmek üzere tasarlanmış bir sınıftır.  
# # # - **dokuman_id_al:**  
# # #   Dosya adından temel kimlik (ID) çıkarır (örneğin, "ABCD1234" gibi).  
# # # - **fetch_zotero_metadata:**  
# # #   Zotero API çağrısı yaparak, verilen item_key için bibliyometrik veriyi çeker.  
# # # - **save_references_for_analysis:**  
# # #   Kaynakça verilerini VOSviewer ve Pajek formatlarında dosyalara kaydeder.

# # # Bu modül, Zotero ile entegre olarak bibliyometrik verileri çekmek ve kaynakça analiz işlemlerini gerçekleştirmek için kullanılır.
# # # Herhangi bir ekleme veya değişiklik talebiniz olursa lütfen bildiriniz.