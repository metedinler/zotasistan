import re
import requests
from config_module import config

class ZoteroEntegratoru:
    """
    📌 **Zotero API entegrasyonunu yöneten sınıf.**
    
    🔹 **Özellikler:**
    - Zotero API'si ile item key'e göre bibliyometrik verileri çeker.
    - Verilen referans listesini analiz edip, regex kullanarak yazar isimlerini çıkarır.
    - JSON dosyasından Zotero benzersiz kimliklerini alır.
    - Kaynakça metinlerini analiz eder ve veri tabanına kaydeder.

    🔹 **Yeni Güncellemeler:**
    - JSON dosyasından **Zotero ID alma** özelliği eklendi.
    - **Veri tabanı entegrasyonu** için kaynakça metinleri işlenebilir hale getirildi.
    - **Gelişmiş hata yönetimi** ve detaylı loglama eklendi.
    """

    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
        self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

    def dokuman_id_al(self, dosya_adi):
        """
        📌 **Dosya adından Zotero benzersiz kimliğini çıkarır.**
        
        Args:
            dosya_adi (str): PDF veya TXT dosya adı.
        
        Returns:
            str veya None: Zotero'daki benzersiz doküman ID'si.
        """
        match = re.search(r'([A-Z0-9]{8})', dosya_adi)  # Zotero ID'leri genellikle 8 karakter uzunluğundadır.
        if match:
            return match.group(1)
        return None  # Eğer eşleşme bulunamazsa None döndür.

    def fetch_zotero_metadata(self, item_key):
        """
        📌 **Zotero'dan belirli bir öğenin metadatasını çeker.**
        
        Args:
            item_key (str): Zotero'daki dokümanın benzersiz kimliği.
        
        Returns:
            dict veya None: Eğer başarıyla çekilirse JSON verisi, yoksa None.
        """
        try:
            response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                config.logger.error(f"⚠️ Zotero API Hatası! Kod: {response.status_code}")
                return None
        except Exception as e:
            config.logger.error(f"❌ Zotero bağlantı hatası: {str(e)}")
            return None





# import re
# import requests
# from config_module import config

# class ZoteroEntegratoru:
#     """
#     📌 **Zotero API entegrasyonunu yöneten sınıf.**
    
#     🔹 **Özellikler:**
#     - Zotero API'si ile item key'e göre bibliyometrik verileri çeker.
#     - Verilen referans listesini analiz edip, regex kullanarak yazar isimlerini çıkarır.
#     - JSON dosyasından Zotero benzersiz kimliklerini alır.
#     - Kaynakça metinlerini analiz eder ve veri tabanına kaydeder.

#     🔹 **Yeni Güncellemeler:**
#     - JSON dosyasından **Zotero ID alma** özelliği eklendi.
#     - **Veri tabanı entegrasyonu** için kaynakça metinleri işlenebilir hale getirildi.
#     - **Gelişmiş hata yönetimi** ve detaylı loglama eklendi.
#     """

#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
#         self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

#     def dokuman_id_al(self, json_dosya_yolu):
#         """
#         📌 **JSON dosyasından Zotero benzersiz kimliğini çıkarır.**
        
#         Args:
#             json_dosya_yolu (str): JSON dosyasının tam yolu.
        
#         Returns:
#             list: Zotero kimlikleri listesi veya None (hata durumunda).
#         """
#         import json

#         try:
#             with open(json_dosya_yolu, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             zotero_idler = [item.get("key") for item in data if isinstance(item, dict) and "key" in item]
#             return zotero_idler if zotero_idler else None

#         except Exception as e:
#             config.logger.error(f"❌ JSON dosyasından ID okuma hatası: {str(e)}")
#             return None

#     def fetch_zotero_metadata(self, item_key):
#         """
#         📌 **Zotero'dan belirli bir öğenin metadatasını çeker.**
        
#         Args:
#             item_key (str): Zotero'daki dokümanın benzersiz kimliği.
        
#         Returns:
#             dict veya None: Eğer başarıyla çekilirse JSON verisi, yoksa None.
#         """
#         try:
#             response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
#             if response.status_code == 200:
#                 return response.json()
#             else:
#                 config.logger.error(f"⚠️ Zotero API Hatası! Kod: {response.status_code}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"❌ Zotero bağlantı hatası: {str(e)}")
#             return None

#     def meta_veri_al(self, item_key):
#         """
#         📌 **Belirtilen Zotero item key'e göre bibliyometrik verileri çeker.**
        
#         Args:
#             item_key (str): Zotero item key.
            
#         Returns:
#             dict veya None: Çekilen bibliyometrik veriler; hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"✅ Zotero API'den veri çekildi: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"❌ Zotero API hatası {response.status_code}: {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"❌ Zotero API isteğinde hata: {e}")
#             return None

#     def referanslari_analiz_et(self, referans_listesi):
#         """
#         📌 **Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.**
        
#         Args:
#             referans_listesi (list): Kaynakça referans metinlerinin listesi.
            
#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#                   Örnek: `[{"orijinal": "Smith, 2020. Title...", "yazar": "Smith"}, ...]`
#         """
#         analiz_sonuc = []
#         try:
#             for referans in referans_listesi:
#                 # Regex ile referansın başındaki ilk kelimeyi yazar olarak kabul ediyoruz.
#                 match = re.search(r'^([\w\.\-]+)', referans)
#                 yazar = match.group(1) if match else "Bilinmeyen"
#                 analiz_sonuc.append({
#                     "orijinal": referans,
#                     "yazar": yazar
#                 })
#             config.logger.info("✅ Referans analizi tamamlandı.")
#             return analiz_sonuc
#         except Exception as e:
#             config.logger.error(f"❌ Referans analizi hatası: {e}")
#             return []

#     def kaynakca_kaydet(self, pdf_id, kaynakca_listesi):
#         """
#         📌 **Kaynakça metinlerini veritabanına kaydeder.**
        
#         Args:
#             pdf_id (str): PDF dosyasının benzersiz ID'si.
#             kaynakca_listesi (list): Kaynakça referanslarının listesi.
            
#         Returns:
#             bool: Kaydetme işleminin başarılı olup olmadığı.
#         """
#         try:
#             if not kaynakca_listesi:
#                 config.logger.warning(f"⚠️ {pdf_id} için kaynakça boş!")
#                 return False

#             from processing_manager import IslemYoneticisi
#             islem_yoneticisi = IslemYoneticisi()
#             bib_collection = islem_yoneticisi.chroma_client.get_or_create_collection(name="pdf_bibliography")

#             for index, referans in enumerate(kaynakca_listesi):
#                 bib_collection.add(
#                     ids=[f"{pdf_id}_ref_{index}"],
#                     documents=[referans]
#                 )

#             config.logger.info(f"✅ {pdf_id} için {len(kaynakca_listesi)} referans başarıyla kaydedildi.")
#             return True

#         except Exception as e:
#             config.logger.error(f"❌ {pdf_id} için kaynakça kaydedilirken hata oluştu: {e}")
#             return False



# import re
# import requests
# from config_module import config

# class ZoteroEntegratoru:
#     import re
# import requests
# from config_module import config

# class ZoteroEntegratoru:
#     """
#     Zotero API entegrasyonunu yöneten sınıf.
    
#     Bu sınıf:
#     - Zotero API'si aracılığıyla belirtilen item key'e göre bibliyometrik verileri çeker.
#     - Verilen referans listesini analiz edip, basit regex kullanarak yazar isimlerini çıkarmayı sağlar.
#     """
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
#         self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    
#     def dokuman_id_al(self, dosya_adi):
#         """
#         📌 Dosya adından Zotero benzersiz kimliğini çıkarır.
        
#         Args:
#             dosya_adi (str): PDF veya TXT dosya adı.
        
#         Returns:
#             str: Zotero'daki benzersiz doküman ID'si.
#         """
#         match = re.search(r'([A-Z0-9]{8})', dosya_adi)  # Zotero ID'leri genellikle 8 karakter uzunluğundadır.
#         if match:
#             return match.group(1)
#         return None  # Eğer eşleşme bulunamazsa None döndür.

#     def fetch_zotero_metadata(self, item_key):
#         """
#         📌 Zotero'dan belirli bir öğenin metadatasını çeker.
        
#         Args:
#             item_key (str): Zotero'daki dokümanın benzersiz kimliği.
        
#         Returns:
#             dict veya None: Eğer başarıyla çekilirse JSON verisi, yoksa None.
#         """
#         base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items/{item_key}"
#         headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

#         try:
#             response = requests.get(base_url, headers=headers)
#             if response.status_code == 200:
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API Hatası! Kod: {response.status_code}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero bağlantı hatası: {str(e)}")
#             return None

#     def meta_veri_al(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri çeker.
        
#         Args:
#             item_key (str): Zotero item key.
            
#         Returns:
#             dict or None: Çekilen bibliyometrik veriler; hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri çekildi: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası {response.status_code}: {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def referanslari_analiz_et(self, referans_listesi):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.
        
#         Args:
#             referans_listesi (list): Kaynakça referans metinlerinin listesi.
            
#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#                   Örnek: [{"orijinal": "Smith, 2020. Title...", "yazar": "Smith"}, ...]
#         """
#         analiz_sonuc = []
#         try:
#             for referans in referans_listesi:
#                 # Basit regex: Referans metninin başındaki ilk kelimeyi yazar olarak kabul ediyoruz.
#                 match = re.search(r'^([\w\.\-]+)', referans)
#                 yazar = match.group(1) if match else "Bilinmeyen"
#                 analiz_sonuc.append({
#                     "orijinal": referans,
#                     "yazar": yazar
#                 })
#             config.logger.info("Referans analizi tamamlandı.")
#             return analiz_sonuc
#         except Exception as e:
#             config.logger.error(f"Referans analizi hatası: {e}")
#             return []

    # """
    # Zotero API entegrasyonunu yöneten sınıf.
    
    # Bu sınıf:
    # - Zotero API'si aracılığıyla belirtilen item key'e göre bibliyometrik verileri çeker.
    # - Verilen referans listesini analiz edip, basit regex kullanarak yazar isimlerini çıkarmayı sağlar.
    # """
    # def __init__(self):
    #     self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
    #     self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    
    # def dokuman_id_al(dosya_adi):
    # """
    # 📌 Dosya adından Zotero benzersiz kimliğini çıkarır.
    
    # Args:
    #     dosya_adi (str): PDF veya TXT dosya adı.
    
    # Returns:
    #     str: Zotero'daki benzersiz doküman ID'si.
    # """
    # match = re.search(r'([A-Z0-9]{8})', dosya_adi)  # Zotero ID'leri genellikle 8 karakter uzunluğundadır.
    # if match:
    #     return match.group(1)
    # return None  # Eğer eşleşme bulunamazsa None döndür.

    # def fetch_zotero_metadata(item_key):
    # """
    # 📌 Zotero'dan belirli bir öğenin metadatasını çeker.
    
    # Args:
    #     item_key (str): Zotero'daki dokümanın benzersiz kimliği.
    
    # Returns:
    #     dict veya None: Eğer başarıyla çekilirse JSON verisi, yoksa None.
    # """
    # import requests
    # from config_module import config

    # base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items/{item_key}"
    # headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

    # try:
    #     response = requests.get(base_url, headers=headers)
    #     if response.status_code == 200:
    #         return response.json()
    #     else:
    #         config.logger.error(f"Zotero API Hatası! Kod: {response.status_code}")
    #         return None
    # except Exception as e:
    #     config.logger.error(f"Zotero bağlantı hatası: {str(e)}")
    #     return None

    # def meta_veri_al(self, item_key):
    #     """
    #     Belirtilen Zotero item key'e göre bibliyometrik verileri çeker.
        
    #     Args:
    #         item_key (str): Zotero item key.
            
    #     Returns:
    #         dict or None: Çekilen bibliyometrik veriler; hata durumunda None.
    #     """
    #     url = f"{self.base_url}/{item_key}"
    #     try:
    #         response = requests.get(url, headers=self.headers, timeout=10)
    #         if response.status_code == 200:
    #             config.logger.info(f"Zotero API'den veri çekildi: {item_key}")
    #             return response.json()
    #         else:
    #             config.logger.error(f"Zotero API hatası {response.status_code}: {url}")
    #             return None
    #     except Exception as e:
    #         config.logger.error(f"Zotero API isteğinde hata: {e}")
    #         return None

    # def referanslari_analiz_et(self, referans_listesi):
    #     """
    #     Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.
        
    #     Args:
    #         referans_listesi (list): Kaynakça referans metinlerinin listesi.
            
    #     Returns:
    #         list: Her referans için analiz sonucu içeren sözlüklerin listesi.
    #               Örnek: [{"orijinal": "Smith, 2020. Title...", "yazar": "Smith"}, ...]
    #     """
    #     analiz_sonuc = []
    #     try:
    #         for referans in referans_listesi:
    #             # Basit regex: Referans metninin başındaki ilk kelimeyi yazar olarak kabul ediyoruz.
    #             match = re.search(r'^([\w\.\-]+)', referans)
    #             yazar = match.group(1) if match else "Bilinmeyen"
    #             analiz_sonuc.append({
    #                 "orijinal": referans,
    #                 "yazar": yazar
    #             })
    #         config.logger.info("Referans analizi tamamlandı.")
    #         return analiz_sonuc
    #     except Exception as e:
    #         config.logger.error(f"Referans analizi hatası: {e}")
    #         return []
    
# Açıklamalar
# Modülün Amacı:
# Zotero API entegrasyonu sağlamak, verilen Zotero item key üzerinden bibliyometrik verileri çekmek ve referans listesindeki verilerden
# (örneğin, "Smith, 2020. ..." gibi) yazar isimlerini çıkarmak.

# Fonksiyonlar:

# meta_veri_al(item_key): Zotero API'ye istek gönderir. İstek sırasında HTTP hataları veya istisnalar try/except ile yakalanır; 
# durum kodu 200 değilse hata loglanır ve None döndürülür.
# referanslari_analiz_et(referans_listesi): Her referans metninin başından basit bir regex ile yazar ismini alır; hata durumunda boş bir liste döndürülür.
# Değişkenler & Veri Yapıları:

# self.base_url, self.headers: API çağrıları için kullanılan temel URL ve header bilgileri.
# referans_listesi: Liste olarak verilen kaynakça metinleri.
# analiz_sonuc: Her referans için oluşturulan sözlükleri içeren liste.
# Kontrol Yapıları:

# try/except blokları ile API çağrıları ve referans analizi sırasında oluşabilecek hatalar yakalanır ve loglanır.
# Modüller & Kütüphaneler:

# re (regex işlemleri), requests (HTTP istekleri) kullanılır.
# config_module.py üzerinden yapılandırma (örneğin, API anahtarları, loglama) sağlanır.
# Bu final sürümü, tartışmalarımız doğrultusunda tüm gereksinimleri karşılayacak şekilde oluşturulmuştur. 
# Eğer başka bir geliştirme veya ekleme isterseniz, lütfen belirtin.