
b.py
import os
import re
import threading
from datetime import datetime
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import chromadb
from openai import OpenAI

from config_module import config
from zotero_module import ZoteroEntegratoru  # Zotero modülü entegrasyonu
from pdf_processing import (
    extract_text_from_pdf,
    reflow_columns,
    map_pdf_before_extraction,  # Yeni adlandırma
    detect_columns,
    extract_references_enhanced  # Referans çıkarma
)
from embedding_module import embed_text
from helper_module import stack_yukle, stack_guncelle, shorten_title

class IslemYoneticisi:
    """
    📌 **PDF/TXT dosyalarını işleme sürecini yöneten ana sınıf.**

    **İş Akışı:**
      1️⃣ **Dosya tipi** (.pdf veya .txt) belirlenir.
      2️⃣ **Stack güncellenir** (dosya işleme başlıyor olarak işaretlenir).
      3️⃣ **PDF için:**
         - Önce `map_pdf_before_extraction` ile yapısal haritalama yapılır.
         - Sonra `extract_text_from_pdf` ile metin çıkarılır.
      4️⃣ **TXT için:** 
         - Dosya doğrudan okunur.
         - `map_scientific_sections_extended` ile bölümler haritalanır.
      5️⃣ **Metin tek akışa dönüştürülür (`reflow_columns`).**
      6️⃣ **Bilimsel bölümler (`map_scientific_sections_extended`) yeniden tespit edilir.**
      7️⃣ **Sütun yapısı (`detect_columns`) belirlenir.**
      8️⃣ **Kaynakça (`extract_references_enhanced`) çıkarılır.**
      9️⃣ **Zotero entegrasyonu:**
         - `dokuman_id_al` ile temel dosya ID'si alınır.
         - `shorten_title` ile kısaltılır.
         - `fetch_zotero_metadata` ile bibliyografik veriler çekilir.
     🔟 **Embedding oluşturma (`embed_text`).**
     🔟 **İşlenen veriler kaydedilir.**
     🔟 **Stack'ten kaldırılır ve sayaçlar güncellenir.**

    ✅ **Çıktı:** İşlenmiş dosyaya ait tüm verileri içeren bir sözlük.
    """

    def __init__(self):
        self.stack_lock = threading.Lock()
        self.kume_sonuclari = []
        self.sayaçlar = {'toplam': 0, 'başarılı': 0, 'hata': 0}

        # ChromaDB bağlantısı ve koleksiyonlarının oluşturulması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.koleksiyon = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        self.zotero_koleksiyon = self.chroma_client.get_or_create_collection(name="zotero_meta")
        self.zotero = ZoteroEntegratoru()
        self.secili_dosya = None

    def pdf_txt_isle(self, dosya_yolu):
        """
        📌 **Bir PDF veya TXT dosyasını işler ve tüm verileri çıkarır.**
        """
        try:
            # İşleme başlamadan önce stack güncellemesi
            self.stack_guncelle(dosya_yolu.name, "ekle")
            config.logger.info(f"📄 {dosya_yolu.name} işleme başladı.")

            ext = dosya_yolu.suffix.lower()
            if ext == ".pdf":
                # 📌 **PDF için işlem akışı**
                harita = map_pdf_before_extraction(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
                ham_metin = extract_text_from_pdf(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
            elif ext == ".txt":
                # 📌 **TXT için işlem akışı**
                with open(dosya_yolu, "r", encoding="utf-8") as f:
                    ham_metin = f.read()
                harita = map_scientific_sections_extended(ham_metin)
            else:
                config.logger.error(f"❌ Desteklenmeyen dosya uzantısı: {dosya_yolu}")
                return None

            if not ham_metin:
                raise ValueError("❌ Ham metin çıkarılamadı.")

            # 📌 **Metni tek akışa dönüştürme**
            temiz_metin = reflow_columns(ham_metin)

            # 📌 **Bilimsel bölümlerin haritalanması**
            bolum_haritasi = map_scientific_sections_extended(ham_metin)

            # 📌 **Sütun yapısı tespiti**
            sutun_bilgisi = detect_columns(ham_metin)

            # 📌 **Kaynakça çıkarımı**
            try:
                references = extract_references_enhanced(ham_metin)
            except Exception as e:
                config.logger.error(f"❌ Kaynakça çıkarım hatası: {e}")
                references = []

            # 📌 **Zotero entegrasyonu**
            dosya_id = self.zotero.dokuman_id_al(dosya_yolu.name)
            if not dosya_id:
                dosya_id = dosya_yolu.stem
            dosya_id = shorten_title(dosya_id, max_length=80)
            zotero_meta = self.zotero.fetch_zotero_metadata(dosya_id)

            # 📌 **Embedding oluşturma**
            embedding = embed_text(temiz_metin)

            # 📌 **Sonuç sözlüğü hazırlanıyor**
            result = {
                "dosya": dosya_yolu.name,
                "ham_metin": ham_metin,
                "temiz_metin": temiz_metin,
                "harita": harita,
                "bolum_haritasi": bolum_haritasi,
                "sutun_bilgisi": sutun_bilgisi,
                "kaynakca": references,
                "zotero_meta": zotero_meta,
                "embedding": embedding,
                "islem_tarihi": datetime.now().isoformat()
            }

            # 📌 **Stack güncelleme ve sayaç artırma**
            self.stack_guncelle(dosya_yolu.name, "sil")
            self.sayaçlar['başarılı'] += 1
            config.logger.info(f"✅ {dosya_yolu.name} başarıyla işlendi.")
            return result
        except Exception as e:
            self.sayaçlar['hata'] += 1
            config.logger.error(f"❌ {dosya_yolu.name} işlenirken hata: {e}", exc_info=True)
            return None

    def stack_guncelle(self, dosya_adi, islem):
        """
        📌 **Stack güncelleme işlemini `helper_module` üzerinden gerçekleştirir.**
        """
        from helper_module import stack_guncelle
        with self.stack_lock:
            stack_guncelle(dosya_adi, islem)

a.py
# import os
# import re
# import threading
# from datetime import datetime
# from pathlib import Path
# import multiprocessing
# from tqdm import tqdm
# import chromadb
# from openai import OpenAI

# from config_module import config
# from zotero_module import ZoteroEntegratoru  # Buradaki hatayı düzelttik!
# from pdf_processing import (
#     extract_text_from_pdf,
#     reflow_columns,
#     map_scientific_sections_extended,
#     map_pdf_before_extraction,
#     detect_columns,
#     extract_references_enhanced
# )
# from embedding_module import embed_text
# from helper_module import stack_yukle, stack_guncelle, shorten_title

# class IslemYoneticisi:
#     """
#     PDF/TXT dosyalarını işleme sürecini yöneten ana sınıf.
    
#     İş Akışı:
#       1. Dosya tipi (.pdf veya .txt) belirlenir.
#       2. İşleme başlamadan önce, dosya adı stack'e eklenir.
#       3. PDF ise: 
#          - İlk olarak map_pdf_before_extraction ile yapısal haritalama yapılır.
#          - extract_text_from_pdf ile ham metin çıkarılır.
#          TXT ise: dosya doğrudan okunur ve map_scientific_sections_extended ile bölümler haritalanır.
#       4. Çıkarılan metin, reflow_columns ile tek akışa dönüştürülür.
#       5. Bilimsel bölümler map_scientific_sections_extended ile yeniden tespit edilir.
#       6. Sütun yapısı detect_columns ile belirlenir.
#       7. Kaynakça extract_references_enhanced ile çıkarılır.
#       8. Zotero entegrasyonu: dokuman_id_al ile temel dosya ID'si alınır, shorten_title ile kısaltılır, fetch_zotero_metadata ile bibliyometrik veriler çekilir.
#       9. Temiz metin üzerinden embed_text fonksiyonu ile embedding oluşturulur.
#      10. Tüm bu bilgiler, bir sonuç sözlüğünde toplanır.
#      11. İşlem sonunda, dosya adı stack'ten kaldırılır ve sayaçlar güncellenir.
    
#     Döndürdüğü sonuç sözlüğü, işlenmiş dosyaya ait tüm bilgileri içerir.
#     """
# def __init__(self):
#         self.stack_lock = threading.Lock()
#         self.kume_sonuclari = []
#         self.sayaçlar = {'toplam': 0, 'başarılı': 0, 'hata': 0}

#         # ChromaDB bağlantısı ve koleksiyonlarının oluşturulması
#         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
#         self.koleksiyon = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
#         self.zotero_koleksiyon = self.chroma_client.get_or_create_collection(name="zotero_meta")
#         self.zotero = ZoteroEntegratoru()  # HATA BURADA: Önceki versiyonda eksik tanımlama var
#         self.secili_dosya = None

#     def pdf_txt_isle(self, dosya_yolu):
#         try:
#             # İşleme başlamadan önce stack güncellemesi
#             self.stack_guncelle(dosya_yolu.name, "ekle")
#             config.logger.info(f"{dosya_yolu.name} işleme başladı.")
            
#             ext = dosya_yolu.suffix.lower()
#             if ext == ".pdf":
#                 # PDF için: önce yapısal haritalama, sonra metin çıkarımı
#                 harita = map_pdf_before_extraction(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
#                 ham_metin = extract_text_from_pdf(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
#             elif ext == ".txt":
#                 with open(dosya_yolu, "r", encoding="utf-8") as f:
#                     ham_metin = f.read()
#                 harita = map_scientific_sections_extended(ham_metin)
#             else:
#                 config.logger.error(f"Desteklenmeyen dosya uzantısı: {dosya_yolu}")
#                 return None

#             if not ham_metin:
#                 raise ValueError("Ham metin çıkarılamadı.")

#             # Metni tek akışa dönüştürme (reflow)
#             temiz_metin = reflow_columns(ham_metin)

#             # Bilimsel bölümlerin haritalanması
#             bolum_haritasi = map_scientific_sections_extended(ham_metin)

#             # Sütun yapısı tespiti
#             sutun_bilgisi = detect_columns(ham_metin)

#             # Kaynakça çıkarımı (hata yönetimiyle)
#             try:
#                 references = extract_references_enhanced(ham_metin)
#             except Exception as e:
#                 config.logger.error(f"Kaynakça çıkarım hatası: {e}")
#                 references = []

#             # Zotero entegrasyonu: Dosya temel ID'si alınır ve kısaltılır
#             dosya_id = dokuman_id_al(dosya_yolu.name)
#             if not dosya_id:
#                 dosya_id = dosya_yolu.stem
#             dosya_id = shorten_title(dosya_id, max_length=80)
#             zotero_meta = fetch_zotero_metadata(dosya_id)

#             # Embedding oluşturma (temiz metin üzerinden)
#             embedding = embed_text(temiz_metin)

#             # Sonuç sözlüğü hazırlanıyor
#             result = {
#                 "dosya": dosya_yolu.name,
#                 "ham_metin": ham_metin,
#                 "temiz_metin": temiz_metin,
#                 "harita": harita,
#                 "bolum_haritasi": bolum_haritasi,
#                 "sutun_bilgisi": sutun_bilgisi,
#                 "kaynakca": references,
#                 "zotero_meta": zotero_meta,
#                 "embedding": embedding,
#                 "islem_tarihi": datetime.now().isoformat()
#             }

#             self.stack_guncelle(dosya_yolu.name, "sil")
#             self.sayaçlar['başarılı'] += 1
#             config.logger.info(f"{dosya_yolu.name} başarıyla işlendi.")
#             return result
#         except Exception as e:
#             self.sayaçlar['hata'] += 1
#             config.logger.error(f"{dosya_yolu.name} işlenirken hata: {e}", exc_info=True)
#             return None

#     def stack_guncelle(self, dosya_adi, islem):
#         """
#         Stack güncelleme işlemini helper_module üzerinden gerçekleştirir.
#         """
#         from helper_module import stack_guncelle
#         with self.stack_lock:
#             stack_guncelle(dosya_adi, islem)
 
# b.py
# import os
# import re
# import threading
# from datetime import datetime
# from pathlib import Path
# import multiprocessing
# from tqdm import tqdm
# import chromadb
# from openai import OpenAI

# from config_module import config
# from zotero_module import ZoteroEntegratoru  # Hata düzetildi, doğru sınıf çağırıldı!
# from pdf_processing import (
#     extract_text_from_pdf,
#     reflow_columns,
#     map_scientific_sections_extended,
#     map_pdf_before_extraction,
#     detect_columns,
#     extract_references_enhanced
# )
# from embedding_module import embed_text
# from helper_module import stack_yukle, stack_guncelle, shorten_title

# class IslemYoneticisi:
#     """
#     📌 **PDF/TXT dosyalarını işleme sürecini yöneten ana sınıf.**

#     **İş Akışı:**
#       1️⃣ Dosya tipi (.pdf veya .txt) belirlenir.
#       2️⃣ İşleme başlamadan önce, dosya adı stack'e eklenir.
#       3️⃣ **PDF ise:** 
#          - İlk olarak `map_pdf_before_extraction` ile yapısal haritalama yapılır.
#          - `extract_text_from_pdf` ile ham metin çıkarılır.
#          - **TXT ise:** doğrudan okunur ve `map_scientific_sections_extended` ile bölümler haritalanır.
#       4️⃣ Çıkarılan metin `reflow_columns` ile tek akışa dönüştürülür.
#       5️⃣ Bilimsel bölümler `map_scientific_sections_extended` ile yeniden tespit edilir.
#       6️⃣ **Sütun yapısı** `detect_columns` ile belirlenir.
#       7️⃣ **Kaynakça** `extract_references_enhanced` ile çıkarılır.
#       8️⃣ **Zotero entegrasyonu:** `dokuman_id_al` ile temel dosya ID'si alınır, `shorten_title` ile kısaltılır, `fetch_zotero_metadata` ile bibliyometrik veriler çekilir.
#       9️⃣ **Embedding** `embed_text` fonksiyonu ile oluşturulur.
#       🔟 Tüm bilgiler bir **sonuç sözlüğünde** toplanır.
#       ✅ İşlem sonunda, dosya adı stack'ten kaldırılır ve sayaçlar güncellenir.
#     """

#     def __init__(self):
#         self.stack_lock = threading.Lock()
#         self.kume_sonuclari = []
#         self.sayaçlar = {'toplam': 0, 'başarılı': 0, 'hata': 0}

#         # ChromaDB bağlantısı ve koleksiyonlarının oluşturulması
#         self.chroma_client = chromadb.PersistentClient(path="chroma_db")
#         self.koleksiyon = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
#         self.zotero_koleksiyon = self.chroma_client.get_or_create_collection(name="zotero_meta")
#         self.zotero = ZoteroEntegratoru()  # 🛠️ HATA DÜZELTİLDİ!
#         self.secili_dosya = None

#     def pdf_txt_isle(self, dosya_yolu):
#         """
#         📌 **Belirtilen dosyayı işler.**  
#         - PDF veya TXT dosyası olabilir.
#         - Zotero ile entegre çalışır.
#         - Kaynakçaları çıkarır, embedding oluşturur ve temiz metin oluşturur.

#         Args:
#             dosya_yolu (Path): İşlenecek dosyanın tam yolu.

#         Returns:
#             dict veya None: Başarılı işlenirse detaylı sonuç sözlüğü döndürülür. Hata olursa None.
#         """
#         try:
#             # İşleme başlamadan önce stack güncellemesi
#             self.stack_guncelle(dosya_yolu.name, "ekle")
#             config.logger.info(f"🚀 {dosya_yolu.name} işleme başladı.")

#             ext = dosya_yolu.suffix.lower()
#             if ext == ".pdf":
#                 # PDF için: önce yapısal haritalama, sonra metin çıkarımı
#                 harita = map_pdf_before_extraction(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
#                 ham_metin = extract_text_from_pdf(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
#             elif ext == ".txt":
#                 with open(dosya_yolu, "r", encoding="utf-8") as f:
#                     ham_metin = f.read()
#                 harita = map_scientific_sections_extended(ham_metin)
#             else:
#                 config.logger.error(f"❌ Desteklenmeyen dosya uzantısı: {dosya_yolu}")
#                 return None

#             if not ham_metin:
#                 raise ValueError("❌ Ham metin çıkarılamadı.")

#             # Metni tek akışa dönüştürme (reflow)
#             temiz_metin = reflow_columns(ham_metin)

#             # Bilimsel bölümlerin haritalanması
#             bolum_haritasi = map_scientific_sections_extended(ham_metin)

#             # Sütun yapısı tespiti
#             sutun_bilgisi = detect_columns(ham_metin)

#             # Kaynakça çıkarımı (hata yönetimiyle)
#             try:
#                 references = extract_references_enhanced(ham_metin)
#             except Exception as e:
#                 config.logger.error(f"⚠️ Kaynakça çıkarım hatası: {e}")
#                 references = []

#             # Zotero entegrasyonu: Dosya temel ID'si alınır ve kısaltılır
#             dosya_id = self.zotero.dokuman_id_al(dosya_yolu.name)  # ❗ Düzeltme yapıldı
#             if not dosya_id:
#                 dosya_id = dosya_yolu.stem
#             dosya_id = shorten_title(dosya_id, max_length=80)
#             zotero_meta = self.zotero.fetch_zotero_metadata(dosya_id)  # ❗ Düzeltme yapıldı

#             # Embedding oluşturma (temiz metin üzerinden)
#             embedding = embed_text(temiz_metin)

#             # Sonuç sözlüğü hazırlanıyor
#             result = {
#                 "dosya": dosya_yolu.name,
#                 "ham_metin": ham_metin,
#                 "temiz_metin": temiz_metin,
#                 "harita": harita,
#                 "bolum_haritasi": bolum_haritasi,
#                 "sutun_bilgisi": sutun_bilgisi,
#                 "kaynakca": references,
#                 "zotero_meta": zotero_meta,
#                 "embedding": embedding,
#                 "islem_tarihi": datetime.now().isoformat()
#             }

#             # İşlem tamamlandı, stack'ten kaldır
#             self.stack_guncelle(dosya_yolu.name, "sil")
#             self.sayaçlar['başarılı'] += 1
#             config.logger.info(f"✅ {dosya_yolu.name} başarıyla işlendi.")
#             return result
#         except Exception as e:
#             self.sayaçlar['hata'] += 1
#             config.logger.error(f"❌ {dosya_yolu.name} işlenirken hata: {e}", exc_info=True)
#             return None

#     def stack_guncelle(self, dosya_adi, islem):
#         """
#         📌 **Stack güncelleme işlemini `helper_module` üzerinden gerçekleştirir.**
#         - İşlem **"ekle"** veya **"sil"** olabilir.

#         Args:
#             dosya_adi (str): Dosya adı.
#             islem (str): "ekle" veya "sil".
#         """
#         from helper_module import stack_guncelle
#         with self.stack_lock:
#             stack_guncelle(dosya_adi, islem)


# Aşağıda, önceki tartışmalarımızı, güncellemeleri ve geliştirme önerilerini 
# (örn. hata yönetimi, loglama, çoklu iş parçacığı desteği, dosya tipi ayırt etme,
#  Zotero entegrasyonu, metin işleme, embedding, bölüm haritalama, kaynakça çıkarımı vb.) 
# dikkate alarak oluşturulmuş final sürümünü bulabilirsiniz. Aşağıdaki kod, **`processing_manager.py`** modülünün final versiyonudur:

# ### Açıklamalar

# - **Modülün Amacı:**  
#   PDF veya TXT dosyalarını işleyip, ham metin çıkarımı, metin reflow işlemi, bilimsel bölümlerin haritalanması, sütun ve kaynakça çıkarımı, Zotero entegrasyonu, embedding oluşturma gibi tüm iş akışını yöneten ana sınıfı (IslemYoneticisi) içerir.

# - **Fonksiyonlar ve İşlevleri:**  
#   - **`pdf_txt_isle(dosya_yolu)`**:  
#     - Dosya türüne göre (PDF/TXT) uygun metin çıkarımı yapılır.
#     - PDF dosyalarında önce `map_pdf_before_extraction` ile yapısal haritalama, ardından `extract_text_from_pdf` ile ham metin çıkarılır.
#     - TXT dosyalarında dosya doğrudan okunur.
#     - Metin `reflow_columns` ile temizlenir.
#     - Bilimsel bölümler, `map_scientific_sections_extended` ile tespit edilir.
#     - Sütun yapısı `detect_columns` ile belirlenir.
#     - Kaynakça, `extract_references_enhanced` ile çıkarılır.
#     - Zotero entegrasyonu: `dokuman_id_al` ve `fetch_zotero_metadata` kullanılarak dosya ID'si ve bibliyometrik veriler çekilir.
#     - Temiz metin üzerinden embedding, `embed_text` ile oluşturulur.
#     - İşlem sonucu, tüm bu bilgileri içeren bir sözlük olarak döndürülür.
#     - İşlem başladığında dosya stack'e eklenir, işlem bitiminde stack'ten kaldırılır.
#   - **`stack_guncelle(dosya_adi, islem)`**:  
#     - Yardımcı modülden alınan `stack_guncelle` fonksiyonu kullanılarak, dosya stack'i güncellenir (ekle/sil).

# - **Değişkenler ve Veri Yapıları:**  
#   - `sayaçlar`: Toplam, başarılı ve hatalı dosya sayısını takip eden bir sözlük.
#   - `kume_sonuclari`: İşlemden elde edilen sonuçların saklandığı liste.
#   - `chroma_client`, `koleksiyon`, `zotero_koleksiyon`: Vektör veritabanı ve Zotero ile ilgili koleksiyonlar.
#   - `secili_dosya`: GUI üzerinden seçilen dosya yolu (varsa).

# - **Kontrol Yapıları:**  
#   - Dosya tipi kontrolü için if/elif kullanılıyor.
#   - Hata yönetimi try/except blokları ile yapılmış; her adımda hatalar loglanıyor.

# - **Kütüphaneler:**  
#   - `chromadb`, `openai`, `tqdm`, `threading` gibi kütüphaneler kullanılarak çoklu iş parçacığı ve hata yönetimi sağlanmış.

# Bu final sürümü, tartışmalarımız ve yapılan güncellemeler doğrultusunda tüm temel iş akışını 
# sağlam ve kapsamlı bir şekilde yerine getiriyor. Her fonksiyon, gerekli hata yönetimi, 
# loglama ve kullanıcı/ortam ayarlarını dikkate alarak hazırlanmıştır.

# Eğer ek bir soru veya geliştirme talebiniz varsa, lütfen belirtin.