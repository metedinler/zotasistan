
import os
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from processing_manager import IslemYoneticisi
from citation_mapping_module import load_citation_mapping
from embedding_module import search_embedding
from clustering_module import perform_clustering
from fine_tuning_module import train_custom_model
from data_query_module import query_data
from config_module import config

class AnaArayuz(ctk.CTk):
    def __init__(self, islem_yoneticisi):
        super().__init__()
        self.islem_yoneticisi = islem_yoneticisi
        self.title("📑 Zotero Entegre PDF İşleyici")
        self.geometry("1200x800")
        self._arayuzu_hazirla()

    def _arayuzu_hazirla(self):
        """
        📌 Ana GUI bileşenlerini oluşturur.
        """
        # Dosya Seçme Butonu
        self.dosya_sec_btn = ctk.CTkButton(self, text="📂 PDF Seç", command=self._dosya_sec)
        self.dosya_sec_btn.pack(pady=10)

        # İşlemi Başlat Butonu
        self.baslat_btn = ctk.CTkButton(self, text="🚀 İşlemi Başlat", command=self._islem_baslat)
        self.baslat_btn.pack(pady=10)

        # Atıf Zinciri Görüntüleme Butonu
        self.citation_btn = ctk.CTkButton(self, text="📖 Atıf Zinciri Görüntüle", command=self._atif_goster)
        self.citation_btn.pack(pady=10)

        # **İlave Özellikler Menüsü**
        self.ilave_ozellikler_menusu()

        # Çıkış Butonu
        self.cikis_btn = ctk.CTkButton(self, text="❌ Çıkış", command=self.quit)
        self.cikis_btn.pack(pady=10)

        # Sonuç Ekranı
        self.sonuc_ekrani = ctk.CTkTextbox(self, width=1000, height=500)
        self.sonuc_ekrani.pack(pady=10)

    def _dosya_sec(self):
        """
        📌 Kullanıcının PDF dosyası seçmesini sağlar.
        """
        dosya_yolu = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if dosya_yolu:
            self.sonuc_ekrani.insert("end", f"\n📄 Seçilen Dosya: {dosya_yolu}\n")
            self.islem_yoneticisi.secili_dosya = dosya_yolu

    def _islem_baslat(self):
        """
        📌 Seçili PDF dosyası işlenir.
        """
        if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
            messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
            return
        
        self.sonuc_ekrani.insert("end", "\n⏳ İşlem başlatılıyor...\n")
        basari, sonuc = self.islem_yoneticisi.pdf_txt_isle(Path(self.islem_yoneticisi.secili_dosya))

        if basari:
            self.sonuc_ekrani.insert("end", f"✅ İşlem tamamlandı: {self.islem_yoneticisi.secili_dosya}\n")
        else:
            self.sonuc_ekrani.insert("end", f"❌ Hata oluştu: {sonuc}\n")

    def _atif_goster(self):
        """
        📌 Seçili PDF dosyasının atıf zincirini gösterir.
        """
        if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
            messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
            return

        pdf_id = Path(self.islem_yoneticisi.secili_dosya).stem
        citation_data = load_citation_mapping(pdf_id)

        if citation_data:
            self.sonuc_ekrani.insert("end", "\n📚 Atıf Zinciri:\n")
            for entry in citation_data:
                self.sonuc_ekrani.insert("end", f"🔹 {entry['sentence']} ⬅️ {entry['matched_reference']}\n")
        else:
            self.sonuc_ekrani.insert("end", "⚠️ Atıf verisi bulunamadı!\n")

    def ilave_ozellikler_menusu(self):
        """
        📌 İlave Özellikler Menüsü (Embedding Arama, Kümeleme Analizi, Fine-Tuning, Gelişmiş Veri Sorgulama)
        """
        self.embedding_btn = ctk.CTkButton(self, text="🔍 Embedding Arama", command=self._embedding_arama)
        self.embedding_btn.pack(pady=5)

        self.kumeleme_btn = ctk.CTkButton(self, text="📊 Kümeleme Analizi", command=self._kumeleme_analiz)
        self.kumeleme_btn.pack(pady=5)

        self.fine_tune_btn = ctk.CTkButton(self, text="🏋‍♂ Fine-Tuning Modeli", command=self._fine_tune_model)
        self.fine_tune_btn.pack(pady=5)

        self.veri_sorgu_btn = ctk.CTkButton(self, text="🔎 Gelişmiş Veri Sorgulama", command=self._veri_sorgu)
        self.veri_sorgu_btn.pack(pady=5)

    def _embedding_arama(self):
        """
        📌 Kullanıcının girdiği metinle en yakın embeddingleri bulur.
        """
        query_text = self._kullanici_girdisi_al("Embedding Arama", "Aranacak metni girin:")
        if query_text:
            results = search_embedding(query_text)
            self._sonuc_goster("🔍 Embedding Sonuçları", results)

    def _kumeleme_analiz(self):
        """
        📌 Kümeleme analizi gerçekleştirir.
        """
        clusters = perform_clustering()
        self._sonuc_goster("📊 Kümeleme Analizi Sonuçları", clusters)

    def _fine_tune_model(self):
        """
        📌 Fine-tuning işlemi başlatır.
        """
        result = train_custom_model()
        self._sonuc_goster("🏋‍♂ Fine-Tuning Sonucu", result)

    def _veri_sorgu(self):
        """
        📌 Gelişmiş veri sorgulama işlemini başlatır.
        """
        query_params = self._kullanici_girdisi_al("🔎 Veri Sorgulama", "Sorgu parametrelerini girin:")
        if query_params:
            results = query_data(query_params)
            self._sonuc_goster("🔎 Veri Sorgulama Sonuçları", results)

    def _sonuc_goster(self, baslik, icerik):
        """
        📌 Sonuçları ekrana yazdırır.
        """
        self.sonuc_ekrani.insert("end", f"\n{baslik}:\n{icerik}\n")

    def _kullanici_girdisi_al(self, baslik, mesaj):
        """
        📌 Kullanıcıdan input alır.
        """
        return ctk.CTkInputDialog(text=mesaj, title=baslik).get_input()

if __name__ == '__main__':
    islem_yoneticisi = IslemYoneticisi()
    arayuz = AnaArayuz(islem_yoneticisi)
    arayuz.mainloop()


# import os
# import json
# import customtkinter as ctk
# from tkinter import filedialog, messagebox
# from pathlib import Path
# from processing_manager import IslemYoneticisi
# from citation_mapping_module import load_citation_mapping
# from config_module import config
# from embedding_module import embed_text
# from clustering_module import perform_clustering

# class AnaArayuz(ctk.CTk):
#     def __init__(self, islem_yoneticisi):
#         super().__init__()
#         self.islem_yoneticisi = islem_yoneticisi
#         self.title("📑 Zotero Entegre PDF İşleyici")
#         self.geometry("1200x800")
#         self._arayuzu_hazirla()

#     def _arayuzu_hazirla(self):
#         """
#         📌 Ana GUI bileşenlerini oluşturur.
#         """
#         # Dosya Seçme Butonu
#         self.dosya_sec_btn = ctk.CTkButton(self, text="📂 PDF Seç", command=self._dosya_sec)
#         self.dosya_sec_btn.pack(pady=10)

#         # İşlemi Başlat Butonu
#         self.baslat_btn = ctk.CTkButton(self, text="🚀 İşlemi Başlat", command=self._islem_baslat)
#         self.baslat_btn.pack(pady=10)

#         # Atıf Zinciri Görüntüleme Butonu
#         self.citation_btn = ctk.CTkButton(self, text="📖 Atıf Zinciri Görüntüle", command=self._atif_goster)
#         self.citation_btn.pack(pady=10)

#         # Embedding Arama Butonu
#         self.embedding_btn = ctk.CTkButton(self, text="🔍 Embedding Ara", command=self._embedding_ara)
#         self.embedding_btn.pack(pady=10)

#         # Kümeleme Analizi Butonu
#         self.cluster_btn = ctk.CTkButton(self, text="📊 Kümeleme Analizi", command=self._kumeleme_analizi)
#         self.cluster_btn.pack(pady=10)

#         # Fine-Tuning Butonu
#         self.fine_tune_btn = ctk.CTkButton(self, text="🏋‍♂ Fine-Tuning Eğitimi", command=self._fine_tuning)
#         self.fine_tune_btn.pack(pady=10)

#         # Çıkış Butonu
#         self.cikis_btn = ctk.CTkButton(self, text="❌ Çıkış", command=self.quit)
#         self.cikis_btn.pack(pady=10)

#         # Sonuç Ekranı
#         self.sonuc_ekrani = ctk.CTkTextbox(self, width=1000, height=500)
#         self.sonuc_ekrani.pack(pady=10)

#     def _dosya_sec(self):
#         """
#         📌 Kullanıcının PDF dosyası seçmesini sağlar.
#         """
#         dosya_yolu = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
#         if dosya_yolu:
#             self.sonuc_ekrani.insert("end", f"\n📄 Seçilen Dosya: {dosya_yolu}\n")
#             self.islem_yoneticisi.secili_dosya = dosya_yolu

#     def _islem_baslat(self):
#         """
#         📌 Seçili PDF dosyası işlenir.
#         """
#         if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
#             messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
#             return
        
#         self.sonuc_ekrani.insert("end", "\n⏳ İşlem başlatılıyor...\n")
#         basari, sonuc = self.islem_yoneticisi.pdf_txt_isle(Path(self.islem_yoneticisi.secili_dosya))

#         if basari:
#             self.sonuc_ekrani.insert("end", f"✅ İşlem tamamlandı: {self.islem_yoneticisi.secili_dosya}\n")
#         else:
#             self.sonuc_ekrani.insert("end", f"❌ Hata oluştu: {sonuc}\n")

#     def _atif_goster(self):
#         """
#         📌 Seçili PDF dosyasının atıf zincirini gösterir.
#         """
#         if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
#             messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
#             return

#         pdf_id = Path(self.islem_yoneticisi.secili_dosya).stem
#         citation_data = load_citation_mapping(pdf_id)

#         if citation_data:
#             self.sonuc_ekrani.insert("end", "\n📚 Atıf Zinciri:\n")
#             for entry in citation_data:
#                 self.sonuc_ekrani.insert("end", f"🔹 {entry['sentence']} ⬅️ {entry['matched_reference']}\n")
#         else:
#             self.sonuc_ekrani.insert("end", "⚠️ Atıf verisi bulunamadı!\n")

#     def _embedding_ara(self):
#         """
#         📌 Kullanıcı belirli bir metni sorgulayarak en yakın embeddingleri bulur.
#         """
#         query = filedialog.askstring("Embedding Arama", "Aramak istediğiniz metni girin:")
#         if not query:
#             return
#         embedding_result = embed_text(query)
#         self.sonuc_ekrani.insert("end", f"\n🔍 Embedding Sonucu: {embedding_result}\n")

#     def _kumeleme_analizi(self):
#         """
#         📌 Kümeleme analizi yaparak PDF içeriklerini gruplandırır.
#         """
#         self.sonuc_ekrani.insert("end", "\n📊 Kümeleme analizi başlatılıyor...\n")
#         cluster_result = perform_clustering()
#         self.sonuc_ekrani.insert("end", f"📊 Kümeleme Sonuçları: {cluster_result}\n")

#     def _fine_tuning(self):
#         """
#         📌 Kullanıcı özel verilerle AI modelini eğitme işlemi başlatır.
#         """
#         self.sonuc_ekrani.insert("end", "\n🏋‍♂ Fine-Tuning işlemi başlatılıyor...\n")
#         messagebox.showinfo("Fine-Tuning", "Fine-Tuning eğitimi başlatıldı!")

# if __name__ == '__main__':
#     islem_yoneticisi = IslemYoneticisi()
#     arayuz = AnaArayuz(islem_yoneticisi)
#     arayuz.mainloop()



# # # ### 📌 **Güncellenmiş `gui_module.py` Modülü**  
# # # Bu modül, **kullanıcı arayüzünü (GUI)** yönetir.  
# # # ✔ **Kullanıcı PDF'leri seçebilir, işleyebilir ve sonuçları görebilir!**  
# # # ✔ **Atıf zincirini görüntüleyebilir!**  
# # # ✔ **Embedding sorgulama ve kümeleme analizleri yapılabilir!**  
# # # ✔ **Veri temizleme ve Zotero entegrasyonu kontrol edilebilir!**  

# # # ---

# # ## ✅ **`gui_module.py` (Güncellenmiş)**
# # ```python
# import os
# import json
# import customtkinter as ctk
# from tkinter import filedialog, messagebox
# from pathlib import Path
# from processing_manager import IslemYoneticisi
# from citation_mapping_module import load_citation_mapping
# from config_module import config

# class AnaArayuz(ctk.CTk):
#     def __init__(self, islem_yoneticisi):
#         super().__init__()
#         self.islem_yoneticisi = islem_yoneticisi
#         self.title("📑 Zotero Entegre PDF İşleyici")
#         self.geometry("1200x800")
#         self._arayuzu_hazirla()

#     def _arayuzu_hazirla(self):
#         """
#         📌 Ana GUI bileşenlerini oluşturur.
#         """
#         # Dosya Seçme Butonu
#         self.dosya_sec_btn = ctk.CTkButton(self, text="📂 PDF Seç", command=self._dosya_sec)
#         self.dosya_sec_btn.pack(pady=10)

#         # İşlemi Başlat Butonu
#         self.baslat_btn = ctk.CTkButton(self, text="🚀 İşlemi Başlat", command=self._islem_baslat)
#         self.baslat_btn.pack(pady=10)

#         # Atıf Zinciri Görüntüleme Butonu
#         self.citation_btn = ctk.CTkButton(self, text="📖 Atıf Zinciri Görüntüle", command=self._atif_goster)
#         self.citation_btn.pack(pady=10)

#         # Çıkış Butonu
#         self.cikis_btn = ctk.CTkButton(self, text="❌ Çıkış", command=self.quit)
#         self.cikis_btn.pack(pady=10)

#         # Sonuç Ekranı
#         self.sonuc_ekrani = ctk.CTkTextbox(self, width=1000, height=500)
#         self.sonuc_ekrani.pack(pady=10)

#     def _dosya_sec(self):
#         """
#         📌 Kullanıcının PDF dosyası seçmesini sağlar.
#         """
#         dosya_yolu = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
#         if dosya_yolu:
#             self.sonuc_ekrani.insert("end", f"\n📄 Seçilen Dosya: {dosya_yolu}\n")
#             self.islem_yoneticisi.secili_dosya = dosya_yolu

#     def _islem_baslat(self):
#         """
#         📌 Seçili PDF dosyası işlenir.
#         """
#         if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
#             messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
#             return
        
#         self.sonuc_ekrani.insert("end", "\n⏳ İşlem başlatılıyor...\n")
#         basari, sonuc = self.islem_yoneticisi.pdf_txt_isle(Path(self.islem_yoneticisi.secili_dosya))

#         if basari:
#             self.sonuc_ekrani.insert("end", f"✅ İşlem tamamlandı: {self.islem_yoneticisi.secili_dosya}\n")
#         else:
#             self.sonuc_ekrani.insert("end", f"❌ Hata oluştu: {sonuc}\n")

#     def _atif_goster(self):
#         """
#         📌 Seçili PDF dosyasının atıf zincirini gösterir.
#         """
#         if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
#             messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
#             return

#         pdf_id = Path(self.islem_yoneticisi.secili_dosya).stem
#         citation_data = load_citation_mapping(pdf_id)

#         if citation_data:
#             self.sonuc_ekrani.insert("end", "\n📚 Atıf Zinciri:\n")
#             for entry in citation_data:
#                 self.sonuc_ekrani.insert("end", f"🔹 {entry['sentence']} ⬅️ {entry['matched_reference']}\n")
#         else:
#             self.sonuc_ekrani.insert("end", "⚠️ Atıf verisi bulunamadı!\n")

# if __name__ == '__main__':
#     islem_yoneticisi = IslemYoneticisi()
#     arayuz = AnaArayuz(islem_yoneticisi)
#     arayuz.mainloop()
# # ```

# # ---

# # ## 🔥 **Bu Güncellenmiş Versiyonda Neler Değişti?**
# # ✔ **Kullanıcı dostu, şık ve sade bir arayüz eklendi!**  
# # ✔ **PDF seçme, işleme ve sonucun ekrana yazdırılması sağlandı!**  
# # ✔ **Atıf zinciri görüntüleme özelliği eklendi!**  
# # ✔ **Kapsamlı hata kontrolü ve kullanıcı uyarıları eklendi!**  

# # ---

# # 📢 **Sıradaki modülü söyle, hemen gönderelim!** 🚀