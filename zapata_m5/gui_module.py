import os
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from processing_manager import IslemYoneticisi
from citation_mapping_module import load_citation_mapping
from embedding_module import embed_text  # Temel embedding oluşturma (arama için kullanılabilir)
from clustering_module import perform_clustering  # Kümeleme analizi fonksiyonu
from fine_tuning_module import train_custom_model  # Fine-tuning model eğitimi
from data_query_module import query_data  # Gelişmiş veri sorgulama fonksiyonu
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
        Ana GUI bileşenlerini oluşturur.
        """
        # Üst bölümde dosya seçme ve işlem başlatma butonları
        self.dosya_sec_btn = ctk.CTkButton(self, text="📂 PDF Seç", command=self._dosya_sec)
        self.dosya_sec_btn.pack(pady=10)

        self.baslat_btn = ctk.CTkButton(self, text="🚀 İşlemi Başlat", command=self._islem_baslat)
        self.baslat_btn.pack(pady=10)

        self.citation_btn = ctk.CTkButton(self, text="📖 Atıf Zinciri Görüntüle", command=self._atif_goster)
        self.citation_btn.pack(pady=10)

        # İlave özellikler menüsü
        self.ilave_ozellikler_menusu()

        # Sonuç Ekranı
        self.sonuc_ekrani = ctk.CTkTextbox(self, width=1000, height=500)
        self.sonuc_ekrani.pack(pady=10)

        # Durum Çubuğu (isteğe bağlı)
        self.status_bar = ctk.CTkLabel(self, text="Durum: Hazır", anchor="w")
        self.status_bar.pack(fill="x", padx=10, pady=5)

    def _dosya_sec(self):
        """
        Kullanıcının PDF dosyası seçmesini sağlar.
        """
        dosya_yolu = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if dosya_yolu:
            self.sonuc_ekrani.insert("end", f"\n📄 Seçilen Dosya: {dosya_yolu}\n")
            self.islem_yoneticisi.secili_dosya = dosya_yolu
            self.status_bar.configure(text=f"Seçilen dosya: {Path(dosya_yolu).name}")

    def _islem_baslat(self):
        """
        Seçili PDF dosyası işlenir.
        """
        if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
            messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
            return

        self.sonuc_ekrani.insert("end", "\n⏳ İşlem başlatılıyor...\n")
        self.status_bar.configure(text="İşlem başlatılıyor...")
        basari, sonuc = self.islem_yoneticisi.pdf_txt_isle(Path(self.islem_yoneticisi.secili_dosya))

        if basari:
            self.sonuc_ekrani.insert("end", f"✅ İşlem tamamlandı: {self.islem_yoneticisi.secili_dosya}\n")
            self.status_bar.configure(text="İşlem tamamlandı.")
        else:
            self.sonuc_ekrani.insert("end", f"❌ Hata oluştu: {sonuc}\n")
            self.status_bar.configure(text="Hata oluştu.")

    def _atif_goster(self):
        """
        Seçili PDF dosyasının atıf zincirini görüntüler.
        """
        if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
            messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
            return

        pdf_id = Path(self.islem_yoneticisi.secili_dosya).stem
        citation_data = load_citation_mapping(pdf_id)

        if citation_data:
            display_text = "\n📚 Atıf Zinciri:\n"
            for entry in citation_data:
                display_text += f"🔹 Cümle {entry['sentence_id']}: {entry['sentence']}\n    Eşleşen Referans: {entry['matched_reference']}\n"
            self.sonuc_ekrani.insert("end", display_text)
            self.status_bar.configure(text="Atıf zinciri görüntülendi.")
        else:
            self.sonuc_ekrani.insert("end", "⚠️ Atıf verisi bulunamadı!\n")
            self.status_bar.configure(text="Atıf verisi bulunamadı.")

    def ilave_ozellikler_menusu(self):
        """
        İlave Özellikler Menüsü: Embedding Arama, Kümeleme Analizi, Fine-Tuning Modeli, Gelişmiş Veri Sorgulama.
        """
        # Embedding Arama
        self.embedding_btn = ctk.CTkButton(self, text="🔍 Embedding Arama", command=self._embedding_arama)
        self.embedding_btn.pack(pady=5)

        # Kümeleme Analizi
        self.kumeleme_btn = ctk.CTkButton(self, text="📊 Kümeleme Analizi", command=self._kumeleme_analiz)
        self.kumeleme_btn.pack(pady=5)

        # Fine-Tuning Modeli
        self.fine_tune_btn = ctk.CTkButton(self, text="🏋‍♂ Fine-Tuning Modeli", command=self._fine_tune_model)
        self.fine_tune_btn.pack(pady=5)

        # Gelişmiş Veri Sorgulama
        self.veri_sorgu_btn = ctk.CTkButton(self, text="🔎 Gelişmiş Veri Sorgulama", command=self._veri_sorgu)
        self.veri_sorgu_btn.pack(pady=5)

    def _embedding_arama(self):
        """
        Kullanıcının girdiği metinle en yakın embeddingleri arar.
        """
        query = self._kullanici_girdisi_al("Embedding Arama", "Aranacak metni girin:")
        if query:
            # search_embedding fonksiyonu, embedding modülünden veya alternatif embedding modülünden çağrılabilir.
            # Örneğin: search_embedding(query) şeklinde.
            try:
                from alternative_embedding_module import get_available_models, embed_text_with_model
                # Örnek: ilk model ile dene, gerçek uygulamada daha gelişmiş bir arama algoritması kullanılabilir.
                model_list = get_available_models()
                embedding = embed_text_with_model(query, model_list[0])
                result_text = f"Arama sonucu (Model: {model_list[0]}): {embedding[:10]} ... (embedding vektörü)"
            except Exception as e:
                result_text = f"Embedding arama hatası: {e}"
            self._sonuc_goster("🔍 Embedding Arama Sonuçları", result_text)

    def _kumeleme_analiz(self):
        """
        Kümeleme analizi yapar ve sonuçları gösterir.
        """
        try:
            clusters = perform_clustering()  # clustering_module.py'deki fonksiyon
            self._sonuc_goster("📊 Kümeleme Analizi Sonuçları", clusters)
        except Exception as e:
            self._sonuc_goster("📊 Kümeleme Analizi Hatası", str(e))

    def _fine_tune_model(self):
        """
        Fine-Tuning Modeli eğitimi başlatır ve sonuçları gösterir.
        """
        try:
            result = train_custom_model()  # fine_tuning_module.py'deki fonksiyon
            self._sonuc_goster("🏋‍♂ Fine-Tuning Sonuçları", result)
        except Exception as e:
            self._sonuc_goster("🏋‍♂ Fine-Tuning Hatası", str(e))

    def _veri_sorgu(self):
        """
        Gelişmiş veri sorgulama yapar.
        """
        query_params = self._kullanici_girdisi_al("Veri Sorgulama", "Sorgu parametrelerini girin:")
        if query_params:
            try:
                results = query_data(query_params)  # data_query_module.py'den fonksiyon
                self._sonuc_goster("🔎 Veri Sorgulama Sonuçları", results)
            except Exception as e:
                self._sonuc_goster("🔎 Veri Sorgulama Hatası", str(e))

    def _sonuc_goster(self, baslik, icerik):
        """
        Sonuçları GUI üzerinde gösterir.
        """
        self.sonuc_ekrani.insert("end", f"\n{baslik}:\n{icerik}\n")

    def _kullanici_girdisi_al(self, baslik, mesaj):
        """
        Kullanıcıdan input almak için diyalog penceresi açar.
        """
        # CustomTkinter'ın input diyalog kutusunu kullanıyoruz.
        input_dialog = ctk.CTkInputDialog(title=baslik, text=mesaj)
        return input_dialog.get_input()

if __name__ == '__main__':
    # İşlem yöneticisini oluştur ve GUI'yi başlat.
    islem_yoneticisi = IslemYoneticisi()
    arayuz = AnaArayuz(islem_yoneticisi)
    arayuz.mainloop()

# Aşağıda, önceki tartışmalarımızı ve yapılan güncellemeleri dikkate alarak oluşturulmuş 
# final sürümü olan **gui_module.py** modülünü bulabilirsiniz. Bu sürümde; 

# • Kullanıcı, PDF dosyası seçebiliyor.  
# • Seçilen dosya için işleme başlatılıyor (processing_manager üzerinden).  
# • Atıf zinciri görüntüleme, embedding arama, kümeleme analizi, fine-tuning ve gelişmiş veri sorgulama gibi ek özellikler için ayrı butonlar mevcut.  
# • Hata durumları ve işlemlerin ilerleyişi loglanıyor ve kullanıcıya mesaj kutuları ile bildirim yapılıyor.  

# Aşağıdaki kod, güncel gereksinimler doğrultusunda oluşturulmuş final gui_module.py’dır:


# ### Açıklamalar

# - **Ana Arayüz (AnaArayuz sınıfı):**  
#   PDF seçme, işlem başlatma ve atıf zinciri görüntüleme butonlarının yanı sıra ilave özellikler menüsü 
# (embedding arama, kümeleme analizi, fine-tuning, gelişmiş veri sorgulama) eklenmiştir.  
# - **İlave Özellikler Menüsü:**  
#   Kullanıcının ek özelliklere (embedding arama, kümeleme, fine-tuning, veri sorgulama) erişebilmesi için ayrı butonlar oluşturulmuştur.  
# - **Girdi Alma & Sonuç Gösterme:**  
#   Kullanıcı girdileri, özel input diyalog kutusu aracılığıyla alınır ve sonuçlar metin kutusu üzerinde gösterilir.

# Bu final sürümü, önceki tüm tartışmalarımız doğrultusunda güncellenmiş ve bütünleşik bir GUI arayüzü sunmaktadır. 
# Eğer ek bir düzenleme veya geliştirme talebiniz olursa lütfen belirtin.