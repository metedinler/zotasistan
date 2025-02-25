import os
import json
import pandas as pd
import customtkinter as ctk
import tkinter.messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaTokenizer
from config_module import config
from helper_module import shorten_title
from file_save_module import save_citation_file
from citation_mapping_module import load_citation_mapping  # Fonksiyon, CITATION dosyasını yüklemek için

# Örnek: Kümeleme analizi fonksiyonunun basit bir implementasyonu
def cluster_analysis_from_chromadb(kume_sonuclari, n_clusters=5, output_dir="cluster_results"):
    try:
        vectorizer = TfidfVectorizer(max_features=1000)
        texts = [record["icerik"] for record in kume_sonuclari]
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        for i, cluster in enumerate(kmeans.labels_):
            kume_sonuclari[i]['cluster'] = int(cluster)
        os.makedirs(output_dir, exist_ok=True)
        cluster_data = pd.DataFrame({
            "Index": list(range(len(texts))),
            "Cluster": kmeans.labels_,
            "Summary": [text[:100] + "..." for text in texts]
        })
        cluster_file = os.path.join(output_dir, "cluster_results.csv")
        cluster_data.to_csv(cluster_file, index=False, encoding='utf-8')
        config.logger.info(f"Kümeleme analizi tamamlandı, sonuçlar {cluster_file} dosyasına kaydedildi.")
    except Exception as e:
        config.logger.error(f"Kümeleme analizi hatası: {str(e)}")

# Ek Özellikler paneli
class AnalizPaneli(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master  # master, AnaArayuz nesnesi
        self.kume_btn = ctk.CTkButton(self, text="Kümeleme Analizi Yap", command=self._kumeleme_baslat)
        self.kume_btn.pack(pady=10)

    def _kumeleme_baslat(self):
        try:
            if not self.master.kume_sonuclari:
                ctk.messagebox.showwarning("Uyarı", "Kümeleme için yeterli veri mevcut değil.")
                return
            n_input = self.klm_entry.get() if hasattr(self, "klm_entry") else "5"
            n_clusters = int(n_input) if n_input.isdigit() else 5
            cluster_analysis_from_chromadb(self.master.kume_sonuclari, n_clusters=n_clusters, output_dir="cluster_results")
            self.master.sonuc_text.delete("1.0", "end")
            self.master.sonuc_text.insert("1.0", f"Kümeleme analizi tamamlandı. Sonuçlar 'cluster_results/cluster_results.csv' dosyasına kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Kümeleme analizi sırasında hata: {e}")

# Ek özellikler için GUI penceresi
class AdditionalFeaturesGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ek Özellikler")
        self.geometry("950x800")
        self._arayuzu_hazirla()

    def _arayuzu_hazirla(self):
        # Embedding Arama Bölümü
        self.ara_label = ctk.CTkLabel(self, text="Embedding Arama", font=("Arial", 16))
        self.ara_label.pack(pady=10)
        self.ara_entry = ctk.CTkEntry(self, placeholder_text="Arama sorgusunu girin...")
        self.ara_entry.pack(pady=5, padx=20, fill="x")
        self.ara_button = ctk.CTkButton(self, text="Ara", command=self.embedding_arama)
        self.ara_button.pack(pady=5)

        # Kümeleme Analizi Yeniden Çalıştırma
        self.klm_label = ctk.CTkLabel(self, text="Kümeleme Analizi Yeniden Çalıştırma", font=("Arial", 16))
        self.klm_label.pack(pady=20)
        self.klm_entry = ctk.CTkEntry(self, placeholder_text="Küme sayısı (varsayılan: 5)")
        self.klm_entry.pack(pady=5, padx=20, fill="x")
        self.klm_button = ctk.CTkButton(self, text="Analizi Başlat", command=self.kumelenme_analizi)
        self.klm_button.pack(pady=5)

        # Fine-Tuning Veri Seti Dışa Aktarma
        self.ft_label = ctk.CTkLabel(self, text="Fine-Tuning Veri Seti Dışa Aktar", font=("Arial", 16))
        self.ft_label.pack(pady=20)
        self.ft_button = ctk.CTkButton(self, text="Veri Setini Dışa Aktar", command=self.export_fine_tuning_dataset)
        self.ft_button.pack(pady=5)

        # Fine-Tuning Hazırlık (LLaMA Tokenization)
        self.ft_prep_label = ctk.CTkLabel(self, text="Fine-Tuning Hazırlık (LLaMA Tokenization)", font=("Arial", 16))
        self.ft_prep_label.pack(pady=20)
        self.ft_prep_button = ctk.CTkButton(self, text="Fine-Tuning Başlat", command=self.fine_tuning_preparation_view)
        self.ft_prep_button.pack(pady=5)

        # Atıf Zinciri Görüntüleme
        self.citation_label = ctk.CTkLabel(self, text="Atıf Zinciri Görüntüleme", font=("Arial", 16))
        self.citation_label.pack(pady=20)
        self.citation_entry = ctk.CTkEntry(self, placeholder_text="Atıf zinciri için PDF dosya adını girin...")
        self.citation_entry.pack(pady=5, padx=20, fill="x")
        self.citation_button = ctk.CTkButton(self, text="Atıf Zincirini Görüntüle", command=self.citation_chain_view)
        self.citation_button.pack(pady=5)

        # Veri Tarama ve Sorgulama
        self.search_label = ctk.CTkLabel(self, text="Veri Tarama ve Sorgulama", font=("Arial", 16))
        self.search_label.pack(pady=20)
        self.search_entry = ctk.CTkEntry(self, placeholder_text="Sorgu girin...")
        self.search_entry.pack(pady=5, padx=20, fill="x")
        self.search_button = ctk.CTkButton(self, text="Sorgulama Yap", command=self.data_search)
        self.search_button.pack(pady=5)

        # Sonuç Görselleştirme Alanı
        self.sonuc_text = ctk.CTkTextbox(self, width=900, height=300)
        self.sonuc_text.pack(pady=20, padx=20)

    def embedding_arama(self):
        query = self.ara_entry.get()
        if not query:
            ctk.messagebox.showwarning("Uyarı", "Lütfen bir arama sorgusu girin.")
            return
        if not self.master.kume_sonuclari:
            ctk.messagebox.showwarning("Uyarı", "Arama için yeterli veri mevcut değil.")
            return
        try:
            texts = [record["icerik"] for record in self.master.kume_sonuclari]
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            sims = cosine_similarity(X, query_vec).flatten()
            top_indices = sims.argsort()[-10:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    "Index": idx,
                    "Benzerlik": round(sims[idx], 4),
                    "Özet": texts[idx][:100] + "..."
                })
            df_results = pd.DataFrame(results)
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", df_results.to_string(index=False))
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Embedding arama sırasında hata: {e}")

    def kumelenme_analizi(self):
        n = self.klm_entry.get()
        n_clusters = int(n) if n.isdigit() else 5
        try:
            from processing_manager import cluster_analysis_from_chromadb
            cluster_analysis_from_chromadb(self.master.kume_sonuclari, n_clusters=n_clusters, output_dir="cluster_results")
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", "Kümeleme analizi tamamlandı. Sonuçlar 'cluster_results/cluster_results.csv' dosyasına kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Kümeleme analizi sırasında hata: {e}")

    def export_fine_tuning_dataset(self):
        if not self.master.kume_sonuclari:
            ctk.messagebox.showwarning("Uyarı", "Fine-tuning için veri seti oluşturulacak veri bulunamadı.")
            return
        try:
            df = pd.DataFrame(self.master.kume_sonuclari)
            output_file = "fine_tuning_dataset.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", f"Fine-tuning veri seti '{output_file}' olarak kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Veri seti dışa aktarılırken hata: {e}")

    def fine_tuning_preparation_view(self):
        try:
            if os.path.exists("fine_tuning_dataset.csv"):
                df = pd.read_csv("fine_tuning_dataset.csv")
                texts = df["icerik"].tolist()
            else:
                texts = [record["icerik"] for record in self.master.kume_sonuclari if "icerik" in record]
            if not texts:
                ctk.messagebox.showwarning("Uyarı", "Fine-tuning için yeterli veri bulunamadı.")
                return
            tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
            tokenized_data = [tokenizer(text, truncation=True, max_length=512) for text in texts]
            token_counts = [len(item["input_ids"]) for item in tokenized_data]
            summary = (
                f"Toplam kayıt: {len(tokenized_data)}\n"
                f"Ortalama token sayısı: {sum(token_counts)/len(token_counts):.2f}\n"
                f"En fazla token: {max(token_counts)}\n"
                f"En az token: {min(token_counts)}\n"
            )
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", "Fine-Tuning Hazırlık Sonuçları:\n" + summary)
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Fine-tuning hazırlık sırasında hata: {e}")

    def citation_chain_view(self):
        """
        Kullanıcının girdiği PDF dosya adından, ilgili Citation Mapping dosyasını okur ve içeriği GUI'de görüntüler.
        """
        pdf_name = self.citation_entry.get().strip()
        if not pdf_name:
            ctk.messagebox.showwarning("Uyarı", "Lütfen görüntülenecek PDF dosya adını girin.")
            return
        # Temel dosya ID'sini belirleyin (dokuman_id_al ve shorten_title kullanılarak)
        from zotero_module import dokuman_id_al
        file_id = dokuman_id_al(pdf_name)
        if not file_id:
            file_id = pdf_name.split('.')[0]
        file_id = shorten_title(file_id, 80)
        citation_file = f"{file_id}.citation.json"
        citation_path = config.CITATIONS_DIR / citation_file
        if not citation_path.exists():
            ctk.messagebox.showwarning("Uyarı", f"Citation Mapping dosyası bulunamadı: {citation_path}")
            return
        try:
            with open(citation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # JSON verisini düzenli bir biçimde görüntüleyelim
            pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", pretty_json)
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Citation Mapping görüntülenirken hata: {e}")

    def data_search(self):
        query = self.search_entry.get()
        if not query:
            ctk.messagebox.showwarning("Uyarı", "Lütfen bir sorgu girin.")
            return
        try:
            texts = [result["icerik"] for result in self.master.kume_sonuclari]
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            sims = cosine_similarity(X, query_vec).flatten()
            top_indices = sims.argsort()[-10:][::-1]
            results = []
            for idx in top_indices:
                results.append({
                    "Index": idx,
                    "Benzerlik": round(sims[idx], 4),
                    "Özet": texts[idx][:100] + "..."
                })
            df_results = pd.DataFrame(results)
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", df_results.to_string(index=False))
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Veri tarama/sorgulama sırasında hata: {e}")

# Ana Arayüz
class AnaArayuz(ctk.CTk):
    def __init__(self, islem_yoneticisi):
        super().__init__()
        self.islem_yoneticisi = islem_yoneticisi
        self.kume_sonuclari = []  # İşlenen dosya sonuçlarının saklanacağı liste
        self.title("Zotero Entegre PDF/TXT İşleyici")
        self.geometry("1200x800")
        self._arayuzu_hazirla()

    def _arayuzu_hazirla(self):
        # Sol tarafta dosya listesi (opsiyonel)
        self.dosya_listesi = ctk.CTkListbox(self, width=400)
        self.dosya_listesi.pack(side="left", fill="both", padx=10, pady=10)
        # Üst kısımda işlem başlatma butonu
        self.baslat_btn = ctk.CTkButton(self, text="İşlemi Başlat", command=self._islem_baslat)
        self.baslat_btn.pack(side="top", padx=5, pady=5)
        # Sağ tarafta ek özellikler paneli
        self.ek_panel = AnalizPaneli(self)
        self.ek_panel.pack(side="right", fill="both", padx=10, pady=10)
        # Alt tarafta sonuçların görüntüleneceği metin kutusu
        self.sonuc_text = ctk.CTkTextbox(self, width=900, height=300)
        self.sonuc_text.pack(pady=20, padx=20)

    def _islem_baslat(self):
        from processing_manager import pdf_txt_isle
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        try:
            dosyalar = [os.path.join(config.STORAGE_DIR, f) for f in os.listdir(config.STORAGE_DIR)]
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = {executor.submit(pdf_txt_isle, Path(dosya)): dosya for dosya in dosyalar}
                for future in as_completed(futures):
                    dosya = futures[future]
                    try:
                        result = future.result()
                        if result:
                            config.logger.info(f"{dosya} başarıyla işlendi")
                            self.kume_sonuclari.append(result)
                    except Exception as e:
                        config.logger.error(f"{dosya} işlenirken hata: {str(e)}")
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", "Dosya işlemleri tamamlandı.\n")
        except Exception as e:
            config.logger.error(f"Ana iş akışı hatası: {str(e)}", exc_info=True)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    from processing_manager import IslemYoneticisi
    islem_yoneticisi = IslemYoneticisi()
    # STORAGE_DIR içerisindeki toplam dosya sayısını güncelledik
    islem_yoneticisi.sayaçlar['toplam'] = len(os.listdir(config.STORAGE_DIR))
    
    try:
        arayuz = AnaArayuz(islem_yoneticisi)
        arayuz.mainloop()
    except Exception as e:
        config.logger.critical(f"Ana uygulama hatası: {str(e)}", exc_info=True)
    finally:
        print("\nİstatistikler:")
        print(f"Toplam Dosya: {islem_yoneticisi.sayaçlar['toplam']}")
        print(f"Başarılı: {islem_yoneticisi.sayaçlar['başarılı']}")
        print(f"Hatalı: {islem_yoneticisi.sayaçlar['hata']}")
