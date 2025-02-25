# module_gui_and_user_interaction

# Below is the implementation of the `module_gui_and_user_interaction` module. This module focuses on creating a graphical user interface (GUI) for interacting with the system, including functionalities such as embedding search, clustering analysis, fine-tuning dataset export, and citation chain visualization.
# 
# ---

### **`module_gui_and_user_interaction.py`**

```python
import customtkinter as ctk
import tkinter.messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# ----------------------------
# GUI Class Definition
# ----------------------------

class AdditionalFeaturesGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Ek Özellikler")
        self.geometry("950x800")
        
        # Menu Bar
        self.menu = ctk.CTkMenu(self)
        self.config(menu=self.menu)
        self.file_menu = ctk.CTkMenu(self.menu, tearoff=0)
        self.file_menu.add_command(label="Veri Yükle", command=self.load_data)
        self.file_menu.add_command(label="Çıkış", command=self.quit)
        self.menu.add_cascade(label="Dosya", menu=self.file_menu)
        
        # Embedding Search Section
        self.search_label = ctk.CTkLabel(self, text="Embedding Arama", font=("Arial", 16))
        self.search_label.pack(pady=10)
        self.search_entry = ctk.CTkEntry(self, placeholder_text="Arama sorgusunu girin...")
        self.search_entry.pack(pady=5, padx=20, fill="x")
        self.search_button = ctk.CTkButton(self, text="Ara", command=self.embedding_search)
        self.search_button.pack(pady=5)
        
        # Clustering Analysis Restart
        self.cluster_label = ctk.CTkLabel(self, text="Kümelenme Analizi Yeniden Çalıştırma", font=("Arial", 16))
        self.cluster_label.pack(pady=20)
        self.cluster_entry = ctk.CTkEntry(self, placeholder_text="Küme sayısı (varsayılan: 5)")
        self.cluster_entry.pack(pady=5, padx=20, fill="x")
        self.cluster_button = ctk.CTkButton(self, text="Analizi Başlat", command=self.restart_clustering_analysis)
        self.cluster_button.pack(pady=5)
        
        # Fine-Tuning Dataset Export
        self.fine_tune_label = ctk.CTkLabel(self, text="Fine-Tuning Veri Seti Dışa Aktar", font=("Arial", 16))
        self.fine_tune_label.pack(pady=20)
        self.fine_tune_button = ctk.CTkButton(self, text="Veri Setini Dışa Aktar", command=self.export_fine_tuning_dataset)
        self.fine_tune_button.pack(pady=5)
        
        # Fine-Tuning Preparation (Tokenization) Start
        self.tokenization_label = ctk.CTkLabel(self, text="Fine-Tuning Hazırlık (LLaMA Tokenization)", font=("Arial", 16))
        self.tokenization_label.pack(pady=20)
        self.tokenization_button = ctk.CTkButton(self, text="Fine-Tuning Başlat", command=self.prepare_fine_tuning)
        self.tokenization_button.pack(pady=5)
        
        # Citation Chain Visualization
        self.citation_label = ctk.CTkLabel(self, text="Atıf Zinciri Görüntüleme", font=("Arial", 16))
        self.citation_label.pack(pady=20)
        self.citation_entry = ctk.CTkEntry(self, placeholder_text="Atıf zinciri için kayıt indeksi girin...")
        self.citation_entry.pack(pady=5, padx=20, fill="x")
        self.citation_button = ctk.CTkButton(self, text="Atıf Zincirini Görüntüle", command=self.view_citation_chain)
        self.citation_button.pack(pady=5)
        
        # Data Search and Query
        self.data_search_label = ctk.CTkLabel(self, text="Veri Tarama ve Sorgulama", font=("Arial", 16))
        self.data_search_label.pack(pady=20)
        self.data_search_entry = ctk.CTkEntry(self, placeholder_text="Sorgu girin...")
        self.data_search_entry.pack(pady=5, padx=20, fill="x")
        self.data_search_button = ctk.CTkButton(self, text="Sorgulama Yap", command=self.perform_data_search)
        self.data_search_button.pack(pady=5)
        
        # Results Display Area
        self.result_text = ctk.CTkTextbox(self, width=900, height=300)
        self.result_text.pack(pady=20, padx=20)

    # ----------------------------
    # GUI Functions
    # ----------------------------

    def load_data(self):
        """Loads data into the system."""
        tkinter.messagebox.showinfo("Bilgi", "Veri yükleme işlemi burada gerçekleştirilecek.")

    def embedding_search(self):
        """Performs an embedding search based on user input."""
        query = self.search_entry.get()
        if not query:
            tkinter.messagebox.showwarning("Uyarı", "Lütfen bir arama sorgusu girin.")
            return
        if not cluster_results_list:
            tkinter.messagebox.showwarning("Uyarı", "Arama için yeterli veri mevcut değil.")
            return
        try:
            texts = [result["content"] for result in cluster_results_list]
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
                    "Özet": texts[idx][:200] + "..."
                })
            df_results = pd.DataFrame(results)
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", df_results.to_string(index=False))
        except Exception as e:
            tkinter.messagebox.showerror("Hata", f"Embedding arama sırasında hata: {e}")

    def restart_clustering_analysis(self):
        """Restarts clustering analysis with user-defined parameters."""
        n_clusters = self.cluster_entry.get()
        n_clusters = int(n_clusters) if n_clusters.isdigit() else 5
        try:
            cluster_analysis_from_chromadb(cluster_results_list, n_clusters=n_clusters, output_dir="cluster_results")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "Kümelenme analizi tamamlandı. Sonuçlar 'cluster_results/cluster_results.csv' dosyasına kaydedildi.\n")
        except Exception as e:
            tkinter.messagebox.showerror("Hata", f"Kümeleme analizi sırasında hata: {e}")

    def export_fine_tuning_dataset(self):
        """Exports the fine-tuning dataset to a CSV file."""
        if not cluster_results_list:
            tkinter.messagebox.showwarning("Uyarı", "Fine-tuning için veri seti oluşturulacak veri bulunamadı.")
            return
        try:
            df = pd.DataFrame(cluster_results_list)
            output_file = "fine_tuning_dataset.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"Fine-tuning veri seti '{output_file}' olarak kaydedildi.\n")
        except Exception as e:
            tkinter.messagebox.showerror("Hata", f"Veri seti dışa aktarılırken hata: {e}")

    def prepare_fine_tuning(self):
        """Prepares data for fine-tuning using LLaMA tokenization."""
        try:
            if os.path.exists("fine_tuning_dataset.csv"):
                df = pd.read_csv("fine_tuning_dataset.csv")
                texts = df["content"].tolist()
            else:
                texts = [record["content"] for record in cluster_results_list if "content" in record]
            if not texts:
                tkinter.messagebox.showwarning("Uyarı", "Fine-tuning için yeterli veri bulunamadı.")
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
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "Fine-Tuning Hazırlık Sonuçları:\n" + summary)
        except Exception as e:
            tkinter.messagebox.showerror("Hata", f"Fine-tuning hazırlık sırasında hata: {e}")

    def view_citation_chain(self):
        """Displays the citation chain for a given index."""
        idx_str = self.citation_entry.get()
        if not idx_str.isdigit():
            tkinter.messagebox.showwarning("Uyarı", "Lütfen geçerli bir indeks girin.")
            return
        idx = int(idx_str)
        if idx < 0 or idx >= len(cluster_results_list):
            tkinter.messagebox.showwarning("Uyarı", "Girilen indeks mevcut değil.")
            return
        record = cluster_results_list[idx]
        bib_data = record.get("bibliography", "No bibliographic data available.")
        try:
            bib_json = json.loads(bib_data)
            bib_str = json.dumps(bib_json, indent=2, ensure_ascii=False)
        except Exception:
            bib_str = str(bib_data)
        chain_report = f"Atıf Zinciri Raporu (Kayıt {idx}):\n{bib_str}\n"
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", chain_report)

    def perform_data_search(self):
        """Performs a data search based on user input."""
        query = self.data_search_entry.get()
        if not query:
            tkinter.messagebox.showwarning("Uyarı", "Lütfen bir sorgu girin.")
            return
        try:
            texts = [result["content"] for result in cluster_results_list]
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
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", df_results.to_string(index=False))
        except Exception as e:
            tkinter.messagebox.showerror("Hata", f"Veri tarama/sorgulama sırasında hata: {e}")


# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    app = AdditionalFeaturesGUI()
    app.mainloop()
# ```
# 
# ---
# 
# ### **Module Description**
# 
# This module provides a GUI for interacting with various functionalities of the system. The main features include:
# 
# 1. **Embedding Search**:
#    - Allows users to search for embeddings based on a query.
#    - Uses TF-IDF vectorization and cosine similarity to find relevant results.
# 
# 2. **Clustering Analysis Restart**:
#    - Enables users to restart clustering analysis with a specified number of clusters.
#    - Results are saved to a CSV file.
# 
# 3. **Fine-Tuning Dataset Export**:
#    - Exports the current dataset to a CSV file for fine-tuning purposes.
# 
# 4. **Fine-Tuning Preparation**:
#    - Prepares data for fine-tuning by tokenizing text using the LLaMA tokenizer.
#    - Displays tokenization statistics.
# 
# 5. **Citation Chain Visualization**:
#    - Displays the citation chain for a given index in the dataset.
# 
# 6. **Data Search and Query**:
#    - Allows users to perform a data search