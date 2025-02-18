import os
import re
import json
import fitz  # PyMuPDF
import shutil
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import logging
import sys
import requests
import colorlog
import pandas as pd
from datetime import datetime
import pytz
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import psutil
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# CustomTkinter GUI kütüphanesi
import customtkinter as ctk
import tkinter.messagebox

# ----------------------------
# Dizin ve Ortam Ayarları
# ----------------------------
BASE_DIR = r"C:\Users\mete\Zotero\zotai"  # Ana dizin (kutuphane)
STORAGE_DIR = os.getenv('STORAGE_DIR')  # PDF/TXT dosyalarının bulunduğu dizin
SUCCESS_DIR = os.getenv('SUCCESS_DIR')  # Çıktıların saklanacağı dizin
TEMIZMETIN_DIR = os.path.join(BASE_DIR, "TemizMetin")
TEMIZ_TABLO_DIR = os.path.join(BASE_DIR, "TemizTablo")
TEMIZ_KAYNAKCA_DIR = os.path.join(BASE_DIR, "TemizKaynakca")
EMBEDDING_DIR = os.path.join(SUCCESS_DIR, "embedingparca") if SUCCESS_DIR else os.path.join(os.getcwd(), "embedingparca")
PDF_DIR = os.path.join(SUCCESS_DIR, "pdfler") if SUCCESS_DIR else os.path.join(os.getcwd(), "pdfler")

# ----------------------------
# Loglama Yapılandırması
# ----------------------------
LOG_DOSYASI = os.path.join(BASE_DIR, "islem_logu.json")
HATA_LOG_DOSYASI = os.path.join(BASE_DIR, "hata_logu.json")
STACK_DOSYASI = os.path.join(BASE_DIR, "islem_stack.json")  # İşlem yığını takibi için

log_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'}
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------------------------
# API ve Client Ayarları
# ----------------------------
load_dotenv()
if not STORAGE_DIR:
    logger.warning("STORAGE_DIR ortam değişkeni tanımlı değil.")
if not SUCCESS_DIR:
    logger.warning("SUCCESS_DIR ortam değişkeni tanımlı değil.")
if not os.getenv('OPENAI_API_KEY'):
    logger.error("OPENAI_API_KEY tanımlı değil, OpenAI işlemleri başarısız olacaktır.")

ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
    logger.error("Zotero API anahtarı veya kullanıcı kimliği tanımlı değil.")
ZOTERO_API_URL = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items"
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

try:
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
    bib_collection = chroma_client.get_or_create_collection(name="pdf_bibliography")
except Exception as e:
    logger.error(f"ChromaDB bağlantısı kurulamadı: {e}")
    chroma_client = None

# ----------------------------
# Sayaçlar ve Takip Değişkenleri
# ----------------------------
total_files = 0
success_count = 0
embedding_failed_count = 0
text_extraction_failed_count = 0

# Global liste: Kümeleme ve arama için kullanılacak veriler
cluster_results_list = []


# ----------------------------
# Dosya Takip Fonksiyonları
# ----------------------------
def dosya_listesini_guncelle(new_files: list):
    try:
        if os.path.exists(LOG_DOSYASI):
            with open(LOG_DOSYASI, 'r', encoding='utf-8') as f:
                current_list = json.load(f)
        else:
            current_list = []
        updated_list = list(set(current_list + new_files))
        with open(LOG_DOSYASI, 'w', encoding='utf-8') as f:
            json.dump(updated_list, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Dosya listesi güncelleme hatası: {e}")


def kalan_dosyalari_oku() -> list:
    try:
        if os.path.exists(LOG_DOSYASI):
            with open(LOG_DOSYASI, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        logger.error(f"Kalan dosyaları okuma hatası: {e}")
        return []


def stack_yukle() -> set:
    try:
        if os.path.exists(STACK_DOSYASI):
            with open(STACK_DOSYASI, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        else:
            return set()
    except Exception as e:
        logger.error(f"Stack yükleme hatası: {e}")
        return set()


def stack_guncelle(dosya_adi: str, islem_tipi: str):
    try:
        stack = stack_yukle()
        stack.add(f"{dosya_adi}:{islem_tipi}")
        with open(STACK_DOSYASI, 'w', encoding='utf-8') as f:
            json.dump(list(stack), f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Stack güncelleme hatası: {e}")


# ----------------------------
# Yardımcı Fonksiyonlar (Metin İşleme)
# ----------------------------
def memory_usage():
    return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"


def clean_text(text):
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def split_text(text, chunk_size=256):
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks


def save_text_file(directory, filename, content):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as f:
        f.write(content)


# ----------------------------
# PDF İşleme Fonksiyonları
# ----------------------------
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        raw_text = ""
        for page in doc:
            page_text = page.get_text("text", sort=True)
            raw_text += page_text + "\n\n"
        doc.close()
        return raw_text
    except Exception as e:
        logger.error(f"❌ PDF metni çıkarılamadı: {pdf_path}, Hata: {e}")
        traceback.print_exc()
        return None


def detect_tables(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        tables = []
        for page in doc:
            text = page.get_text("text", sort=True)
            table_patterns = [
                (r'(?i)(Tablo\s*\d+)', 'tablo'),
                (r'(?i)(Table\s*\d+)', 'table'),
                (r'(?i)(Çizelge\s*\d+)', 'çizelge'),
                (r'(?i)(Figure\s*\d+)', 'figure')
            ]
            for pattern, tip in table_patterns:
                for match in re.finditer(pattern, text):
                    tables.append({'tip': tip, 'baslik': match.group(0)})
        doc.close()
        return tables
    except Exception as e:
        logger.error(f"❌ Tablo çıkarma hatası: {e}")
        return []


def reflow_columns(text):
    if "Abstract" in text:
        parts = text.split("Abstract", 1)
        abstract = parts[0].strip()
        body = parts[1].strip()
        body = re.sub(r'\n', ' ', body)
        body = re.sub(r'\s{2,}', ' ', body)
        return abstract + "\n\n" + body
    return text


# ----------------------------
# Gelişmiş Referans Çıkarma ve Kaydetme Fonksiyonları
# ----------------------------
def extract_references_enhanced(text):
    references = []
    ref_patterns = [
        r'(?i)(?:KAYNAKÇA|KAYNAKLAR|REFERENCES|BIBLIOGRAPHY).*?\n(.*?)(?=\n\s*\n|\Z)',
        r'(?i)(?:REFERANSLAR|KAYNAK LİSTESİ).*?\n(.*?)(?=\n\s*\n|\Z)',
        r'\n[\[\(](\d{4})[\]\)].*?(?=\n|$)',
        r'\n(?:[A-Za-z\u00C0-\u017F]+,\s+[A-Za-z\u00C0-\u017F\.]+\s+\(\d{4}\).*?)(?=\n|$)',
    ]
    text_lower = text.lower()
    start_indices = []
    ref_headers = ['kaynakça', 'kaynaklar', 'references', 'bibliography', 'referanslar', 'kaynak listesi']
    for header in ref_headers:
        idx = text_lower.find(header)
        if idx != -1:
            start_indices.append(idx)
    if start_indices:
        start_idx = min(start_indices)
        ref_section = text[start_idx:]
        lines = ref_section.split('\n')
        current_ref = ""
        for line in lines[1:]:
            line = line.strip()
            if not line:
                if current_ref:
                    references.append(current_ref)
                    current_ref = ""
            elif any(c.isdigit() for c in line):
                if current_ref:
                    references.append(current_ref)
                current_ref = line
            else:
                if current_ref:
                    current_ref += " " + line
                else:
                    current_ref = line
        if current_ref:
            references.append(current_ref)
    if not references:
        references = ["No references found."]
    cleaned_refs = []
    for ref in references:
        ref = re.sub(r'\s+', ' ', ref).strip()
        if len(ref) > 10 and any(c.isdigit() for c in ref):
            cleaned_refs.append(ref)
    if not cleaned_refs:
        cleaned_refs = ["No references found."]
    return cleaned_refs


def save_references(references, file_path, source_type="pdf"):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ref_file = os.path.join(TEMIZ_KAYNAKCA_DIR, f"{base_name}_references.txt")
    ref_json = os.path.join(TEMIZ_KAYNAKCA_DIR, f"{base_name}_references.json")
    ref_data = {
        "source_file": file_path,
        "source_type": source_type,
        "extraction_date": datetime.now().isoformat(),
        "references": references
    }
    try:
        with open(ref_file, 'w', encoding='utf-8') as f:
            f.write(f"Referanslar ({len(references)}):\n")
            f.write("-" * 50 + "\n")
            for i, ref in enumerate(references, 1):
                f.write(f"{i}. {ref}\n")
        with open(ref_json, 'w', encoding='utf-8') as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Referanslar kaydedildi: {ref_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Referanslar kaydedilirken hata: {e}")
        return False


def save_references_for_analysis(references, vosviewer_file, pajek_file):
    try:
        with open(vosviewer_file, 'w', encoding='utf-8') as vos_file:
            vos_file.write("label\n")
            for ref in references:
                vos_file.write(f"{ref}\n")
        logger.info(f"✅ VOSviewer formatında referanslar kaydedildi: {vosviewer_file}")
        with open(pajek_file, 'w', encoding='utf-8') as pajek_f:
            pajek_f.write("*Vertices {}\n".format(len(references)))
            for i, ref in enumerate(references, 1):
                pajek_f.write(f'{i} "{ref}"\n')
        logger.info(f"✅ Pajek formatında referanslar kaydedildi: {pajek_file}")
    except Exception as e:
        logger.error(f"❌ Referanslar analiz formatlarına kaydedilemedi: {e}")


# ----------------------------
# Embedding ve Zotero Fonksiyonları
# ----------------------------
def embed_text(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        global embedding_failed_count
        embedding_failed_count += 1
        logger.error(f"❌ Embedding oluşturulamadı: {e}")
        return None


def fetch_zotero_metadata(item_key):
    headers = {"Zotero-API-Key": ZOTERO_API_KEY}
    try:
        response = requests.get(f"{ZOTERO_API_URL}/{item_key}", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Zotero API isteğinde hata: {e}")
        return None


# ----------------------------
# Durum Takip Fonksiyonları (Index)
# ----------------------------
def save_last_processed_index(index):
    with open("last_processed_index.txt", "w", encoding="utf-8") as f:
        f.write(str(index))


def get_last_processed_index():
    if os.path.exists("last_processed_index.txt"):
        with open("last_processed_index.txt", "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0


# ----------------------------
# Kümeleme Analizi Fonksiyonu
# ----------------------------
def cluster_analysis_from_chromadb(embedding_results, n_clusters=5, output_dir="cluster_results"):
    try:
        texts = [result["content"] for result in embedding_results]
        bibliographies = [result["bibliography"] for result in embedding_results]
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        os.makedirs(output_dir, exist_ok=True)
        cluster_data = pd.DataFrame({
            "content": texts,
            "cluster": clusters,
            "bibliography": bibliographies
        })
        cluster_file = os.path.join(output_dir, "cluster_results.csv")
        cluster_data.to_csv(cluster_file, index=False, encoding='utf-8')
        print(f"✅ Kümelenme sonuçları '{cluster_file}' dosyasına kaydedildi.")
    except Exception as e:
        logger.error(f"Kümeleme analizi sırasında hata: {e}")


# ----------------------------
# Ek Özellikler Menüsü Fonksiyonları (GUI)
# ----------------------------
class AdditionalFeaturesGUI(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Ek Özellikler")
        self.geometry("900x700")
        self.menu = ctk.CTkMenu(self)
        self.config(menu=self.menu)
        self.file_menu = ctk.CTkMenu(self.menu, tearoff=0)
        self.file_menu.add_command(label="Veri Yükle", command=self.veri_yukle)
        self.file_menu.add_command(label="Çıkış", command=self.quit)
        self.menu.add_cascade(label="Dosya", menu=self.file_menu)
        
        # Embedding Arama Bölümü
        self.ara_label = ctk.CTkLabel(self, text="Embedding Arama", font=("Arial", 16))
        self.ara_label.pack(pady=10)
        self.ara_entry = ctk.CTkEntry(self, placeholder_text="Arama sorgusunu girin...")
        self.ara_entry.pack(pady=5, padx=20, fill="x")
        self.ara_button = ctk.CTkButton(self, text="Ara", command=self.embedding_arama)
        self.ara_button.pack(pady=5)
        
        # Kümelenme Analizi Yeniden Çalıştırma
        self.klm_label = ctk.CTkLabel(self, text="Kümelenme Analizi Yeniden Çalıştırma", font=("Arial", 16))
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
        
        # Atıf Zinciri Görüntüleme
        self.citation_label = ctk.CTkLabel(self, text="Atıf Zinciri Görüntüleme", font=("Arial", 16))
        self.citation_label.pack(pady=20)
        self.citation_entry = ctk.CTkEntry(self, placeholder_text="Atıf zinciri için kayıt indeksi girin...")
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
        self.sonuc_text = ctk.CTkTextbox(self, width=800, height=250)
        self.sonuc_text.pack(pady=20, padx=20)

    def veri_yukle(self):
        ctk.messagebox.showinfo("Bilgi", "Veri yükleme işlemi burada gerçekleştirilecek.")

    def embedding_arama(self):
        query = self.ara_entry.get()
        if not query:
            ctk.messagebox.showwarning("Uyarı", "Lütfen bir arama sorgusu girin.")
            return
        if not cluster_results_list:
            ctk.messagebox.showwarning("Uyarı", "Arama için yeterli veri mevcut değil.")
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
                    "Benzerlik": sims[idx],
                    "Özet": texts[idx][:200] + "..."
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
            cluster_analysis_from_chromadb(cluster_results_list, n_clusters=n_clusters, output_dir="cluster_results")
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", "Kümelenme analizi tamamlandı. Sonuçlar 'cluster_results/cluster_results.csv' dosyasına kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Kümeleme analizi sırasında hata: {e}")

    def export_fine_tuning_dataset(self):
        if not cluster_results_list:
            ctk.messagebox.showwarning("Uyarı", "Fine-tuning için veri seti oluşturulacak veri bulunamadı.")
            return
        try:
            df = pd.DataFrame(cluster_results_list)
            output_file = "fine_tuning_dataset.csv"
            df.to_csv(output_file, index=False, encoding='utf-8')
            self.sonuc_text.delete("1.0", "end")
            self.sonuc_text.insert("1.0", f"Fine-tuning veri seti '{output_file}' olarak kaydedildi.\n")
        except Exception as e:
            ctk.messagebox.showerror("Hata", f"Veri seti dışa aktarılırken hata: {e}")

    def citation_chain_view(self):
        # Gerçek atıf zinciri analizi: Kullanıcıdan bir kayıt indeksi alıp, o kaydın bibliyografik bilgisini detaylandırıyoruz.
        idx_str = self.citation_entry.get()
        if not idx_str.isdigit():
            ctk.messagebox.showwarning("Uyarı", "Lütfen geçerli bir indeks girin.")
            return
        idx = int(idx_str)
        if idx < 0 or idx >= len(cluster_results_list):
            ctk.messagebox.showwarning("Uyarı", "Girilen indeks mevcut değil.")
            return
        record = cluster_results_list[idx]
        bib_data = record.get("bibliography", "No bibliographic data available.")
        try:
            # Eğer bibliyografik veri JSON formatındaysa, ayrıntılı görünüm için parse edelim
            bib_json = json.loads(bib_data)
            bib_str = json.dumps(bib_json, indent=2, ensure_ascii=False)
        except Exception:
            bib_str = str(bib_data)
        # Burada basitçe atıf zinciri raporu üretiyoruz. Gerçek uygulamada, atıf zinciri (ileri/geri) analiz algoritmaları kullanılabilir.
        chain_report = f"Atıf Zinciri Raporu için seçilen kayıt (indeks {idx}):\n{bib_str}\n"
        self.sonuc_text.delete("1.0", "end")
        self.sonuc_text.insert("1.0", chain_report)

    def data_search(self):
        # Gelişmiş veri tarama: Kullanıcı sorgusuna göre cluster_results_list üzerinde arama yapıp, sonuçları tablo halinde gösterelim.
        query = self.search_entry.get()
        if not query:
            ctk.messagebox.showwarning("Uyarı", "Lütfen bir sorgu girin.")
            return
        try:
            texts = [result["content"] for result in cluster_results_list]
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(texts)
            query_vec = vectorizer.transform([query])
            sims = cosine_similarity(X, query_vec).flatten()
            # Top 10 sonucu alalım
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


# ----------------------------
# Ana İşlem Fonksiyonu
# ----------------------------
def process_file(item):
    try:
        if not isinstance(item, dict):
            raise ValueError("Geçersiz öğe formatı")
        title = item.get("title")
        if not title:
            raise ValueError("Başlık bulunamadı")
        dosya_path = os.path.join(STORAGE_DIR, title)
        if not os.path.exists(dosya_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {dosya_path}")
        if title.lower().endswith('.pdf'):
            raw_text = extract_text_from_pdf(dosya_path)
            source_type = "pdf"
        elif title.lower().endswith('.txt'):
            with open(dosya_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            source_type = "txt"
        else:
            raise ValueError("Desteklenmeyen dosya uzantısı.")
        if not raw_text:
            raise Exception("Ham metin çıkarılamadı")
        reflowed_text = reflow_columns(raw_text)
        tables = detect_tables(dosya_path) if source_type == "pdf" else []
        references = extract_references_enhanced(raw_text)
        if references:
            ref_filename = os.path.splitext(title)[0] + "_references.txt"
            save_text_file(TEMIZ_KAYNAKCA_DIR, ref_filename, "\n".join(references))
            vosviewer_file = os.path.join(TEMIZ_KAYNAKCA_DIR, os.path.splitext(title)[0] + "_references_vosviewer.txt")
            pajek_file = os.path.join(TEMIZ_KAYNAKCA_DIR, os.path.splitext(title)[0] + "_references_pajek.net")
            save_references_for_analysis(references, vosviewer_file, pajek_file)
        else:
            references = ["No references found."]
        temiz_metin = clean_text(reflowed_text)
        if not temiz_metin:
            temiz_metin = "No clean text extracted."
        temiz_metin_filename = os.path.splitext(title)[0] + ".temizmetin.txt"
        save_text_file(TEMIZMETIN_DIR, temiz_metin_filename, temiz_metin)
        chunks = split_text(temiz_metin, chunk_size=256)
        chunk_ids = [f"{os.path.splitext(title)[0]}_{i}" for i in range(len(chunks))]
        embeddings = []
        for idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            if emb is None:
                logger.warning(f"Chunk {idx} için embedding oluşturulamadı.")
                embeddings.append([0.0] * 768)
            else:
                embeddings.append(emb)
            embed_filename = f"{os.path.splitext(title)[0]}_{idx}.embed.txt"
            save_text_file(EMBEDDING_DIR, embed_filename, chunk)
        try:
            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=[{'title': title, 'source': source_type, 'chunk_index': i,
                            'timestamp': datetime.now().isoformat()} for i in range(len(chunks))]
            )
            logger.info(f"✅ {title} için tüm chunk embedding'leri ChromaDB'ye eklendi.")
        except Exception as e:
            logger.error(f"Embedding eklenirken hata oluştu: {e}")
        item_key = item.get("key")
        if item_key:
            bib_data = fetch_zotero_metadata(item_key)
            if bib_data:
                try:
                    bib_collection.add(
                        ids=[os.path.splitext(title)[0]],
                        embeddings=[[0.0] * 768],
                        metadatas=[{'title': title, 'bibliography': bib_data,
                                    'timestamp': datetime.now().isoformat()}]
                    )
                    logger.info(f"✅ {title} için Zotero bibliyografi bilgisi eklendi.")
                    bib_str = json.dumps(bib_data)
                except Exception as e:
                    logger.error(f"Bibliyografi eklenirken hata: {e}")
                    bib_str = "No bibliographic data available."
            else:
                bib_str = "No bibliographic data available."
        else:
            bib_str = "No bibliographic data available."
        stack_guncelle(title, "işlendi")
        log_entry = {
            'dosya': title,
            'tarih': datetime.now(pytz.timezone('Turkey')).isoformat(),
            'tablo_sayisi': len(tables),
            'referans_sayisi': len(references),
            'dosya_tipi': source_type,
            'bellek_kullanim': memory_usage(),
            'clean_text': temiz_metin,
            'bibliography': bib_str
        }
        return (True, log_entry)
    except Exception as e:
        error_log = {
            'dosya': title if 'title' in locals() else 'unknown',
            'hata': str(e),
            'traceback': traceback.format_exc(),
            'zaman': datetime.now().isoformat()
        }
        return (False, error_log)


def main():
    global total_files, success_count, embedding_failed_count, text_extraction_failed_count
    global collection, bib_collection, cluster_results_list
    try:
        init_dirs()
        print("\n" + "="*60)
        print("### PDF/TXT İŞLEME, TEMİZ METİN, EMBEDDING, ZOTERO VE KÜMELEME ANALİZİ SİSTEMİ ###")
        print("="*60)
        json_file_name = input("İşlenecek JSON dosyasının adını girin (örneğin: Kitap.json): ")
        json_file_path = os.path.join(SUCCESS_DIR, json_file_name)
        if not os.path.exists(json_file_path):
            logger.error(f"❌ {json_file_name} dosyası bulunamadı!")
            return
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_items = [item for item in data if isinstance(item, dict) and item.get('title')]
        total_files = len(valid_items)
        if total_files == 0:
            logger.error("❌ İşlenecek geçerli kayıt bulunamadı!")
            return
        user_input = input("Baştan başlamak için [B], kaldığınız yerden devam için [C], güncelleme için [G]: ").lower()
        if user_input == 'b':
            logger.warning("⚠️ Veritabanı sıfırlanıyor...")
            try:
                collection.delete(where={})
                bib_collection.delete(where={})
                collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
                bib_collection = chroma_client.get_or_create_collection(name="pdf_bibliography")
            except Exception as e:
                logger.error(f"❌ Veritabanı sıfırlama hatası: {e}")
                return
            last_index = 0
        elif user_input == 'c':
            last_index = get_last_processed_index()
        else:
            last_index = 0
        print(f"\nİşlem başlıyor... ({last_index + 1}/{total_files})")
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {executor.submit(process_file, item): item for item in valid_items[last_index:]}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Dosyalar İşleniyor"):
                item = futures[future]
                try:
                    success, result = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"✅ {item.get('title', 'Bilinmeyen dosya')} işlendi")
                        cluster_results_list.append({
                            "content": result.get("clean_text", "No clean text extracted."),
                            "bibliography": result.get("bibliography", "No bibliographic data available.")
                        })
                    else:
                        logger.error(f"❌ {item.get('title', 'Bilinmeyen dosya')} hatası: {result['hata']}")
                except Exception as e:
                    logger.error(f"❌ İşlem hatası: {item.get('title', 'Bilinmeyen dosya')} - {str(e)}")
                save_last_processed_index(valid_items.index(item))
    except Exception as e:
        logger.error(f"Ana programda hata oluştu: {str(e)}")
        error_log = {
            'dosya': 'main',
            'hata': str(e),
            'traceback': traceback.format_exc(),
            'zaman': datetime.now().isoformat()
        }
        logger.error(error_log)
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print(f"İşlem tamamlandı!")
        print(f"Toplam dosya: {total_files}")
        print(f"Başarılı: {success_count}")
        print(f"Embedding hatası: {embedding_failed_count}")
        print(f"Metin çıkarma hatası: {text_extraction_failed_count}")
        print("="*60)
        if not cluster_results_list:
            try:
                df = pd.read_csv("fine_tuning_dataset.csv")
                cluster_results_list = df.to_dict(orient="records")
            except Exception as e:
                logger.warning("Önceki fine-tuning veri seti bulunamadı; kümeleme analizi atlanacak.")
        if cluster_results_list:
            cluster_analysis_from_chromadb(cluster_results_list, n_clusters=5, output_dir="cluster_results")
        # Ek özellikler GUI'sini başlat
        app = AdditionalFeaturesGUI()
        app.mainloop()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
