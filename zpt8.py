# chatgpt yine birseyleri kesti.
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

# ----------------------------
# Dizin Tanımlamaları
# ----------------------------
BASE_DIR = r"C:\Users\mete\Zotero\zotai"
STORAGE_DIR = os.getenv('STORAGE_DIR')  # PDF dosyalarının bulunduğu dizin
SUCCESS_DIR = os.getenv('SUCCESS_DIR')  # Çıktıların saklanacağı ana dizin
TEMIZMETIN_DIR = os.path.join(BASE_DIR, "TemizMetin")  # Temizlenmiş metinler için
TEMIZ_TABLO_DIR = os.path.join(BASE_DIR, "TemizTablo")
TEMIZ_KAYNAKCA_DIR = os.path.join(BASE_DIR, "TemizKaynakca")
EMBEDDING_DIR = os.path.join(SUCCESS_DIR, "embedingparca") if SUCCESS_DIR else os.path.join(os.getcwd(), "embedingparca")
PDF_DIR = os.path.join(SUCCESS_DIR, "pdfler") if SUCCESS_DIR else os.path.join(os.getcwd(), "pdfler")

# ----------------------------
# Loglama Yapılandırması
# ----------------------------
LOG_DOSYASI = os.path.join(BASE_DIR, "islem_logu.json")
HATA_LOG_DOSYASI = os.path.join(BASE_DIR, "hata_logu.json")

log_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
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
# API ve Client Yapılandırmaları
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
# Sayaçlar
# ----------------------------
total_files = 0
success_count = 0
embedding_failed_count = 0
text_extraction_failed_count = 0

# ----------------------------
# Yardımcı Fonksiyonlar
# ----------------------------
def init_dirs():
    dirs = [TEMIZMETIN_DIR, TEMIZ_TABLO_DIR, TEMIZ_KAYNAKCA_DIR, EMBEDDING_DIR, PDF_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    for fmt in ['json', 'csv', 'excel']:
        os.makedirs(os.path.join(TEMIZ_TABLO_DIR, fmt), exist_ok=True)

def memory_usage():
    return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"

def clean_text(text):
    # Temiz metin: gereksiz boşluklar, satır sonları kaldırılarak, paragraflar korunur.
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def split_text(text, chunk_size=256):
    # Metni paragraflara bölüp, ardından her paragrafı belirli boyutta parçalara ayırır.
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
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
            # Sütun tespiti ve akış düzenleme
            # Burada basitçe "raw" metin kullanılıyor, gelişmiş algoritma eklenebilir.
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
            # Doğrudan PDF'den tablo çıkarımı için özel yöntemler uygulanabilir.
            # Burada örnek amaçlı basit regex veya OCR tabanlı yöntem entegre edilebilir.
            text = page.get_text("text", sort=True)
            table_patterns = [
                (r'(?i)(Tablo\s*\d+)', 'tablo'),
                (r'(?i)(Table\s*\d+)', 'table')
            ]
            for pattern, tip in table_patterns:
                for match in re.finditer(pattern, text):
                    tables.append({'tip': tip, 'baslik': match.group(0)})
        doc.close()
        return tables
    except Exception as e:
        logger.error(f"❌ Tablo çıkarma hatası: {e}")
        return []

def extract_references(text):
    ref_patterns = [
        r'(?i)(?:KAYNAKÇA|REFERENCES|BIBLIOGRAPHY).*?\n(.*?)(?=\n\s*\n|\Z)',
        r'\[\d+\]\s.*?(?=\n|$)'
    ]
    references = []
    for pattern in ref_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            ref = match.group(0).strip()
            if ref not in references:
                references.append(ref)
    cleaned_refs = [re.sub(r'\s+', ' ', ref).strip() for ref in references if len(ref) > 10 and any(c.isdigit() for c in ref)]
    return cleaned_refs

def reflow_columns(text):
    # Çok sütunlu yapıları tespit edip, akışı düzeltmek için basit örnek algoritma.
    # Gelişmiş algoritma için OCR ve sütun tespiti kullanılabilir.
    # Burada, eğer metin içerisinde "Abstract" bulunuyorsa, ilk paragrafı ayırıp tek sütunlu hale getiriyoruz.
    if "Abstract" in text:
        parts = text.split("Abstract", 1)
        abstract = parts[0].strip()
        body = parts[1].strip()
        # Basit sütun ayrımı düzeltmesi; gelişmiş algoritma eklenebilir.
        body = re.sub(r'\n', ' ', body)
        body = re.sub(r'\s{2,}', ' ', body)
        return abstract + "\n\n" + body
    return text

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
# Durum Takip Fonksiyonları
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
# Ana İşlem Fonksiyonu
# ----------------------------
def process_file(item):
    try:
        if not isinstance(item, dict):
            raise ValueError("Geçersiz öğe formatı")
        title = item.get("title")
        if not title:
            raise ValueError("Başlık bulunamadı")
        pdf_path = os.path.join(STORAGE_DIR, title)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {pdf_path}")
        # Ham metin PDF'den çıkarılıyor
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise Exception("Ham metin çıkarılamadı")
        # Sütun tespiti ve akış düzeltme
        reflowed_text = reflow_columns(raw_text)
        # Tablolar, şekiller, resimler, figurler doğrudan PDF'den çıkarılıyor
        tables = detect_tables(pdf_path)
        # Kaynakça çıkarımı
        references = extract_references(raw_text)
        if references:
            ref_filename = os.path.splitext(title)[0] + "_references.txt"
            save_text_file(TEMIZ_KAYNAKCA_DIR, ref_filename, "\n".join(references))
        # Temiz metin: tablolar, resimler, referanslar gibi ekstralar metinden çıkarılarak ayrık tutulur.
        # Varsayılan olarak reflowed_text içerisinden bu kısımlar çıkarılabilir; burada basitçe temizleniyor.
        temiz_metin = clean_text(reflowed_text)
        # Temiz metin dosyası, wrap olmadan (paragraflar korunarak) kaydediliyor.
        temiz_metin_filename = os.path.splitext(title)[0] + ".temizmetin.txt"
        save_text_file(TEMIZMETIN_DIR, temiz_metin_filename, temiz_metin)
        # Metin, küçük parçalara bölünüyor.
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
            # Her chunk embedding metni ayrı dosya olarak kaydediliyor.
            embed_filename = f"{os.path.splitext(title)[0]}_{idx}.embed.txt"
            save_text_file(EMBEDDING_DIR, embed_filename, chunk)
        try:
            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=[{'title': title, 'chunk_index': i, 'timestamp': datetime.now().isoformat()} for i in range(len(chunks))]
            )
            logger.info(f"✅ {title} için tüm chunk embedding'leri ChromaDB'ye eklendi.")
        except Exception as e:
            logger.error(f"Embedding eklenirken hata oluştu: {e}")
        # Zotero'dan bibliyografik bilgi çekilerek ChromaDB'ye ekleniyor.
        item_key = item.get("key")
        if item_key:
            bib_data = fetch_zotero_metadata(item_key)
            if bib_data:
                try:
                    bib_collection.add(
                        ids=[os.path.splitext(title)[0]],
                        embeddings=[[0.0] * 768],  # Bibliyografik veri için dummy embedding
                        metadatas=[{'title': title, 'bibliography': bib_data, 'timestamp': datetime.now().isoformat()}]
                    )
                    logger.info(f"✅ {title} için Zotero bibliyografi bilgisi eklendi.")
                except Exception as e:
                    logger.error(f"Bibliyografi eklenirken hata: {e}")
        log_entry = {
            'dosya': title,
            'tarih': datetime.now(pytz.timezone('Turkey')).isoformat(),
            'tablo_sayisi': len(tables),
            'referans_sayisi': len(references),
            'dosya_tipi': "pdf",
            'bellek_kullanim': memory_usage()
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
    global collection, bib_collection
    try:
        init_dirs()
        print("\n" + "="*60)
        print("### PDF İŞLEME, TEMİZ METİN, EMBEDDING VE ZOTERO BİLGİSİ SİSTEMİ ###")
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
                collection.delete()
                bib_collection.delete()
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

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
