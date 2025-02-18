# ZOTERO PDF VE TXT DOSYA ISLEME VE METIN, KAYNAKCA, TABLO CIKARTMA PROGRAMI
# ZUHTU METE DINLER - 2025

# Program, PDF dosyalarını işler, metin çıkarır, sütun yapısını düzeltir, tablo ve kaynakça çıkarır, temiz metin oluşturur, metni küçük parçalara böler, her parçayı embedding'e dönüştürür ve ChromaDB'ye ekler.
# Ayrıca, Zotero'dan bibliyografik bilgileri çeker ve ChromaDB'ye ekler.    
# Programın çalışabilmesi için, OpenAI API anahtarını ve Zotero API anahtarını ayarlamalısınız.




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
BASE_DIR = r"C:\Users\mete\Zotero\zotai"  # Ana dizin
STORAGE_DIR = os.getenv('STORAGE_DIR')  # PDF dosyalarının bulunduğu dizin
SUCCESS_DIR = os.getenv('SUCCESS_DIR')  # Çıktıların saklanacağı dizin
TEMIZMETIN_DIR = os.path.join(BASE_DIR, "TemizMetin")  # Temiz metinlerin kaydedileceği dizin
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
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levellevel)s - %(message)s"))
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
# Sayaçlar ve Takip Değişkenleri
# ----------------------------
total_files = 0
success_count = 0
embedding_failed_count = 0
text_extraction_failed_count = 0

# ----------------------------
# Yardımcı Fonksiyonlar
# ----------------------------
def init_dirs():
    """Gerekli tüm dizinleri oluşturur."""
    dirs = [TEMIZMETIN_DIR, TEMIZ_TABLO_DIR, TEMIZ_KAYNAKCA_DIR, EMBEDDING_DIR, PDF_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    for fmt in ['json', 'csv', 'excel']:
        os.makedirs(os.path.join(TEMIZ_TABLO_DIR, fmt), exist_ok=True)

def memory_usage():
    """Anlık bellek kullanımını döndürür."""
    return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"

def clean_text(text):
    """
    Metni temizler:
      - \r\n yerine \n
      - Fazla boşluk ve satır aralıklarını azaltır
      - Paragrafların yapısını korur (wrap olmadan)
    """
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def split_text(text, chunk_size=256):
    """
    Metni önce paragraflara bölüp, ardından her paragrafı belirli kelime sayısına göre küçük parçalara ayırır.
    Böylece wrap olmadan orijinal paragraf yapısı korunur.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
    return chunks

def save_text_file(directory, filename, content):
    """Belirtilen dizine, verilen dosya adıyla içerik kaydeder."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as f:
        f.write(content)

# ----------------------------
# PDF İşleme Fonksiyonları
# ----------------------------
def extract_text_from_pdf(pdf_path):
    """
    PDF'den ham metni çıkarır.
    Gelişmiş sütun tespiti ile akış düzenlemesi yapılabilir.
    """
    try:
        doc = fitz.open(pdf_path)
        raw_text = ""
        for page in doc:
            # Sayfa metnini sıralı şekilde alıyoruz.
            page_text = page.get_text("text", sort=True)
            raw_text += page_text + "\n\n"
        doc.close()
        return raw_text
    except Exception as e:
        logger.error(f"❌ PDF metni çıkarılamadı: {pdf_path}, Hata: {e}")
        traceback.print_exc()
        return None

def detect_tables(pdf_path):
    """
    PDF'den tablo, şekil, figür çıkarımı yapar.
    Örnek amaçlı basit regex tabanlı tespit yöntemi kullanılmıştır.
    """
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

def extract_references_enhanced(text):
    """
    Gelişmiş kaynakça çıkarımı:
    PDF veya TXT dosyalarında bulunan kaynakça bölümünü, çeşitli pattern'lerle tespit eder.
    Deneyimle optimize edilmiş hali korunmuştur.
    Gelişmiş referans çıkarma fonksiyonu.
    Hem PDF hem de TXT dosyalarından referansları çıkarır.
    """
    references = []
    ref_patterns = [
        r'(?i)(?:KAYNAKÇA|KAYNAKLAR|REFERENCES|BIBLIOGRAPHY).*?\n(.*?)(?=\n\s*\n|\Z)',
        r'(?i)(?:REFERANSLAR|KAYNAK LİSTESİ).*?\n(.*?)(?=\n\s*\n|\Z)',
        r'\n[\[\(](\d{4})[\]\)].*?(?=\n|$)',  # Yıl tabanlı referanslar
        r'\n(?:[A-Za-z\u00C0-\u017F]+,\s+[A-Za-z\u00C0-\u017F\.]+\s+\(\d{4}\).*?)(?=\n|$)',  # Yazar, Yıl formatı
    ]
    
    # Önce kaynakça bölümünü bul
    text_lower = text.lower()
    start_indices = []
    ref_headers = ['kaynakça', 'kaynaklar', 'references', 'bibliography', 'referanslar', 'kaynak listesi']
    
    for header in ref_headers:
        idx = text_lower.find(header)
        if idx != -1:
            start_indices.append(idx)
    
    if start_indices:
        # En erken başlayan kaynakça bölümünü al
        start_idx = min(start_indices)
        ref_section = text[start_idx:]
        
        # Referansları satır satır ayır
        lines = ref_section.split('\n')
        current_ref = ""
        
        for line in lines[1:]:  # İlk satırı (başlığı) atla
            line = line.strip()
            if not line:  # Boş satır
                if current_ref:
                    references.append(current_ref)
                    current_ref = ""
            elif any(c.isdigit() for c in line):  # Yeni referans başlangıcı
                if current_ref:
                    references.append(current_ref)
                current_ref = line
            else:
                if current_ref:
                    current_ref += " " + line
                else:
                    current_ref = line
        
        if current_ref:  # Son referansı ekle
            references.append(current_ref)
    
    # Referansları temizle ve filtrele
    cleaned_refs = []
    for ref in references:
        ref = re.sub(r'\s+', ' ', ref).strip()
        if len(ref) > 10 and any(c.isdigit() for c in ref):  # Geçerli referans kontrolü
            cleaned_refs.append(ref)
    
    return cleaned_refs

def reflow_columns(text):
    """
    Bilimsel yayınlarda özet genellikle tek sütun, asıl metin sütunlu yapıda olabilir.
    Bu fonksiyon, metindeki sütun yapısını tespit edip akışı düzeltir.
    Gelişmiş tespit algoritması eklenebilir; burada basit bir örnek verilmiştir.
    """
    if "Abstract" in text:
        parts = text.split("Abstract", 1)
        abstract = parts[0].strip()
        body = parts[1].strip()
        body = re.sub(r'\n', ' ', body)
        body = re.sub(r'\s{2,}', ' ', body)
        return abstract + "\n\n" + body
    return text

# ----------------------------
# Embedding ve Zotero Fonksiyonları
# ----------------------------
def embed_text(text):
    """
    OpenAI API kullanarak metin için embedding oluşturur.
    Hata durumunda sayaç artar ve None döner.
    """
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
    """
    Belirtilen Zotero item_key için bibliyografik bilgileri getirir.
    """
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

def save_last_processed_chunk_index(pdf_identifier, chunk_index):
    """Son işlenen chunk indeksini kaydeder"""
    with open(f"last_processed_chunk_{pdf_identifier}.txt", "w", encoding="utf-8") as f:
        f.write(str(chunk_index))

def get_last_processed_chunk_index(pdf_identifier):
    """Son işlenen chunk indeksini okur"""
    chunk_file = f"last_processed_chunk_{pdf_identifier}.txt"
    if os.path.exists(chunk_file):
        with open(chunk_file, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0
# ----------------------------
# Ana İşlem Fonksiyonu
# ----------------------------
def process_file(item):
    """
    Bir PDF dosyasını işleyip:
      - Ham metni PDF'den çıkarır.
      - Sütun yapısını düzeltir.
      - Tablo, şekil, figür gibi ekstraları PDF'den çıkarır.
      - Kaynakçayı tespit edip ayrı kaydeder.
      - Temiz metni (ekstralar çıkarılmış) wrap olmadan kaydeder.
      - Temiz metni küçük parçalara bölüp her parçayı OpenAI ile embedding'e dönüştürür,
        ChromaDB'ye ekler ve her chunk embedding metnini ayrı dosya olarak saklar.
      - Zotero'dan bibliyografik bilgileri çekip ChromaDB'ye ekler.
    """
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
        # PDF'den doğrudan tablo, şekil, resim, figür çıkarımı
        tables = detect_tables(pdf_path)
        # Kaynakça çıkarımı
        references = extract_references_enhanced(raw_text)
        if references:
            ref_filename = os.path.splitext(title)[0] + "_references.txt"
            save_text_file(TEMIZ_KAYNAKCA_DIR, ref_filename, "\n".join(references))
        # Temiz metin: ekstralar çıkarılmış, wrap olmadan kaydediliyor
        temiz_metin = clean_text(reflowed_text)
        temiz_metin_filename = os.path.splitext(title)[0] + ".temizmetin.txt"
        save_text_file(TEMIZMETIN_DIR, temiz_metin_filename, temiz_metin)
        # Metni küçük parçalara böl
        chunks = split_text(temiz_metin, chunk_size=256)
        chunk_ids = [f"{os.path.splitext(title)[0]}_{i}" for i in range(len(chunks))]
        embeddings = []
        for idx, chunk in enumerate(chunks):
            emb = embed_text(chunk)
            if emb is None:
                logger.warning(f"Chunk {idx} için embedding oluşturulamadı.")
                embeddings.append([0.0] * 768)  # Modelin embedding boyutuna göre ayarlayın
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
        # Zotero bibliyografik bilgileri çekiliyor ve ChromaDB'ye ekleniyor.
        item_key = item.get("key")
        if item_key:
            bib_data = fetch_zotero_metadata(item_key)
            if bib_data:
                try:
                    bib_collection.add(
                        ids=[os.path.splitext(title)[0]],
                        embeddings=[[0.0] * 768],  # Bibliyografi için dummy embedding
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
        print("### Metin İşleme ve Tablo Çıkarım Sistemi ###")
        print("\n" + "="*60)
        print("### ZOTERO PDF VE TXT İŞLEME SİSTEMİ ###")
        print("="*60)
        print("1. PDF ve TXT dosyalarından referans çıkarma")
        print("2. Tablo çıkarma ve dönüştürme")
        print("3. Metin temizleme ve işleme")
        print("4. Embedding oluşturma")
        print("="*60 + "\n")
        print(f"Kaynak        : {KAYNAK_DIZIN}")
        print(f"Temiz Metin   : {HEDEF_DIZIN}")
        print(f"Temiz Tablo   : {TEMIZ_TABLO_DIZIN}")
        print(f"Temiz Kaynakça: {TEMIZ_KAYNAKCA_DIZIN}\n")
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
    