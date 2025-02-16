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
KAYNAK_DIZIN = r"C:\Users\mete\Zotero\zotai"  # Ana dizin
STORAGE_DIR = os.getenv('STORAGE_DIR')      # Örn: C:\Users\mete\Zotero\storage
SUCCESS_DIR = os.getenv('SUCCESS_DIR')      # Örn: C:\Users\mete\Zotero\zotasistan
HEDEF_DIZIN = os.path.join(KAYNAK_DIZIN, "TemizMetin")
TEMIZ_TABLO_DIZIN = os.path.join(KAYNAK_DIZIN, "TemizTablo")
TEMIZ_KAYNAKCA_DIZIN = os.path.join(KAYNAK_DIZIN, "TemizKaynakca")
PDF_DIR = Path(SUCCESS_DIR) / "pdfler" if SUCCESS_DIR else Path("pdfler")
EMBEDDING_PARCA_DIR = Path(SUCCESS_DIR) / "embedingparca" if SUCCESS_DIR else Path("embedingparca")

# ----------------------------
# Loglama Yapılandırması
# ----------------------------
LOG_DOSYASI = os.path.join(KAYNAK_DIZIN, "islem_logu.json" if KAYNAK_DIZIN else Path("islem_logu.json"))    # islem_logu.json dosyasi
HATA_LOG_DOSYASI = os.path.join(KAYNAK_DIZIN, "hata_logu.json" if KAYNAK_DIZIN else Path("hata_logu.json")) # HATA_LOG_DOSYASI

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
# ortam tanimlari meevcut mu
if not STORAGE_DIR:
    logger.warning("STORAGE_DIR ortam değişkeni tanımlı değil.")
if not SUCCESS_DIR:
    logger.warning("SUCCESS_DIR ortam değişkeni tanımlı değil.")
if not os.getenv('OPENAI_API_KEY'):
    logger.error("OPENAI_API_KEY tanımlı değil, OpenAI işlemleri başarısız olacaktır.")
if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
    logger.error("Zotero API anahtarı veya kullanıcı kimliği tanımlı değil.")


# OpenAI ve Zotero API anahtarlarını yükle
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
ZOTERO_API_URL = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items"
# ZOTERO_API_URL = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items/top?limit=1000"

# zoterodan veri alma hatasi var mi
if response.status_code != 200:
    logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
    return {}

# ChromaDB Yapılandırması

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
basarili_embedding_sayisi = 0
basarili_dosya_sayisi = 0
basarisiz_embedding_sayisi = 0
basarisiz_metin_cikartma_sayisi = 0
total_files = 0
success_count = 0
embedding_failed_count = 0
text_extraction_failed_count = 0

# ----------------------------
# Yardımcı Fonksiyonlar
# ----------------------------
def init_dirs():
    """Tüm gerekli dizinleri oluşturur"""
    for d in [HEDEF_DIZIN, TEMIZ_TABLO_DIZIN, TEMIZ_KAYNAKCA_DIZIN, PDF_DIR, EMBEDDING_PARCA_DIR]:
        os.makedirs(d, exist_ok=True)
    for fmt in ['json', 'csv', 'excel']:
        os.makedirs(os.path.join(TEMIZ_TABLO_DIZIN, fmt), exist_ok=True)

def memory_usage():
    """Anlık bellek kullanımını gösterir"""
    return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"

def clean_text(text):
    """Metni temel temizleme işlemlerinden geçirir"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def shorten_title(title, max_length=80):
    """Başlığı belirtilen uzunluğa göre kısaltır"""
    if len(title) > max_length:
        return title[:max_length]
    return title

# ----------------------------
# PDF ve Metin İşleme Fonksiyonları
# ----------------------------
def extract_text_from_pdf(pdf_path):
    """PDF'den metin çıkarma"""
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text("text", sort=True) for page in doc)
        return text
    except Exception as e:
        logger.error(f"❌ İşlem hatası: {dosya_adi} - {str(e)}")
        traceback.print_exc()
        logger.error(f"❌ PDF metni çıkarılamadı: {pdf_path}, Hata: {e}")
        return None

def split_text(text, chunk_size=256):
    """Metni belirli kelime uzunluklarına böler"""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def detect_tables(text):
    """Gelişmiş tablo tespit algoritması"""
    table_patterns = [
        (r'(?i)(?:^|\n)(Tablo \d+)(?:\n)(.*?)(?=\nTablo \d+|\n\s*$)', 'tablo'),
        (r'(?i)(?:^|\n)(Table \d+)(?:\n)(.*?)(?=\nTable \d+|\n\s*$)', 'table'),
        (r'(?i)(?:^|\n)(Çizelge \d+)(?:\n)(.*?)(?=\nÇizelge \d+|\n\s*$)', 'çizelge'),
        (r'(?i)(?:^|\n)(Figure \d+)(?:\n)(.*?)(?=\nFigure \d+|\n\s*$)', 'figure'),
    ]
    
    tables = []
    for pattern, tip in table_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            baslik = match.group(1)
            icerik = match.group(2)
            
            if re.search(r'\d+\.\d+', baslik):
                continue
                
            rows = []
            for line in icerik.split('\n'):
                line = line.strip()
                if line:
                    cells = re.split(r'\t|\s{3,}', line)
                    rows.append([cell.strip() for cell in cells if cell.strip()])
            
            if len(rows) > 1:
                tables.append({
                    'tip': tip,
                    'baslik': baslik,
                    'veriler': rows
                })
    
    return tables
#          Referans çıkarma için yeni fonksiyonlar ekleniyor
def extract_references_enhanced(text):
# #    Kaynakça çıkarımı için gelişmiş algoritma
#     Gelişmiş referans çıkarma fonksiyonu.
#     Hem PDF hem de TXT dosyalarından referansları çıkarır.
    
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

def save_references(references, file_path, source_type="pdf"):
    """
    Çıkarılan referansları kaydeder
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ref_file = os.path.join(TEMIZ_KAYNAKCA_DIZIN, f"{base_name}_references.txt")
    
    # JSON formatında kaydet
    ref_json = os.path.join(TEMIZ_KAYNAKCA_DIZIN, f"{base_name}_references.json")
    ref_data = {
        "source_file": file_path,
        "source_type": source_type,
        "extraction_date": datetime.now().isoformat(),
        "references": references
    }
    
    try:
        # TXT formatında kaydet
        with open(ref_file, 'w', encoding='utf-8') as f:
            f.write(f"Referanslar ({len(references)}):\n")
            f.write("-" * 50 + "\n")
            for i, ref in enumerate(references, 1):
                f.write(f"{i}. {ref}\n")
        
        # JSON formatında kaydet
        with open(ref_json, 'w', encoding='utf-8') as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Referanslar kaydedildi: {ref_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Referanslar kaydedilirken hata: {e}")
        return False
        
# def extract_references(text):  #eskisini sakla dedigim iin saklanmis yukarida daha iyisi yazuldi
#     """Kaynakça çıkarımı için gelişmiş algoritma"""
#     ref_patterns = [
#         r'(?i)(?:Kaynakça|References|Bibliography)[\s\S]*?(?=\n\s*\n)',
#         r'(?i)(?:Referanslar|Kaynak Listesi)[\s\S]*?(?=\n\s*\n)'
#     ]
#     
#     for pattern in ref_patterns:
#         match = re.search(pattern, text)
#         if match:
#             return match.group(0)
#     return None

# ----------------------------
# Embedding ve API İşlemleri
# ----------------------------
def get_embedding(text):
    # extract embedding hatasi var mi
    if embedding is None:
        logger.warning("Metin vektörleştirilemedi, sonraki işlemler atlanacak.")
        return None
    return embedding    

def embed_text(text):    
    global embedding_failed_count            
    """OpenAI API ile metni vektörleştirir"""    
    embedding_failed_count += 1                
    logger.warning("Metin vektörleştirilemedi, sonraki işlemler atlanacak.")                                    
        embedding = get_embedding(text)
if embedding is None:
    logger.warning("Metin vektörleştirilemedi, sonraki işlemler atlanacak.")
    
    """OpenAI API ile metni vektörleştirir"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Embedding oluşturulamadı: {e}")
        return None

def fetch_zotero_metadata(item_key):
    """Belirtilen Zotero kaydının bibliyografik bilgilerini alır"""
    headers = {"Zotero-API-Key": ZOTERO_API_KEY}
    response = requests.get(f"{ZOTERO_API_URL}/{item_key}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
        return None

# ----------------------------
# Durum Takip Fonksiyonları
# ----------------------------
def save_last_processed_index(index):
    """Son işlenen dosya indeksini kaydeder"""
    with open("last_processed_index.txt", "w", encoding="utf-8") as f:
        f.write(str(index))

def get_last_processed_index():
    """Son işlenen dosya indeksini okur"""
    if os.path.exists("last_processed_index.txt"):
        with open("last_processed_index.txt", "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0

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

def print_status(current_file, total, success, embed_fail, text_fail):
    """Konsolda ilerleme durumunu gösterir"""
    total_embed = success + embed_fail
    status_msg = (f"ߓ İşlenen dosya: {current_file}/{total} | "
                 f"✅ Başarı: {success}/{total_embed} | "
                 f"\033[31m❌ Embed Hata: {embed_fail}\033[0m | "
                 f"\033[33m⚠️ Metin Hata: {text_fail}\033[0m")
                 
    sys.stdout.write("\033[s")
    sys.stdout.write("\033[1;1H" + status_msg.ljust(100))
    sys.stdout.write("\033[u")
    sys.stdout.flush()

# ----------------------------
# Ana İşlem Fonksiyonları
# ----------------------------

# Process_file fonksiyonunu güncelle
# ===========================================================================
def process_file(item):
    """Bir dosyayı işleyen ana fonksiyon"""
    try:
        # Gerekli alan kontrolü
        if not isinstance(item, dict):
            raise ValueError("Geçersiz öğe formatı")
            
        title = item.get("title")
        if not title:
            raise ValueError("Başlık bulunamadı")
            
        # Dosya yolunu oluştur
        kaynak_path = os.path.join(STORAGE_DIR, title)
        if not os.path.exists(kaynak_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {kaynak_path}")
            
        hedef_path = os.path.join(HEDEF_DIZIN, f"{os.path.splitext(title)[0]}.temizmetin.txt")
        
        # Dosya türünü belirle
        is_pdf = title.lower().endswith('.pdf')
        
        # Metin çıkarma
        if is_pdf:
            text = extract_text_from_pdf(kaynak_path)
            source_type = "pdf"
        else:
            with open(kaynak_path, 'r', encoding='utf-8') as f:
                text = f.read()
            source_type = "txt"
        
        if not text:
            raise Exception("Metin çıkarılamadı")
        
        # Temel işlemler
        temiz_metin = clean_text(text)
        tablolar = detect_tables(text)
        
        # Referans çıkarma
        references = extract_references_enhanced(text)
        if references:
            save_references(references, kaynak_path, source_type)
        
        # Çıktıları kaydet
        with open(hedef_path, 'w', encoding='utf-8') as f:
            f.write(temiz_metin)
        
        # Tabloları kaydet
        if tablolar:
            base_name = os.path.splitext(title)[0]
            
            # JSON formatında kaydet
            with open(os.path.join(TEMIZ_TABLO_DIZIN, 'json', f"{base_name}.json"), 'w', encoding='utf-8') as f:
                json.dump(tablolar, f, indent=2, ensure_ascii=False)
            
            # CSV ve Excel formatlarında kaydet
            for i, table in enumerate(tablolar, 1):
                if table['veriler'] and len(table['veriler']) > 0:
                    try:
                        df = pd.DataFrame(table['veriler'][1:], columns=table['veriler'][0])
                        
                        csv_path = os.path.join(TEMIZ_TABLO_DIZIN, 'csv', f"{base_name}_tablo{i}.csv")
                        df.to_csv(csv_path, index=False, encoding='utf-8')
                        
                        excel_path = os.path.join(TEMIZ_TABLO_DIZIN, 'excel', f"{base_name}_tablo{i}.xlsx")
                        df.to_excel(excel_path, index=False)
                    except Exception as e:
                        logger.warning(f"Tablo dönüştürme hatası: {e}")
                        continue
        
        log_entry = {
            'dosya': title,
            'tarih': datetime.now(pytz.timezone('Turkey')).isoformat(),
            'tablo_sayisi': len(tablolar),
            'referans_sayisi': len(references),
            'dosya_tipi': source_type,
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
    """Ana program akışı"""
    global total_files, success_count, embedding_failed_count, text_extraction_failed_count
    
    try:
        init_dirs()
        
        print("\n" + "="*60)
        print("### ZOTERO PDF VE TXT İŞLEME SİSTEMİ ###")
        print("="*60)
        print("1. PDF ve TXT dosyalarından referans çıkarma")
        print("2. Tablo çıkarma ve dönüştürme")
        print("3. Metin temizleme ve işleme")
        print("4. Embedding oluşturma")
        print("="*60 + "\n")
        
        json_file_name = input("İşlenecek JSON dosyasının adını girin (örneğin: Kitap.json): ")
        print(f"\nÇalışma dizini: {os.path.join(os.getcwd(), SUCCESS_DIR)}\n")
        
        json_file_path = os.path.join(SUCCESS_DIR, json_file_name)
        if not os.path.exists(json_file_path):
            logger.error(f"❌ {json_file_name} dosyası bulunamadı!")
            return
            
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Geçerli kayıtları filtrele
        valid_items = [item for item in data if isinstance(item, dict) and item.get('title')]
        total_files = len(valid_items)
        
        if total_files == 0:
            logger.error("❌ İşlenecek geçerli kayıt bulunamadı!")
            return
            
        user_input = input("Baştan başlamak için [B], kaldığınız yerden devam için [C], "
                          "güncelleme için [G]: ").lower()
        
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
            futures = {executor.submit(process_file, item): item 
                      for item in valid_items[last_index:]}
            
            for future in tqdm(as_completed(futures), total=len(futures),
                             desc="Dosyalar İşleniyor"):
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
# ===========================================================================

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()