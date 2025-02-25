import os
import json
import csv
from datetime import datetime
from pathlib import Path
from config_module import config
from zotero_module import dokuman_id_al
from helper_module import shorten_title
from embedding_module import split_text  # Metni parçalara ayırmak için

# Genel dosya kaydetme fonksiyonları

def save_text_file(directory, filename, content):
    """
    Belirtilen dizine, dosya adıyla metni TXT formatında kaydeder.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath

def save_json_file(directory, filename, content):
    """
    Belirtilen dizine, dosya adıyla içerik JSON formatında kaydeder.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    return filepath

# Temiz metin dosyaları kaydetme

def save_clean_text_files(original_filename, clean_text, bib_info):
    """
    Temiz metin içeriğini bibliyometrik bilgi ile birlikte kaydeder.
    TXT dosyası: TEMIZMETIN_DIR/txt/ID.temizmetin.txt
    JSON dosyası: TEMIZMETIN_DIR/json/ID.temizmetin.json
    Bibliyometrik bilgiler, TXT dosyasında "# " yorum satırları olarak, JSON dosyasında "bibinfo" alanı altında eklenir.
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    
    # Bibliyometrik bilgi bloğu oluştur (TXT için)
    bib_lines = []
    if bib_info:
        bib_lines.append("# Bibliyometrik Bilgiler:")
        for key, value in bib_info.items():
            if isinstance(value, list):
                value = "; ".join(value)
            bib_lines.append(f"# {key}: {value}")
        bib_lines.append("")  # Boş satır ekle
    bib_block = "\n".join(bib_lines)
    
    # TXT olarak kaydet
    txt_dir = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin")) / "txt"
    txt_filename = f"{file_id}.temizmetin.txt"
    txt_content = bib_block + "\n" + clean_text
    txt_path = save_text_file(txt_dir, txt_filename, txt_content)
    
    # JSON olarak kaydet
    json_dir = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin")) / "json"
    json_filename = f"{file_id}.temizmetin.json"
    json_content = {
        "bibinfo": bib_info,
        "content": clean_text,
        "olusturma_tarihi": datetime.now().isoformat()
    }
    json_path = save_json_file(json_dir, json_filename, json_content)
    
    return txt_path, json_path

# Kaynakça dosyaları kaydetme

def save_references_files(original_filename, references, bib_info):
    """
    Kaynakça verilerini, bibliyometrik bilgi ile birlikte aşağıdaki formatlarda kaydeder:
      - TXT: TEMIZ_KAYNAKCA_DIZIN/txt/ID.kaynakca.txt
      - JSON: TEMIZ_KAYNAKCA_DIZIN/json/ID.kaynakca.json
      - VOSviewer: TEMIZ_KAYNAKCA_DIZIN/vos/ID.vos.kaynak.txt
      - Pajek: TEMIZ_KAYNAKCA_DIZIN/paj/ID.pjk.kaynak.paj
      - CSV: TEMIZ_KAYNAKCA_DIZIN/csv/ID.kaynakca.csv
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    
    base_dir = Path(os.getenv("TEMIZ_KAYNAKCA_DIZIN", "temizkaynakca"))
    
    # TXT kaydı
    txt_dir = base_dir / "txt"
    txt_filename = f"{file_id}.kaynakca.txt"
    bib_lines = []
    if bib_info:
        bib_lines.append("# Bibliyometrik Bilgiler:")
        for key, value in bib_info.items():
            if isinstance(value, list):
                value = "; ".join(value)
            bib_lines.append(f"# {key}: {value}")
        bib_lines.append("")
    txt_content = "\n".join(bib_lines) + "\n" + "\n".join(references)
    txt_path = save_text_file(txt_dir, txt_filename, txt_content)
    
    # JSON kaydı
    json_dir = base_dir / "json"
    json_filename = f"{file_id}.kaynakca.json"
    json_content = {
        "bibinfo": bib_info,
        "references": references,
        "olusturma_tarihi": datetime.now().isoformat()
    }
    json_path = save_json_file(json_dir, json_filename, json_content)
    
    # VOSviewer formatı
    vos_dir = base_dir / "vos"
    vos_filename = f"{file_id}.vos.kaynak.txt"
    vos_content = "label\n" + "\n".join(references)
    vos_path = save_text_file(vos_dir, vos_filename, vos_content)
    
    # Pajek formatı
    paj_dir = base_dir / "paj"
    paj_filename = f"{file_id}.pjk.kaynak.paj"
    paj_content = f"*Vertices {len(references)}\n"
    for i, ref in enumerate(references, 1):
        paj_content += f'{i} "{ref}"\n'
    paj_path = save_text_file(paj_dir, paj_filename, paj_content)
    
    # CSV formatı
    csv_dir = base_dir / "csv"
    csv_filename = f"{file_id}.kaynakca.csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / csv_filename
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Reference"])  # Başlık satırı
            for ref in references:
                writer.writerow([ref])
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {e}")
    
    return {
        "txt": txt_path,
        "json": json_path,
        "vos": vos_path,
        "paj": paj_path,
        "csv": str(csv_path)
    }

# Tablo dosyaları kaydetme

def save_table_files(original_filename, table_data_list):
    """
    Bir yayına ait tabloları, belirlenen formatlarda kaydeder:
      - JSON: TEMIZ_TABLO_DIZIN/json/ID_tabloX.json
      - CSV: TEMIZ_TABLO_DIZIN/csv/ID_tabloX.csv
      - Excel: TEMIZ_TABLO_DIZIN/excel/ID_tabloX.xlsx
    Her tablo, numaralandırılarak kaydedilir.
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    
    base_dir = Path(os.getenv("TEMIZ_TABLO_DIZIN", "temiztablo"))
    results = []
    for i, table_data in enumerate(table_data_list, start=1):
        # JSON olarak kaydet
        json_dir = base_dir / "json"
        json_filename = f"{file_id}_tablo{i}.json"
        json_path = save_json_file(json_dir, json_filename, {"table": table_data, "olusturma_tarihi": datetime.now().isoformat()})
        
        # CSV olarak kaydet
        csv_dir = base_dir / "csv"
        csv_filename = f"{file_id}_tablo{i}.csv"
        csv_path = csv_dir / csv_filename
        csv_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in table_data:
                    writer.writerow(row)
        except Exception as e:
            config.logger.error(f"CSV dosyası kaydedilemedi: {e}")
        
        # Excel olarak kaydet
        excel_dir = base_dir / "excel"
        excel_filename = f"{file_id}_tablo{i}.xlsx"
        excel_path = excel_dir / excel_filename
        excel_dir.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd
            if len(table_data) > 1:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
            else:
                df = pd.DataFrame(table_data)
            df.to_excel(excel_path, index=False)
        except Exception as e:
            config.logger.error(f"Excel dosyası kaydedilemedi: {e}")
        
        results.append({
            "json": json_path,
            "csv": str(csv_path),
            "excel": str(excel_path)
        })
    return results

# Embedding dosyası kaydetme

def save_embedding_file(original_filename, embedding_text, chunk_index):
    """
    Her PDF/TXT dosyasının embedding verilerini, belirlenen formatta kaydeder.
    Dosya adı: ID_{chunk_index}.embed.txt
    Kaydetme yeri: EMBEDDING_PARCA_DIR altında.
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    base_dir = Path(os.getenv("EMBEDDING_PARCA_DIZIN", "embedingparca"))
    filename = f"{file_id}_{chunk_index}.embed.txt"
    filepath = save_text_file(base_dir, filename, embedding_text)
    return filepath

# Chunk'lanmış ham metin dosyaları kaydetme

def save_chunked_text_files(original_filename, full_text, chunk_size=256):
    """
    Büyük metinler, chunk_size kadar kelimeye bölünür ve her parça ayrı .txt dosyası olarak kaydedilir.
    Dosya adı: ID_<chunk_index>.txt
    Kaydetme yeri: TEMIZMETIN_DIR/chunks
    """
    file_id = dokuman_id_al(original_filename)
    if not file_id:
        file_id = Path(original_filename).stem
    file_id = shorten_title(file_id, max_length=80)
    chunks = split_text(full_text, chunk_size=chunk_size)
    base_dir = Path(os.getenv("TEMIZMETIN_DIR", "temizmetin")) / "chunks"
    saved_files = []
    for i, chunk in enumerate(chunks):
        filename = f"{file_id}_{i}.txt"
        path = save_text_file(base_dir, filename, chunk)
        saved_files.append(str(path))
    return saved_files
