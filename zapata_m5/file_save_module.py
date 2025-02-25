import os
import json
import csv
import pandas as pd
from pathlib import Path
from config_module import config

def save_text_file(directory, filename, content):
    """
    Genel metin dosyalarını TXT formatında kaydeder.
    
    Args:
        directory (str or Path): Dosyanın kaydedileceği dizin.
        filename (str): Dosya adı (uzantı eklenmeyecek, fonksiyon ekleyecek).
        content (str): Kaydedilecek metin içeriği.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{filename}.txt"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        config.logger.info(f"TXT dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"TXT dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_json_file(directory, filename, content):
    """
    Veriyi JSON formatında kaydeder.
    
    Args:
        directory (str or Path): Dosyanın kaydedileceği dizin.
        filename (str): Dosya adı (uzantı eklenmeyecek).
        content (dict): Kaydedilecek veri.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{filename}.json"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        config.logger.info(f"JSON dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"JSON dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_clean_text_files(original_filename, clean_text, bib_info):
    """
    Temiz metin dosyasını hem TXT hem de JSON formatında, bibliyometrik bilgilerle birlikte kaydeder.
    
    Dosya isimlendirme: {ID}.clean.txt ve {ID}.clean.meta.json
    ID, dosyanın temel adı (dokuman_id_al + shorten_title ile elde edilir).
    
    Args:
        original_filename (str): İşlenen dosyanın orijinal adı.
        clean_text (str): Temizlenmiş metin.
        bib_info (dict): Bibliyometrik bilgiler.
    
    Returns:
        dict: {"txt": dosya_yolu, "json": dosya_yolu} şeklinde dosya yolları.
    """
    base_name = Path(original_filename).stem
    txt_path = save_text_file(config.CLEAN_TEXT_DIR / "txt", f"{base_name}.clean", clean_text)
    json_path = save_json_file(config.CLEAN_TEXT_DIR / "json", f"{base_name}.clean.meta", bib_info)
    return {"txt": txt_path, "json": json_path}

def save_references_files(original_filename, references, bib_info):
    """
    Kaynakça verilerini farklı formatlarda kaydeder:
      - TXT: {ID}.references.txt
      - JSON: {ID}.references.meta.json
      - VOSviewer: {ID}.vos.references.txt
      - Pajek: {ID}.pjk.references.paj
      - CSV: {ID}.references.csv  (Her satırda bir kaynakça kaydı)
    
    Args:
        original_filename (str): Orijinal dosya adı.
        references (list): Kaynakça metinlerinin listesi.
        bib_info (dict): Bibliyometrik bilgiler.
    
    Returns:
        dict: Kaydedilen dosya yollarını içeren sözlük.
    """
    base_name = Path(original_filename).stem
    ref_txt = save_text_file(config.REFERENCES_DIR / "txt", f"{base_name}.references", "\n".join(references))
    ref_json = save_json_file(config.REFERENCES_DIR / "json", f"{base_name}.references.meta", bib_info)
    # VOSviewer formatı: Basit liste, her satırda bir kaynak
    vos_path = save_text_file(config.REFERENCES_DIR / "vosviewer", f"{base_name}.vos.references", "\n".join(references))
    # Pajek formatı: İlk satırda toplam vertex sayısı, sonraki satırlarda id ve kaynakça metni
    pajek_file = config.REFERENCES_DIR / "pajek" / f"{base_name}.pjk.references.paj"
    (config.REFERENCES_DIR / "pajek").mkdir(parents=True, exist_ok=True)
    try:
        with open(pajek_file, 'w', encoding='utf-8') as f:
            f.write(f"*Vertices {len(references)}\n")
            for idx, ref in enumerate(references, start=1):
                f.write(f'{idx} "{ref}"\n')
        config.logger.info(f"Pajek dosyası kaydedildi: {pajek_file}")
        pajek_path = str(pajek_file)
    except Exception as e:
        config.logger.error(f"Pajek dosyası kaydedilemedi: {pajek_file}, Hata: {e}")
        pajek_path = None

    # CSV formatı: Her kaynakça için bir satır
    csv_file = config.REFERENCES_DIR / "csv" / f"{base_name}.references.csv"
    (config.REFERENCES_DIR / "csv").mkdir(parents=True, exist_ok=True)
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Kaynakça"])
            for ref in references:
                writer.writerow([ref])
        config.logger.info(f"CSV dosyası kaydedildi: {csv_file}")
        csv_path = str(csv_file)
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {csv_file}, Hata: {e}")
        csv_path = None

    return {"txt": ref_txt, "json": ref_json, "vosviewer": vos_path, "pajek": pajek_path, "csv": csv_path}

def save_table_files(original_filename, table_data_list):
    """
    Tabloları JSON, CSV ve Excel formatlarında kaydeder.
    
    Args:
        original_filename (str): Orijinal dosya adı.
        table_data_list (list): Tabloların verilerini içeren liste. Her tablo, 'baslik' ve 'veriler' anahtarlarına sahip sözlük olarak tanımlanmalı.
    
    Returns:
        dict: Kaydedilen dosya yollarını içeren sözlük.
    """
    base_name = Path(original_filename).stem
    table_dir = config.TABLES_DIR
    table_dir.mkdir(parents=True, exist_ok=True)

    # JSON formatında kaydet
    json_path = table_dir / f"{base_name}.tables.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(table_data_list, f, ensure_ascii=False, indent=4)
        config.logger.info(f"Tablolar JSON formatında kaydedildi: {json_path}")
    except Exception as e:
        config.logger.error(f"JSON dosyası kaydedilemedi: {json_path}, Hata: {e}")
    
    # CSV formatında kaydet
    csv_path = table_dir / f"{base_name}.tables.csv"
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Tablo Başlığı", "Tablo İçeriği"])
            for table in table_data_list:
                writer.writerow([table.get("baslik", ""), json.dumps(table.get("veriler", []), ensure_ascii=False)])
        config.logger.info(f"Tablolar CSV formatında kaydedildi: {csv_path}")
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {csv_path}, Hata: {e}")

    # Excel formatında kaydet
    excel_path = table_dir / f"{base_name}.tables.xlsx"
    try:
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        for idx, table in enumerate(table_data_list, start=1):
            df = pd.DataFrame(table.get("veriler", []))
            sheet_name = f"Tablo{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
        config.logger.info(f"Tablolar Excel formatında kaydedildi: {excel_path}")
    except Exception as e:
        config.logger.error(f"Excel dosyası kaydedilemedi: {excel_path}, Hata: {e}")

    return {"json": str(json_path), "csv": str(csv_path), "excel": str(excel_path)}

def save_embedding_file(original_filename, embedding_text, chunk_index):
    """
    Her dosyanın embedding verilerini kaydeder.
    Dosya isimlendirme: {ID}_chunk{chunk_index}.embed.txt
    
    Args:
        original_filename (str): Orijinal dosya adı.
        embedding_text (str): Embedding verisinin kaydedilecek metin hali.
        chunk_index (int): Chunk numarası.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    base_name = Path(original_filename).stem
    embedding_dir = config.EMBEDDINGS_DIR
    embedding_dir.mkdir(parents=True, exist_ok=True)
    file_path = embedding_dir / f"{base_name}_chunk{chunk_index}.embed.txt"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(embedding_text)
        config.logger.info(f"Embedding dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"Embedding dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_chunked_text_files(original_filename, full_text, chunk_size=256):
    """
    Büyük metni, belirlenen chunk boyutuna göre bölerek dosya sistemine kaydeder.
    Dosya isimlendirme: {ID}_part{parça_numarası}.txt
    
    Args:
        original_filename (str): Orijinal dosya adı.
        full_text (str): Tüm metin.
        chunk_size (int): Her chunk için karakter sayısı (varsayılan: 256).
    
    Returns:
        list: Kaydedilen tüm dosya yollarının listesi.
    """
    base_name = Path(original_filename).stem
    chunk_dir = config.CLEAN_TEXT_DIR / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    text_chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    file_paths = []
    for idx, chunk in enumerate(text_chunks, start=1):
        file_path = save_text_file(chunk_dir, f"{base_name}_part{idx}", chunk)
        if file_path:
            file_paths.append(file_path)
    config.logger.info(f"Büyük metin {len(text_chunks)} parçaya bölündü ve kaydedildi.")
    return file_paths

# Aşağıda, tartışmalarımız ve güncellemeler doğrultusunda oluşturulmuş,
# final sürümü olan **`file_save_module.py`** modülünü bulabilirsiniz. 
# Bu modül, PDF veya TXT dosyalarından elde edilen verilerin 
# (temiz metin, kaynakça, tablolar, embedding, chunk’lar vs.) belirlenen
# dizinlerde uygun formatlarda (TXT, JSON, CSV, Excel gibi) saklanmasını sağlar.
# Aşağıdaki kod; dosya isimlendirme kurallarına 
# (ör. “ID.temizmetin.txt”, “ID.citation.json”, “ID.tables.csv” vs.) uymakta, 
# eksiksiz loglama, hata yönetimi ve otomatik dizin oluşturma işlemlerini gerçekleştirmektedir.



# ### Açıklamalar

# - **save_text_file:** Belirtilen dizinde TXT dosyasını oluşturur.
# - **save_json_file:** Belirtilen dizinde JSON dosyasını oluşturur.
# - **save_clean_text_files:** Temiz metin dosyasını hem TXT hem JSON olarak kaydeder; bibliyometrik bilgilerle ilişkilendirir.
# - **save_references_files:** Kaynakça verilerini TXT, JSON, VOSviewer, Pajek ve CSV formatlarında saklar.
# - **save_table_files:** Tabloları JSON, CSV ve Excel formatında kaydeder. CSV için her tablo için "Tablo Başlığı" ve "Veri" bilgisi saklanır.
# - **save_embedding_file:** Her chunk embedding verisini "ID_chunkX.embed.txt" formatında kaydeder.
# - **save_chunked_text_files:** Büyük metni belirlenen boyutlarda parçalar ve her parçayı ayrı TXT dosyası olarak kaydeder.

# Bu final sürümü, tüm tartışmalarımız doğrultusunda ve önceki kodlarla karşılaştırıldığında eksiksiz, gelişmiş hata yönetimi, 
# loglama ve uygun dosya formatları ile kaydetme işlemlerini yerine getirmektedir.

# Herhangi bir ek talebiniz veya geliştirme isteğiniz varsa lütfen bildiriniz.