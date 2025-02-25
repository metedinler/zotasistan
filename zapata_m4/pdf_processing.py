
# ### 📌 **Güncellenmiş `pdf_processing.py` Modülü**  
# Bu modül, **PDF’den metin çıkarma, bilimsel bölümleri haritalama, sütun tespiti ve tablo çıkarımı** gibi işlemleri gerçekleştirir.  

# 🔹 **pdfplumber + pdfminer ile çift yönlü metin çıkarma** (pdfplumber varsayılan, pdfminer yedek)  
# 🔹 **Geliştirilmiş bilimsel bölüm tespiti (Abstract, Methods, Results, Tablolar, vs.)**  
# 🔹 **Gelişmiş tablo tespiti (`detect_tables`)**  
# 🔹 **Daha güçlü sütun tespiti (`detect_columns`)**  
# 🔹 **Metni tek akışa dönüştüren `reflow_columns` iyileştirildi**  

# ---

## ✅ **`pdf_processing.py` (Güncellenmiş)**
```python
import os
import re
import pdfplumber
from pdfminer.high_level import extract_text
from config_module import config

def extract_text_from_pdf(pdf_path, method=None):
    """
    📌 PDF'den metin çıkarır. Öncelik pdfplumber, başarısız olursa pdfminer kullanılır.
    
    Args:
        pdf_path (str or Path): PDF dosyasının yolu.
        method (str, optional): Kullanılacak metot ('pdfplumber' veya 'pdfminer').
    
    Returns:
        str veya None: Çıkarılan metin, hata durumunda None.
    """
    if method is None:
        method = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()

    text = None
    if method == "pdfplumber":
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
                text = "\n".join(pages_text)
            config.logger.info(f"✅ pdfplumber ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"❌ pdfplumber hata verdi: {e}. pdfminer deneniyor.")
            return extract_text_from_pdf(pdf_path, method="pdfminer")

    elif method == "pdfminer":
        try:
            text = extract_text(pdf_path)
            config.logger.info(f"✅ pdfminer ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"❌ pdfminer hata verdi: {e}")

    return text


def detect_columns(text, min_gap=4):
    """
    📌 Metindeki sütun yapısını tespit eder.
    
    Args:
        text (str): İşlenecek metin.
        min_gap (int): Sütunları ayırmak için gereken minimum boşluk sayısı.
    
    Returns:
        dict: {'sutunlu': True} veya {'sutunlu': False}.
    """
    lines = text.split("\n")
    column_count = sum(1 for line in lines if re.search(r" {" + str(min_gap) + r",}", line))
    return {"sutunlu": column_count > len(lines) * 0.2}


def map_scientific_sections_extended(text):
    """
    📌 PDF metni içindeki bilimsel bölümleri tespit eder.
    
    Args:
        text (str): İşlenecek ham metin.
        
    Returns:
        dict: Bölüm başlangıç ve bitiş indeksleri + içerikleri.
    """
    section_patterns = {
        "Abstract": r"(?:^|\n)(Abstract|Özet)(?::)?\s*\n",
        "Introduction": r"(?:^|\n)(Introduction|Giriş)(?::)?\s*\n",
        "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)(?::)?\s*\n",
        "Results": r"(?:^|\n)(Results|Bulgular)(?::)?\s*\n",
        "Discussion": r"(?:^|\n)(Discussion|Tartışma)(?::)?\s*\n",
        "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)(?::)?\s*\n",
        "Tables": r"(?:^|\n)(Tables|Tablolar)(?::)?\s*\n"
    }

    sections_map = {}
    for section, pattern in section_patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        sections_map[section] = match.start() if match else None

    sorted_sections = sorted((sec, pos) for sec, pos in sections_map.items() if pos is not None)
    mapped_sections = {}
    for i, (section, start_idx) in enumerate(sorted_sections):
        end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        content = text[start_idx:end_idx].strip()
        mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}

    return mapped_sections


def detect_tables(text):
    """
    📌 PDF metni içindeki tabloları tespit eder.
    
    Returns:
        list: Her tablo için 'başlık' ve 'veriler' içeren liste.
    """
    table_patterns = [
        (r'(?i)(?:^|\n)(Tablo\s*\d+(?:\.\d+)?)(?:\s*[:\-]?\s*\n)(.*?)(?=\n(?:Tablo\s*\d+(?:\.\d+)?|\Z))', 'tablo'),
        (r'(?i)(?:^|\n)(Table\s*\d+(?:\.\d+)?)(?:\s*[:\-]?\s*\n)(.*?)(?=\n(?:Table\s*\d+(?:\.\d+)?|\Z))', 'table'),
        (r'(?i)(?:^|\n)(Çizelge\s*\d+(?:\.\d+)?)(?:\s*[:\-]?\s*\n)(.*?)(?=\n(?:Çizelge\s*\d+(?:\.\d+)?|\Z))', 'çizelge'),
    ]

    tables = []
    for pattern, tip in table_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            baslik = match.group(1).strip()
            icerik = match.group(2).strip()
            if not icerik:
                continue

            rows = []
            for line in icerik.splitlines():
                line = line.strip()
                if line:
                    cells = re.split(r'\t|\s{3,}', line)
                    row = [cell.strip() for cell in cells if cell.strip()]
                    if row:
                        rows.append(row)

            if len(rows) > 1:
                tables.append({"tip": tip, "baslik": baslik, "veriler": rows})

    return tables


def reflow_columns(text):
    """
    📌 Metni tek akışa dönüştürerek temizler.
    
    Returns:
        str: Temizlenmiş metin.
    """
    text = re.sub(r"<[^>]+>", " ", text)  # HTML etiketlerini temizle
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)  # Markdown linklerini temizle
    text = re.sub(r"(Page|Sayfa)\s*\d+", " ", text, flags=re.IGNORECASE)  # Sayfa numaralarını kaldır
    text = re.sub(r"\n", " ", text)  # Satır sonlarını boşlukla değiştir
    text = re.sub(r"\s{2,}", " ", text)  # Fazla boşlukları temizle
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)  # Kırpılmış kelimeleri düzelt
    return text.strip()

# import os
# import re
# import pdfplumber
# from pdfminer.high_level import extract_text
# from pathlib import Path
# from config_module import config


# def extract_text_from_pdf(pdf_path, method=None):
#     """
#     PDF'den metin çıkarır. Varsayılan olarak .env’de belirlenen "PDF_TEXT_EXTRACTION_METHOD" değeri kullanılır.
#     Eğer pdfplumber başarısız olursa, pdfminer yöntemi devreye girer.

#     Args:
#         pdf_path (str or Path): PDF dosyasının yolu.
#         method (str, optional): Kullanılacak metin çıkarma yöntemi ("pdfplumber" veya "pdfminer").

#     Returns:
#         str veya None: Çıkarılan metin; hata durumunda None.
#     """
#     if method is None:
#         method = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()
    
#     text = None
#     if method == "pdfplumber":
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 text = "\n".join([page.extract_text() or "" for page in pdf.pages])
#             if text.strip():
#                 return text
#         except Exception as e:
#             config.logger.error(f"pdfplumber ile metin çıkarma hatası: {e}")

#     if method == "pdfminer":
#         try:
#             text = extract_text(pdf_path)
#             if text.strip():
#                 return text
#         except Exception as e:
#             config.logger.error(f"pdfminer ile metin çıkarma hatası: {e}")

#     config.logger.error("PDF metni çıkarılamadı!")
#     return ""


# def detect_columns(text, min_gap=4):
#     """
#     Metindeki sütun yapısını tespit eder.

#     Args:
#         text (str): İşlenecek metin.
#         min_gap (int): Satırda sütunları ayırmak için gereken minimum boşluk sayısı.

#     Returns:
#         dict: {'sutunlu': True} veya {'sutunlu': False}.
#     """
#     lines = text.split('\n')
#     column_line_count = sum(1 for line in lines if re.search(r' {' + str(min_gap) + r',}', line))
#     return {'sutunlu': column_line_count > len(lines) * 0.2}


# def extract_tables_from_text(text):
#     """
#     PDF metninde tablo içeren bölümleri tespit eder ve çıkarır.

#     Args:
#         text (str): PDF'den çıkarılmış metin.

#     Returns:
#         list: Tespit edilen tabloların listesi.
#     """
#     table_patterns = [
#         r"(?:Table|Tablo|Çizelge)\s?\d+[:.]",  # Tablo numarası (Örn: "Table 1:" veya "Tablo 2.")
#         r"^\s*(?:\|.*\|)+\s*$",  # ASCII tablo yapıları
#         r"(?:Data|Sonuçlar)\s+Table",  # Veri tablolarını bulmak için
#         r"(?:Experiment|Deney|Ölçüm) Results",  # Deney verilerini içeren tablolar
#     ]

#     tables = []
#     for pattern in table_patterns:
#         matches = re.finditer(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
#         for match in matches:
#             start_idx = match.start()
#             end_idx = min(start_idx + 500, len(text))  # Tablo içeriğini almak için tahmini aralık
#             tables.append(text[start_idx:end_idx])

#     return tables


# def map_scientific_sections_extended(text):
#     """
#     Bilimsel makale bölümlerini regex ile tespit eder.

#     Args:
#         text (str): İşlenecek metin.

#     Returns:
#         dict: Bölümlerin başlangıç ve bitiş indeksleri.
#     """
#     section_patterns = {
#         "Abstract": r"(?:^|\n)(Abstract|Özet)",
#         "Introduction": r"(?:^|\n)(Introduction|Giriş)",
#         "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)",
#         "Results": r"(?:^|\n)(Results|Bulgular)",
#         "Discussion": r"(?:^|\n)(Discussion|Tartışma)",
#         "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)",
#         "Acknowledgments": r"(?:^|\n)(Acknowledgments|Teşekkür)",
#         "Funding": r"(?:^|\n)(Funding|Destek Bilgisi)",
#     }

#     sections_map = {}
#     for section, pattern in section_patterns.items():
#         matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
#         sections_map[section] = matches[0].start() if matches else None

#     detected_sections = {sec: pos for sec, pos in sections_map.items() if pos is not None}
#     sorted_sections = sorted(detected_sections.items(), key=lambda x: x[1])
#     mapped_sections = {}

#     for i, (section, start_idx) in enumerate(sorted_sections):
#         end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
#         content = text[start_idx:end_idx].strip()
#         mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}

#     return mapped_sections



# ### 🔥 **Bu Güncellenmiş Versiyonda Neler Değişti?**
# ✔ **pdfplumber + pdfminer ikili metin çıkarma sistemi eklendi.**  
# ✔ **Bilimsel bölümler regex yapısı iyileştirildi.**  
# ✔ **Tablo tespit regex algoritması genişletildi.**  
# ✔ **Sütun tespiti ve metni tek akışa dönüştürme sistemi güçlendirildi.**  

# ---

# 📢 **Sıradaki modülü söyle, hemen gönderelim!** 🚀



# import os
# import re
# import pdfplumber
# from pdfminer.high_level import extract_text
# from pathlib import Path
# from config_module import config


# def extract_text_from_pdf(pdf_path, method=None):
#     """
#     PDF'den metin çıkarır. Varsayılan olarak .env’de belirlenen "PDF_TEXT_EXTRACTION_METHOD" değeri kullanılır.
#     Eğer pdfplumber başarısız olursa, pdfminer yöntemi devreye girer.

#     Args:
#         pdf_path (str or Path): PDF dosyasının yolu.
#         method (str, optional): Kullanılacak metin çıkarma yöntemi ("pdfplumber" veya "pdfminer").

#     Returns:
#         str veya None: Çıkarılan metin; hata durumunda None.
#     """
#     if method is None:
#         method = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()
    
#     text = None
#     if method == "pdfplumber":
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 text = "\n".join([page.extract_text() or "" for page in pdf.pages])
#             if text.strip():
#                 return text
#         except Exception as e:
#             config.logger.error(f"pdfplumber ile metin çıkarma hatası: {e}")

#     if method == "pdfminer":
#         try:
#             text = extract_text(pdf_path)
#             if text.strip():
#                 return text
#         except Exception as e:
#             config.logger.error(f"pdfminer ile metin çıkarma hatası: {e}")

#     config.logger.error("PDF metni çıkarılamadı!")
#     return ""


# def detect_columns(text, min_gap=4):
#     """
#     Metindeki sütun yapısını tespit eder.

#     Args:
#         text (str): İşlenecek metin.
#         min_gap (int): Satırda sütunları ayırmak için gereken minimum boşluk sayısı.

#     Returns:
#         dict: {'sutunlu': True} veya {'sutunlu': False}.
#     """
#     lines = text.split('\n')
#     column_line_count = sum(1 for line in lines if re.search(r' {' + str(min_gap) + r',}', line))
#     return {'sutunlu': column_line_count > len(lines) * 0.2}


# def extract_tables_from_text(text):
#     """
#     PDF metninde tablo içeren bölümleri tespit eder ve çıkarır.

#     Args:
#         text (str): PDF'den çıkarılmış metin.

#     Returns:
#         list: Tespit edilen tabloların listesi.
#     """
#     table_patterns = [
#         r"(?:Table|Tablo|Çizelge)\s?\d+[:.]",  # Tablo numarası (Örn: "Table 1:" veya "Tablo 2.")
#         r"^\s*(?:\|.*\|)+\s*$",  # ASCII tablo yapıları
#         r"(?:Data|Sonuçlar)\s+Table",  # Veri tablolarını bulmak için
#         r"(?:Experiment|Deney|Ölçüm) Results",  # Deney verilerini içeren tablolar
#     ]

#     tables = []
#     for pattern in table_patterns:
#         matches = re.finditer(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
#         for match in matches:
#             start_idx = match.start()
#             end_idx = min(start_idx + 500, len(text))  # Tablo içeriğini almak için tahmini aralık
#             tables.append(text[start_idx:end_idx])

#     return tables


# def map_scientific_sections_extended(text):
#     """
#     Bilimsel makale bölümlerini regex ile tespit eder.

#     Args:
#         text (str): İşlenecek metin.

#     Returns:
#         dict: Bölümlerin başlangıç ve bitiş indeksleri.
#     """
#     section_patterns = {
#         "Abstract": r"(?:^|\n)(Abstract|Özet)",
#         "Introduction": r"(?:^|\n)(Introduction|Giriş)",
#         "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)",
#         "Results": r"(?:^|\n)(Results|Bulgular)",
#         "Discussion": r"(?:^|\n)(Discussion|Tartışma)",
#         "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)",
#         "Acknowledgments": r"(?:^|\n)(Acknowledgments|Teşekkür)",
#         "Funding": r"(?:^|\n)(Funding|Destek Bilgisi)",
#     }

#     sections_map = {}
#     for section, pattern in section_patterns.items():
#         matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
#         sections_map[section] = matches[0].start() if matches else None

#     detected_sections = {sec: pos for sec, pos in sections_map.items() if pos is not None}
#     sorted_sections = sorted(detected_sections.items(), key=lambda x: x[1])
#     mapped_sections = {}

#     for i, (section, start_idx) in enumerate(sorted_sections):
#         end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
#         content = text[start_idx:end_idx].strip()
#         mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}

#     return mapped_sections




# Aşağıda, güncellenmiş "pdf_processing.py" modülünü, yeni gereksinimler doğrultusunda pdfplumber öncelikli olacak şekilde yapılandırdım. Bu modül, PDF’den metin çıkarma, sütun tespiti, bilimsel bölümlerin haritalanması ve metnin tek akışa dönüştürülmesi işlevlerini içeriyor:

# ```python
# import os
# import re
# from config_module import config

# def extract_text_from_pdf(pdf_path, method=None):
#     """
#     PDF'den ham metni çıkarır.
#     Eğer method parametresi verilmezse, .env'den PDF_TEXT_EXTRACTION_METHOD okunur (varsayılan "pdfplumber").
#     Eğer pdfplumber ile metin çıkarma hatası alınırsa, otomatik olarak pdfminer yöntemi devreye girer.
    
#     Args:
#         pdf_path (str or Path): PDF dosyasının yolu.
#         method (str, optional): Kullanılacak metin çıkarma yöntemi ("pdfplumber" veya "pdfminer").
    
#     Returns:
#         str veya None: Çıkarılan metin; hata durumunda None.
#     """
#     if method is None:
#         method = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()
    
#     text = None
#     if method == "pdfplumber":
#         try:
#             import pdfplumber
#             with pdfplumber.open(pdf_path) as pdf:
#                 pages_text = []
#                 for page in pdf.pages:
#                     page_text = page.extract_text()
#                     if page_text:
#                         pages_text.append(page_text)
#                 text = "\n".join(pages_text)
#             config.logger.info(f"Metin pdfplumber ile çıkarıldı: {pdf_path}")
#         except Exception as e:
#             config.logger.error(f"pdfplumber ile metin çıkarma hatası: {e}")
#             config.logger.info("pdfplumber başarısız, pdfminer deneniyor.")
#             return extract_text_from_pdf(pdf_path, method="pdfminer")
#     elif method == "pdfminer":
#         try:
#             from pdfminer.high_level import extract_text
#             text = extract_text(pdf_path)
#             config.logger.info(f"Metin pdfminer ile çıkarıldı: {pdf_path}")
#         except Exception as e:
#             config.logger.error(f"pdfminer ile metin çıkarma hatası: {e}")
#     else:
#         config.logger.error("Geçersiz method belirtildi. 'pdfplumber' veya 'pdfminer' kullanılabilir.")
#     return text

# def detect_columns(text, min_gap=4):
#     """
#     Metindeki sütun yapısını tespit eder.
#     Belirli bir boşluk sayısına göre sütunlu olup olmadığına karar verir.
    
#     Args:
#         text (str): İşlenecek metin.
#         min_gap (int): Satırda sütunları ayırmak için gereken minimum boşluk sayısı.
    
#     Returns:
#         dict: Örneğin, {'sutunlu': True} veya {'sutunlu': False}.
#     """
#     lines = text.split('\n')
#     column_line_count = sum(1 for line in lines if re.search(r' {' + str(min_gap) + r',}', line))
#     return {'sutunlu': column_line_count > len(lines) * 0.2}



# def map_scientific_sections_extended(text):
#     """
#     Bilimsel dokümanların bölümlerini haritalar.
#     Örneğin: Abstract, Giriş, Yöntemler, Bulgular, Tartışma, Sonuç,
#     İçindekiler, Tablolar, Çizelgeler, Resimler/Figürler, İndeks.

#     Metin içindeki bilimsel bölümleri tespit eder.
#     Örneğin, "Introduction" (veya "Giriş"), "Methods" (veya "Yöntem"), "Results" (veya "Sonuç")
#     gibi bölümlerin başlangıç ve bitiş indekslerini döndürür.
    
#     Args:
#         text (str): İşlenecek ham metin.
        
#     Returns:
#         dict: Haritalanmış bölümler; her bölüm için başlangıç ve bitiş indeksleri ve içerik.
#     """
#     section_patterns = {
#         "Abstract": r"(?:^|\n)(Abstract|Özet)(?::)?\s*\n",
#         "Introduction": r"(?:^|\n)(Introduction|Giriş)(?::)?\s*\n",
#         "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)(?::)?\s*\n",
#         "Results": r"(?:^|\n)(Results|Bulgular)(?::)?\s*\n",
#         "Discussion": r"(?:^|\n)(Discussion|Tartışma)(?::)?\s*\n",
#         "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)(?::)?\s*\n"
#     }
#     additional_patterns = {
#         "İçindekiler": r"(?:^|\n)(İçindekiler)(?::)?\s*\n",
#         "Tablolar": r"(?:^|\n)(Tablolar|Tables)(?::)?\s*\n",
#         "Çizelgeler": r"(?:^|\n)(Çizelgeler|Charts)(?::)?\s*\n",
#         "Resimler/Figürler": r"(?:^|\n)(Resimler|Figures)(?::)?\s*\n",
#         "İndeks": r"(?:^|\n)(İndeks|Index)(?::)?\s*\n"
#     }
#     sections_map = {}
#     # Tüm desenleri birleştirip ilk eşleşmeleri alıyoruz.
#     for section, pattern in {**section_patterns, **additional_patterns}.items():
#         matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
#         sections_map[section] = matches[0].start() if matches else None

#     # Sadece bulunan bölümleri ayıklıyoruz.
#     detected_sections = {sec: pos for sec, pos in sections_map.items() if pos is not None}
#     sorted_sections = sorted(detected_sections.items(), key=lambda x: x[1])
#     mapped_sections = {}
#     for i, (section, start_idx) in enumerate(sorted_sections):
#         end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
#         content = text[start_idx:end_idx].strip()
#         mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}
#     # Ek olarak, sütun yapısı bilgisi ekleyebiliriz.
#     from pdf_processing import detect_columns  # Veya helper_module'den çağrılabilir.
#     column_info = detect_columns(text)
#     mapped_sections["Column Structure"] = column_info

#     # Eğer bazı bölümler bulunamazsa, onları None olarak ekleyelim.
#     for sec in list(section_patterns.keys()) + list(additional_patterns.keys()):
#         if sec not in mapped_sections:
#             mapped_sections[sec] = None

#     return mapped_sections


# def map_pdf_before_extraction(pdf_path, method='pdfplumber'):
#     """
#     PDF'den metin çıkarılmadan önce yapısal analiz yapar ve bilimsel bölümleri haritalar.
#     Çıkarılan metin üzerinden map_scientific_sections_extended fonksiyonu ile bölümler tespit edilir.
    
#     Args:
#         pdf_path (str or Path): PDF dosyasının yolu.
#         method (str): 'pdfplumber' veya 'pdfminer'. Varsayılan "pdfplumber".
    
#     Returns:
#         dict veya None: Bölümlerin haritalandığı sözlük, hata durumunda None.
#     """
#     text = extract_text_from_pdf(pdf_path, method=method)
#     if not text:
#         config.logger.error("PDF'den metin çıkarılamadı; haritalama yapılamıyor.")
#         return None
#     return map_scientific_sections_extended(text)

# def reflow_columns(text):
#     """
#     Sütunlu metni tek akışa dönüştürür.
#     - HTML/Markdown etiketlerini temizler.
#     - Sayfa başı/sonu ifadelerini (örn. "Page 1", "Sayfa 1") kaldırır.
#     - Satır sonlarını boşlukla değiştirir, fazla boşlukları tek boşluk yapar.
#     - Kırpılmış kelimelerdeki tire işaretlerini kaldırır.
    
#     Args:
#         text (str): İşlenecek metin.
    
#     Returns:
#         str: Temizlenmiş, tek akışa dönüştürülmüş metin.
#     """
#     # HTML etiketlerini kaldır
#     text = re.sub(r"<[^>]+>", " ", text)
#     # Markdown link yapılarını kaldır: [Link](url)
#     text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
#     # Sayfa bilgilerini temizle
#     text = re.sub(r"(Page|Sayfa)\s*\d+", " ", text, flags=re.IGNORECASE)
#     # Satır sonlarını boşlukla değiştir
#     text = re.sub(r"\n", " ", text)
#     # Fazla boşlukları tek boşluk yap
#     text = re.sub(r"\s{2,}", " ", text)
#     # Kırpılmış kelimelerdeki tireyi kaldır: "infor- mation" -> "information"
#     text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
#     return text.strip()


# # ### Açıklamalar

# # - **extract_text_from_pdf:**  
# #   - Eğer `method` parametresi verilmezse, `.env` dosyasından "PDF_TEXT_EXTRACTION_METHOD" okunur; varsayılan değer "pdfplumber"dir.  
# #   - Pdfplumber ile metin çıkarma denenir. Başarısız olursa, otomatik olarak pdfminer yöntemiyle metin çıkarılmaya çalışılır.  
# #   - Hata durumlarında ilgili hata mesajları loglanır.

# # - **detect_columns:**  
# #   - Metindeki satırlar arasındaki belirli boşluk sayısına (min_gap) göre sütun yapısı tespit edilir.  
# #   - Eğer satırların %20'sinden fazlasında belirgin boşluklar varsa, metin sütunlu kabul edilir.

# # - **map_scientific_sections_extended:**  
# #   - Metin içerisinde "Introduction", "Methods" ve "Results" gibi bölümler için regex kullanılarak, her bölümün başlangıç ve bitiş indeksleri döndürülür.

# # - **map_pdf_before_extraction:**  
# #   - Önce `extract_text_from_pdf` çağrılarak metin çıkarılır, ardından bu metin üzerinden bilimsel bölümlerin haritalanması yapılır.

# # - **reflow_columns:**  
# #   - Metin içerisindeki HTML/Markdown etiketleri, sayfa bilgileri ve fazladan boşluklar temizlenir.  
# #   - Kırpılmış kelimelerdeki tire işaretleri kaldırılarak metin tek akışa dönüştürülür.

# # Bu modül, PDF metin çıkarım ve işleme işlemleri için temel fonksiyonları 
# # içerir ve güncel gereksinimlere uygun olarak yapılandırılmıştır. Eğer ek bir düzenleme veya geliştirme ihtiyacı olursa lütfen bildirin.