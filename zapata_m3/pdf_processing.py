import os
import re
from config_module import config


def extract_text_from_pdf(pdf_path, method=None):
    """
    PDF'den ham metni çıkarır.

    Args:
        pdf_path (str or Path): PDF dosyasının yolu.
        method (str): 'pdfminer' veya 'textract'. Varsayılan 'pdfminer'.
        method (str, optional): Metin çıkarma yöntemi. Belirtilmezse, .env'den PDF_TEXT_EXTRACTION_METHOD okunur.

    Returns:
        str veya None: Çıkarılan metin; hata durumunda None.
    """
    # Ortam değişkeninden yöntemi al, varsayılan "pdfminer" olacak
    if method is None:
        method = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfminer").lower()
    
    text = None
    if method == 'pdfplumber':
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n".join(pages_text)
            config.logger.info(f"pdfplumber ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"pdfplumber ile metin çıkarma hatası: {e}")
            config.logger.info("pdfplumber başarısız, pdfminer deneniyor.")
            return extract_text_from_pdf(pdf_path, method='pdfminer')
    elif method == 'pdfminer':
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            config.logger.info(f"pdfminer ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"pdfminer ile metin çıkarma hatası: {e}")
    else:
        config.logger.error("Geçersiz method belirtildi. 'pdfplumber' veya 'pdfminer' kullanılabilir.")
    return text

# extextract, pdfminer ile uyumsuz oldugundan textextrat kaldirildi.
# def extract_text_from_pdf(pdf_path, method='pdfminer'):
#     text = None
#     if method.lower() == 'pdfminer':
#         try:
#             from pdfminer.high_level import extract_text
#             text = extract_text(pdf_path)
#         except Exception as e:
#             config.logger.error(f"pdfminer ile metin çıkarma hatası: {e}")
#     elif method.lower() == 'textract':
#         try:
#             import textract
#             text_bytes = textract.process(pdf_path)
#             text = text_bytes.decode('utf-8', errors='replace')
#         except Exception as e:
#             config.logger.error(f"textract ile metin çıkarma hatası: {e}")
#     else:
#         config.logger.error("Geçersiz method belirtildi. 'pdfminer' veya 'textract' kullanılabilir.")
#     return text

def detect_columns(text, min_gap=4):
    """
    Metindeki sütun yapısını tespit eder.
    
    Args:
        text (str): İşlenecek metin.
        min_gap (int): Satırda sütunları ayırmak için gereken minimum boşluk sayısı.
        
    Returns:
        dict: Örneğin, {'sutunlu': True} veya {'sutunlu': False}.
    """
    lines = text.split('\n')
    column_line_count = 0
    for line in lines:
        if re.search(r' {' + str(min_gap) + r',}', line):
            column_line_count += 1
    return {'sutunlu': column_line_count > len(lines) * 0.2}

def map_scientific_sections_extended(text):
    """
    Metin içindeki bilimsel bölümleri tespit eder.
    Örneğin: "Introduction"/"Giriş", "Methods"/"Yöntem" ve "Results"/"Sonuç" bölümlerinin başlangıç ve bitiş indekslerini döndürür.
    
    Args:
        text (str): İşlenecek metin.
        
    Returns:
        dict: Bölümlerin başlangıç ve bitiş indeksleri.
    """
    sections = {
        "Introduction": re.search(r'(?i)(introduction|giriş)', text),
        "Methods": re.search(r'(?i)(method[s]?|yöntem)', text),
        "Results": re.search(r'(?i)(result[s]?|sonuç)', text)
    }
    result = {}
    for section, match in sections.items():
        result[section] = match.span() if match else None
    return result

def reflow_columns(text):
    """
    Sütunlu metni tek akışa dönüştürür. Aşağıdaki adımları uygular:
      - HTML ve Markdown etiketlerini temizler.
      - Sayfa başı/sonu bilgilerini (örn. "Page 1", "Sayfa 1") kaldırır.
      - Satır sonlarını boşlukla değiştirir ve fazla boşlukları tek boşluk yapar.
      - Satır sonlarında kırpılan kelimelerdeki tire işaretlerini kaldırır.
      
    Args:
        text (str): İşlenecek metin.
        
    Returns:
        str: Temizlenmiş, tek akışa dönüştürülmüş metin.
    """
    # HTML etiketlerini kaldır
    text = re.sub(r"<[^>]+>", " ", text)
    # Markdown link yapılarını kaldır: [Link](url)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    # Sayfa bilgilerini temizle
    text = re.sub(r"(Page|Sayfa)\s*\d+", " ", text, flags=re.IGNORECASE)
    # Satır sonlarını boşlukla değiştir
    text = re.sub(r"\n", " ", text)
    # Fazla boşlukları tek boşluk yap
    text = re.sub(r"\s{2,}", " ", text)
    # Kırpılmış kelimelerdeki tire işaretlerini kaldır: "infor- mation" -> "information"
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text.strip()

# =======================================================================
# onceki program parcasi

# def detect_columns(text, min_gap=4):
#     """
#     Metindeki sütun yapısını tespit eder.
    
#     Args:
#         text (str): İşlenecek metin.
#         min_gap (int): Satırda sütunları ayırmak için gereken minimum boşluk sayısı.
        
#     Returns:
#         dict: Örneğin, {'sutunlu': True} veya {'sutunlu': False}.
#     """
#     lines = text.split('\n')
#     column_line_count = 0
#     for line in lines:
#         if re.search(r' {' + str(min_gap) + r',}', line):
#             column_line_count += 1
#     return {'sutunlu': column_line_count > len(lines) * 0.2}


# def map_scientific_sections_extended(text):
#     """
#     Metin içindeki bilimsel bölümleri tespit eder.
#     Örneğin: "Introduction"/"Giriş", "Methods"/"Yöntem" ve "Results"/"Sonuç" bölümlerinin başlangıç ve bitiş indekslerini döndürür.
    
#     Args:
#         text (str): İşlenecek metin.
        
#     Returns:
#         dict: Bölümlerin başlangıç ve bitiş indeksleri. Örnek:
#               {"Introduction": (10, 150), "Methods": (151, 300), "Results": (301, 450)}
#               Eğer bir bölüm bulunamazsa değeri None olur.
#     """
#     sections = {
#         "Introduction": re.search(r'(?i)(introduction|giriş)', text),
#         "Methods": re.search(r'(?i)(method[s]?|yöntem)', text),
#         "Results": re.search(r'(?i)(result[s]?|sonuç)', text)
#     }
#     result = {}
#     for section, match in sections.items():
#         result[section] = match.span() if match else None
#     return result

# def reflow_columns(text):
#     """
#     Sütunlu metni tek akışa dönüştürür. Aşağıdaki adımları uygular:
#       - HTML ve Markdown etiketlerini temizler.
#       - Sayfa başı/sonu bilgilerini (örn. "Page 1") kaldırır.
#       - Satır sonlarını boşluk ile değiştirir ve fazla boşlukları tek boşluk yapar.
#       - Satır sonlarında kırpılan kelimelerdeki tire işaretlerini kaldırır.
      
#     Args:
#         text (str): İşlenecek metin.
        
#     Returns:
#         str: Temizlenmiş, tek akışa dönüştürülmüş metin.
#     """
#     # HTML etiketlerini kaldır (basit regex, ihtiyaca göre genişletilebilir)
#     text = re.sub(r"<[^>]+>", " ", text)
#     # Markdown etiketlerini kaldır (örneğin, [Link](url) yapıları)
#     text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
#     # Sayfa başı/sonu bilgilerini temizle (örneğin, "Page 1" ifadeleri)
#     text = re.sub(r"Page\s*\d+", " ", text)
#     # Satır sonlarını boşlukla değiştir
#     text = re.sub(r"\n", " ", text)
#     # Fazla boşlukları tek boşluk yap
#     text = re.sub(r"\s{2,}", " ", text)
#     # Kırpılmış kelimelerdeki tireyi kaldır: "infor- mation" -> "information"
#     text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
#     return text.strip()
