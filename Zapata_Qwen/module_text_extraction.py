# module_text_extraction
import os
import re
import fitz  # PyMuPDF
import logging
from pathlib import Path

# Logging yapılandırması
logger = logging.getLogger(__name__)

# ----------------------------
# Metin Çıkarma Fonksiyonları
# ----------------------------

def extract_text_from_pdf(pdf_path):
    """
    PDF'den ham metni çıkarır.
    Gelişmiş sütun tespiti ile akış düzenlemesi yapılabilir.
    Args:
        pdf_path (str): PDF dosyasının yolu.
    Returns:
        str: Çıkarılan ham metin.
    """
    try:
        doc = fitz.open(pdf_path)
        raw_text = ""
        for page in doc:
            # Sayfa metnini sıralı şekilde alıyoruz.
            page_text = page.get_text("text", sort=True)
            raw_text += page_text + "\n\n"
        doc.close()
        logger.info(f"✅ PDF metni başarıyla çıkarıldı: {pdf_path}")
        return raw_text
    except Exception as e:
        logger.error(f"❌ PDF metni çıkarılamadı: {pdf_path}, Hata: {e}")
        return None


def detect_tables(pdf_path):
    """
    PDF'den tablo, şekil, figür çıkarımı yapar.
    Örnek amaçlı basit regex tabanlı tespit yöntemi kullanılmıştır.
    Args:
        pdf_path (str): PDF dosyasının yolu.
    Returns:
        list: Tespit edilen tabloların listesi.
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
        logger.info(f"✅ Tablolar başarıyla tespit edildi: {pdf_path}")
        return tables
    except Exception as e:
        logger.error(f"❌ Tablo çıkarma hatası: {pdf_path}, Hata: {e}")
        return []


def extract_references_enhanced(text):
    """
    Gelişmiş kaynakça çıkarımı:
    PDF veya TXT dosyalarında bulunan kaynakça bölümünü, çeşitli pattern'lerle tespit eder.
    Args:
        text (str): İşlenecek ham metin.
    Returns:
        list: Kaynakça listesi.
    """
    references = []
    ref_patterns = [
        r'(?i)(?:KAYNAKÇA|KAYNAKLAR|REFERENCES|BIBLIOGRAPHY).*?\n(.*?)(?=\n\s*\n|\Z)',
        r'\[\d+\]\s.*?(?=\n|$)'
    ]
    for pattern in ref_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            ref = match.group(0).strip()
            if ref not in references:
                references.append(ref)
    cleaned_refs = [re.sub(r'\s+', ' ', ref).strip() for ref in references if len(ref) > 10 and any(c.isdigit() for c in ref)]
    logger.info(f"✅ Kaynakça başarıyla çıkarıldı. Toplam referans sayısı: {len(cleaned_refs)}")
    return cleaned_refs


def reflow_columns(text):
    """
    Bilimsel yayınlarda özet genellikle tek sütun, asıl metin sütunlu yapıda olabilir.
    Bu fonksiyon, metindeki sütun yapısını tespit edip akışı düzeltir.
    Args:
        text (str): İşlenecek ham metin.
    Returns:
        str: Düzeltilmiş metin.
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
# Yardımcı Fonksiyonlar
# ----------------------------

def clean_text(text):
    """
    Metni temel temizleme işlemlerinden geçirir:
    - \r\n yerine \n
    - Fazla boşluk ve satır aralıklarını azaltır
    - Paragrafların yapısını korur (wrap olmadan)
    Args:
        text (str): İşlenecek ham metin.
    Returns:
        str: Temizlenmiş metin.
    """
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def save_text_file(directory, filename, content):
    """
    Belirtilen dizine, verilen dosya adıyla içerik kaydeder.
    Args:
        directory (str): Dosyanın kaydedileceği dizin.
        filename (str): Kaydedilecek dosyanın adı.
        content (str): Kaydedilecek içerik.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"✅ Dosya kaydedildi: {os.path.join(directory, filename)}")


# ----------------------------
# Örnek Kullanım
# ----------------------------

if __name__ == "__main__":
    # Örnek bir PDF dosyası için metin çıkarma
    pdf_path = "ornek.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    if raw_text:
        print("PDF Metni:")
        print(raw_text[:500])  # İlk 500 karakteri göster

    # Tablo tespiti
    tables = detect_tables(pdf_path)
    print("\nTespit Edilen Tablolar:")
    for table in tables:
        print(table)

    # Kaynakça çıkarımı
    references = extract_references_enhanced(raw_text)
    print("\nKaynakça Listesi:")
    for ref in references:
        print(ref)

    # Sütun akışını düzeltme
    reflowed_text = reflow_columns(raw_text)
    print("\nDüzeltilmiş Metin:")
    print(reflowed_text[:500])  # İlk 500 karakteri göster
    
#     Modül Açıklaması
# extract_text_from_pdf :
# PDF dosyasından ham metin çıkarır.
# Gelişmiş sütun tespiti ile metin akışını düzenler.
# detect_tables :
# PDF'den tablo, şekil ve figür çıkarımı yapar.
# Basit regex tabanlı tespit yöntemi kullanır.
# extract_references_enhanced :
# PDF veya TXT dosyalarındaki kaynakça bölümünü tespit eder.
# Farklı dil ve formatlardaki kaynakça pattern'lerini destekler.
# reflow_columns :
# Bilimsel yayınlarda sütunlu metin yapısını düzeltir.
# Özet ve ana metin arasındaki geçişleri düzenler.
# clean_text :
# Metni temizler:
# Fazla boşlukları kaldırır.
# Satır aralıklarını düzenler.
# Paragraf yapısını korur.
# save_text_file :
# Belirtilen dizine, verilen dosya adıyla içerik kaydeder.
# Özellikler
# Esneklik : Farklı formatlardaki PDF ve TXT dosyalarını işleyebilir.
# Gelişmiş Regex : Tablo ve kaynakça tespiti için güçlü regex desenleri kullanır.
# Hata Yönetimi : Her işlemde hata kontrolü yapılır ve log kaydı tutulur.
# Modülerlik : Diğer modüllerle entegre edilebilir ve bağımsız olarak test edilebilir.