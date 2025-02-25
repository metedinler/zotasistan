# ex

import os
import re
from config_module import config

def extract_text_from_pdf(pdf_path, method=None):
    """
    PDF'den ham metni çıkarır.
    Eğer method parametresi verilmezse, .env dosyasından "PDF_TEXT_EXTRACTION_METHOD" okunur 
    (varsayılan: "pdfplumber"). Eğer pdfplumber ile metin çıkarma hatası alınırsa, otomatik olarak 
    pdfminer yöntemi devreye girer.
    
    Args:
        pdf_path (str or Path): PDF dosyasının yolu.
        method (str, optional): Kullanılacak metin çıkarma yöntemi ("pdfplumber" veya "pdfminer").
    
    Returns:
        str or None: Çıkarılan metin; hata durumunda None.
    """
    if method is None:
        method = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()

    text = None
    if method == "pdfplumber":
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n".join(pages_text)
            config.logger.info(f"✅ pdfplumber ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"❌ pdfplumber ile metin çıkarma hatası: {e}. pdfminer deneniyor.")
            return extract_text_from_pdf(pdf_path, method="pdfminer")
    elif method == "pdfminer":
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            config.logger.info(f"✅ pdfminer ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"❌ pdfminer ile metin çıkarma hatası: {e}")
    else:
        config.logger.error("Geçersiz method belirtildi. 'pdfplumber' veya 'pdfminer' kullanılabilir.")
    return text

def detect_columns(text, min_gap=4):
    """
    Metindeki sütun yapısını tespit eder.
    Belirli bir boşluk sayısına göre metnin sütunlu olup olmadığını belirler.
    
    Args:
        text (str): İşlenecek metin.
        min_gap (int): Bir satırda sütunları ayırmak için gereken minimum boşluk sayısı.
    
    Returns:
        dict: Örnek: {'sutunlu': True} veya {'sutunlu': False}
    """
    lines = text.split('\n')
    column_line_count = sum(1 for line in lines if re.search(r' {' + str(min_gap) + r',}', line))
    return {'sutunlu': column_line_count > len(lines) * 0.2}

def map_scientific_sections_extended(text):
    """
    Bilimsel dokümanların bölümlerini haritalar.
    Örneğin: Abstract, Introduction, Methods, Results, Discussion, Conclusion,
    İçindekiler, Tablolar, Çizelgeler, Resimler/Figürler, İndeks.
    (Geliştirilmiş versiyonda funding ve acknowledge gibi bölümler de eklenebilir.)
    
    Args:
        text (str): İşlenecek ham metin.
        
    Returns:
        dict: Haritalanmış bölümler; her bölüm için "start", "end" ve "content" bilgileri.
    """
    section_patterns = {
        "Abstract": r"(?:^|\n)(Abstract|Özet)(?::)?\s*\n",
        "Introduction": r"(?:^|\n)(Introduction|Giriş)(?::)?\s*\n",
        "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)(?::)?\s*\n",
        "Results": r"(?:^|\n)(Results|Bulgular)(?::)?\s*\n",
        "Discussion": r"(?:^|\n)(Discussion|Tartışma)(?::)?\s*\n",
        "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)(?::)?\s*\n"
    }
    additional_patterns = {
        "İçindekiler": r"(?:^|\n)(İçindekiler)(?::)?\s*\n",
        "Tablolar": r"(?:^|\n)(Tablolar|Tables)(?::)?\s*\n",
        "Çizelgeler": r"(?:^|\n)(Çizelgeler|Charts)(?::)?\s*\n",
        "Resimler/Figürler": r"(?:^|\n)(Resimler|Figures)(?::)?\s*\n",
        "İndeks": r"(?:^|\n)(İndeks|Index)(?::)?\s*\n"
    }
    sections_map = {}
    # Her bölüm için ilk eşleşmenin başlangıç indeksini alıyoruz.
    for section, pattern in {**section_patterns, **additional_patterns}.items():
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        sections_map[section] = matches[0].start() if matches else None

    # Sadece bulunan bölümleri ayıkla
    detected_sections = {sec: pos for sec, pos in sections_map.items() if pos is not None}
    sorted_sections = sorted(detected_sections.items(), key=lambda x: x[1])
    mapped_sections = {}
    for i, (section, start_idx) in enumerate(sorted_sections):
        end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        content = text[start_idx:end_idx].strip()
        mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}
    # Ek olarak, sütun yapısı bilgisi ekleyelim
    column_info = detect_columns(text)
    mapped_sections["Column Structure"] = column_info

    # Eğer bazı bölümler bulunamazsa, onları None olarak ekleyelim
    for sec in list(section_patterns.keys()) + list(additional_patterns.keys()):
        if sec not in mapped_sections:
            mapped_sections[sec] = None

    return mapped_sections

import layoutparser as lp
import fitz  # PyMuPDF
import os
import json
from config_module import config

def map_pdf_before_extraction(pdf_path, method='pdfplumber'):
  

    """
    PDF'den metin çıkarılmadan önce, layout-parser kullanarak bilimsel yayın yapısını analiz eder.
    
    Bu fonksiyon, PDF dosyasının her sayfasını inceler, sayfa boyutlarını ve blok yapılarını tespit eder 
    ve bu bilgileri bir sözlük olarak döndürür.
    
    Args:
        pdf_path (str or Path): PDF dosyasının yolu.
        method (str): Bu parametre gelecekte farklı metot seçenekleri için kullanılabilir; 
                      şimdilik layout-parser ile çalışıyor (varsayılan "pdfplumber" değeri korunuyor).
    
    Returns:
        dict or None: PDF'nin layout analizi bilgilerini içeren sözlük; hata durumunda None.
    """
    # Layout-parser modelini yükle
    try:
        model = lp.models.PaddleDetectionLayoutModel(
            config_path="lp://PP-OCRv3/ppyolov2_r50vd_dcn_365e_publaynet_infer",
            label_map={
                0: "Text",
                1: "Title",
                2: "List",
                3: "Table",
                4: "Figure",
                5: "Formula",
                6: "Caption",
                7: "Page-Number",
                8: "Section-Header",
                9: "Footnote",
                10: "Body-Text"
            }
        )
    except Exception as e:
        config.logger.error(f"Layout-parser model yüklenirken hata: {e}")
        return None

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        config.logger.error(f"PDF dosyası açılırken hata: {e}")
        return None

    layout_info = []

    for page_number in range(doc.page_count):
        try:
            page = doc.load_page(page_number)
            page_rect = page.rect
            # Yüksek çözünürlük (DPI 300) için ölçeklendirme
            zoom = 300 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            temp_img_path = f"temp_page_{page_number}.png"
            pix.save(temp_img_path)

            # Geçici görüntü dosyasını layout-parser için yükle
            try:
                image = lp.io.read(temp_img_path)
            except Exception as e:
                config.logger.error(f"Sayfa {page_number+1} için görüntü yüklenirken hata: {e}")
                os.remove(temp_img_path)
                continue

            # Layout analizi: Blokları tespit et
            try:
                layout = model.detect(image)
            except Exception as e:
                config.logger.error(f"Sayfa {page_number+1} düzen analizi hatası: {e}")
                os.remove(temp_img_path)
                continue

            blocks = []
            for block in layout:
                blocks.append({
                    "type": block.type,
                    "coordinates": block.coordinates,
                    "score": block.score
                })

            page_info = {
                "page_number": page_number + 1,
                "dimensions": {"width": page_rect.width, "height": page_rect.height},
                "blocks": blocks
            }
            layout_info.append(page_info)
            os.remove(temp_img_path)
        except Exception as e:
            config.logger.error(f"Sayfa {page_number+1} işlenirken hata: {e}")
            continue

    return {"layout": layout_info}



   

def reflow_columns(text):
    """
    Sütunlu metni tek akışa dönüştürür.
    - HTML/Markdown etiketlerini temizler.
    - Sayfa başı/sonu bilgilerini (örn. "Page 1", "Sayfa 1") kaldırır.
    - Fazla boşlukları tek boşluk yapar.
    - Kırpılmış kelimelerdeki tire işaretlerini kaldırır.
    
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
    # Kırpılmış kelimelerdeki tireyi kaldır: "infor- mation" -> "information"
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text.strip()

# Aşağıda, tartışmalarımız ve yapılan güncellemeler doğrultusunda oluşturulmuş, 
# final versiyonunu bulabileceğiniz pdf_processing.py modülünün eksiksiz halini paylaşıyorum. Bu versiyon;

# Ortam değişkenlerinden (örn. PDF_TEXT_EXTRACTION_METHOD) ayarlara göre öncelikli olarak pdfplumber kullanıyor, başarısız olursa pdfminer’a geçiyor.
# Metin çıkarıldıktan sonra, HTML/Markdown etiketleri, sayfa başı/sonu ifadeleri ve fazladan boşluklar temizlenerek tek akışa dönüştürülüyor (reflow_columns).
# Bilimsel bölümlerin haritalanması için gelişmiş regex desenleri (Abstract, Introduction, Methods, Results, Discussion, Conclusion ve ek bölümler) kullanılıyor.
# Sütun tespiti için detect_columns fonksiyonu mevcut; istenirse, ek tablo tespiti algoritmaları da entegre edilebilir.
# Hata durumlarında config.logger üzerinden ayrıntılı loglama sağlanıyor.
# Aşağıda final kodu bulabilirsiniz:

# Açıklamalar
# extract_text_from_pdf:

# .env dosyasından PDF metin çıkarma yöntemi okunuyor (varsayılan: pdfplumber).
# Pdfplumber kullanılarak metin çıkarılıyor; hata durumunda pdfminer ile deneme yapılıyor.
# Hata mesajları config.logger ile loglanıyor.
# detect_columns:

# Metindeki satırların belirli boşluk sayılarına göre sütun yapısı tespit ediliyor.
# Eğer satırların %20'sinden fazlasında belirtilen boşluk (min_gap) varsa, metnin sütunlu olduğu kabul ediliyor.
# map_scientific_sections_extended:

# Bilimsel bölümlerin (Abstract, Introduction, Methods, Results, Discussion, Conclusion ve ek bölümler) başlangıç ve bitiş indeksleri ile içerikleri çıkarılıyor.
# Ek olarak, detect_columns çağrılarak sütun yapısı bilgisi de ekleniyor.
# map_pdf_before_extraction:

# PDF’den metin çıkarma yöntemi çağrılır ve elde edilen metin üzerinden bilimsel bölüm haritalaması yapılır.
# reflow_columns:

# HTML/Markdown etiketleri, sayfa bilgileri temizlenir; satır sonları boşlukla değiştirilir; fazla boşluklar ve kırpılmış kelimelerdeki tireler kaldırılır.
# Bu modül, PDF dosyalarından metin çıkarımı ve işleme işlemlerinde temel işlevleri yerine getiriyor. Eğer başka geliştirme veya ek özellik isterseniz, lütfen belirtin.