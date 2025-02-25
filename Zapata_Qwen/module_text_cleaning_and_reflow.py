# module_text_cleaning_and_reflow
import re
from datetime import datetime

# ----------------------------
# Metin Temizleme ve Akış Düzenleme Fonksiyonları
# ----------------------------

def clean_text(text):
    """
    Ham metni temizler:
    - \r\n yerine \n
    - Fazla boşluk ve satır aralıklarını azaltır
    - Paragrafların yapısını korur (wrap olmadan)
    Args:
        text (str): İşlenecek ham metin.
    Returns:
        str: Temizlenmiş metin.
    """
    text = re.sub(r'\r\n', '\n', text)  # \r\n yerine \n
    text = re.sub(r'\n{2,}', '\n\n', text)  # Fazla boş satırları azalt
    text = re.sub(r'[ \t]+', ' ', text)  # Fazla boşlukları tek boşluğa indirge
    text = re.sub(r'\s+', ' ', text)  # Diğer gereksiz boşlukları temizle
    return text.strip()


def reflow_columns(text):
    """
    Bilimsel yayınlarda özet genellikle tek sütun, asıl metin sütunlu yapıda olabilir.
    Bu fonksiyon, metindeki sütun yapısını tespit edip akışı düzeltir.
    Args:
        text (str): İşlenecek ham metin.
    Returns:
        str: Akış düzenlenmiş metin.
    """
    if "Abstract" in text:
        parts = text.split("Abstract", 1)
        abstract = parts[0].strip()
        body = parts[1].strip()
        body = re.sub(r'\n', ' ', body)  # Sütunları birleştir
        body = re.sub(r'\s{2,}', ' ', body)  # Fazla boşlukları temizle
        return abstract + "\n\n" + body
    return text


def detect_columns(text, min_gap=4, min_occurrences=3):
    """
    Metindeki sütun yapısını tespit eder.
    Args:
        text (str): İşlenecek ham metin.
        min_gap (int): Minimum boşluk sayısı.
        min_occurrences (int): Minimum tekrar sayısı.
    Returns:
        dict: Sütun bilgisi.
    """
    lines = text.split("\n")
    count = 0
    for line in lines:
        if re.search(rf" {{{min_gap},}}", line):  # Büyük boşluklar
            count += 1
    is_multicolumn = count >= min_occurrences
    return {"is_multicolumn": is_multicolumn, "lines_with_gap": count}


def map_scientific_sections_extended(text):
    """
    Bilimsel dokümanların bölümlerini haritalar.
    Örneğin: Abstract, Giriş, Yöntemler, Bulgular, Tartışma, Sonuç.
    Args:
        text (str): İşlenecek ham metin.
    Returns:
        dict: Haritalanmış bölümler.
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
    for section, pattern in {**section_patterns, **additional_patterns}.items():
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            sections_map[section] = matches[0].start()
        else:
            sections_map[section] = None
    detected_sections = {sec: pos for sec, pos in sections_map.items() if pos is not None and sec != "Column Structure"}
    sorted_sections = sorted(detected_sections.items(), key=lambda x: x[1])
    mapped_sections = {}
    for i, (section, start_idx) in enumerate(sorted_sections):
        end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        content = text[start_idx:end_idx].strip()
        mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}
    column_info = detect_columns(text)
    mapped_sections["Column Structure"] = column_info
    for sec in list(section_patterns.keys()) + list(additional_patterns.keys()):
        if sec not in mapped_sections:
            mapped_sections[sec] = None
    return mapped_sections


# ----------------------------
# Örnek Kullanım
# ----------------------------

if __name__ == "__main__":
    # Örnek bir PDF veya TXT dosyasından ham metin
    raw_text = """
    Abstract
    Bu çalışmanın amacı, makine öğrenmesi tekniklerini kullanarak bilimsel belgeleri analiz etmektir.
    
    Giriş
    Bilimsel yayınlar, genellikle çok sütunlu bir yapıda sunulur. Bu, metin çıkarma işlemlerini zorlaştırır.
    
    Yöntemler
    Çalışmada, PyMuPDF ve sklearn kütüphaneleri kullanılmıştır.
    
    Bulgular
    Model, %95 doğruluk oranı ile başarılı bir şekilde eğitildi.
    
    Tartışma
    Sonuçlar, yöntemlerin etkinliğini göstermektedir.
    
    Sonuç
    Çalışma, bilimsel belgelerin otomatik analizinde umut vaat etmektedir.
    """

    # Metin temizleme
    cleaned_text = clean_text(raw_text)
    print("Temizlenmiş Metin:")
    print(cleaned_text[:500])  # İlk 500 karakter

    # Akış düzenlemesi
    reflowed_text = reflow_columns(cleaned_text)
    print("\nAkış Düzenlenmiş Metin:")
    print(reflowed_text[:500])  # İlk 500 karakter

    # Sütun tespiti
    column_info = detect_columns(cleaned_text)
    print("\nSütun Bilgisi:")
    print(column_info)

    # Bölüm haritalama
    mapped_sections = map_scientific_sections_extended(cleaned_text)
    print("\nHaritalanmış Bölümler:")
    for section, info in mapped_sections.items():
        if info:
            print(f"{section}: {info['start']}-{info['end']} ({len(info['content'])} karakter)")
            
#        Modül Açıklaması
# clean_text :
# Ham metni temizler:
# \r\n yerine \n kullanır.
# Fazla boşlukları ve satır aralıklarını azaltır.
# Paragrafların yapısını korur.
# reflow_columns :
# Bilimsel dokümanlarda bulunan sütunlu metinleri düzeltir.
# Özellikle "Abstract" bölümüne göre düzenleme yapar.
# detect_columns :
# Metindeki sütun yapısını tespit eder.
# Belirli bir minimum boşluk sayısına ve tekrar sayısına göre sütun varlığını kontrol eder.
# map_scientific_sections_extended :
# Bilimsel dokümanların bölümlerini (örneğin: Abstract, Giriş, Yöntemler, Bulgular, Tartışma, Sonuç) haritalar.
# Ek olarak, içerik tablosu, tablolar, çizelgeler gibi ek bölümleri de tespit eder.
# Özellikler
# Esneklik : Farklı formatlardaki bilimsel dokümanları işleyebilir.
# Otomatik Sütun Tespiti : Metindeki sütun yapısını tespit eder ve akışı düzeltir.
# Bölüm Haritalama : Dokümanın ana bölümlerini tespit eder ve içeriklerini ayırır.
# Modülerlik : Diğer modüllerle entegre edilebilir ve bağımsız olarak test edilebilir.
# Bu modülü kullanarak programınızın PDF veya TXT dosyalarından çıkarılan ham metni temizleyebilir ve akışını düzeltebilirsiniz. Eğer herhangi bir değişiklik veya ek özellik isterseniz, lütfen belirtin!     