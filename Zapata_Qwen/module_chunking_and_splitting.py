# module_chunking_and_splitting.py

import re
from typing import List, Dict

# ----------------------------
# Metin Bölme ve Parçalama Fonksiyonları
# ----------------------------

def split_text_into_chunks(text: str, chunk_size: int = 256) -> List[str]:
    """
    Metni önce paragraflara bölüp, ardından her paragrafı belirli kelime sayısına göre küçük parçalara ayırır.
    Böylece wrap olmadan orijinal paragraf yapısı korunur.
    Args:
        text (str): İşlenecek metin.
        chunk_size (int): Her parçadaki maksimum kelime sayısı.
    Returns:
        List[str]: Metin parçaları listesi.
    """
    paragraphs = text.split("\n\n")  # Paragrafları ayır
    chunks = []
    for para in paragraphs:
        words = para.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks


def detect_columns_and_reflow(text: str, min_gap: int = 4, min_occurrences: int = 3) -> str:
    """
    Bilimsel yayınlarda özet genellikle tek sütun, asıl metin sütunlu yapıda olabilir.
    Bu fonksiyon, metindeki sütun yapısını tespit edip akışı düzeltir.
    Args:
        text (str): İşlenecek metin.
        min_gap (int): Sütunlar arasındaki minimum boşluk sayısı.
        min_occurrences (int): Minimum tekrar sayısı.
    Returns:
        str: Akış düzenlenmiş metin.
    """
    lines = text.split("\n")
    count = 0
    for line in lines:
        if re.search(rf" {{{min_gap},}}", line):  # Büyük boşluklar
            count += 1
    is_multicolumn = count >= min_occurrences

    if is_multicolumn:
        # Sütunları birleştir
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)  # Fazla boşlukları temizle
    return text


def map_scientific_sections(text: str) -> Dict[str, Dict]:
    """
    Bilimsel dokümanların bölümlerini haritalar.
    Örneğin: Abstract, Giriş, Yöntemler, Bulgular, Tartışma, Sonuç.
    Args:
        text (str): İşlenecek metin.
    Returns:
        Dict[str, Dict]: Haritalanmış bölümler.
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

    detected_sections = {sec: pos for sec, pos in sections_map.items() if pos is not None}
    sorted_sections = sorted(detected_sections.items(), key=lambda x: x[1])

    mapped_sections = {}
    for i, (section, start_idx) in enumerate(sorted_sections):
        end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        content = text[start_idx:end_idx].strip()
        mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}

    column_info = detect_columns_and_reflow(text)
    mapped_sections["Column Structure"] = column_info

    for sec in list(section_patterns.keys()) + list(additional_patterns.keys()):
        if sec not in mapped_sections:
            mapped_sections[sec] = None

    return mapped_sections


# ----------------------------
# Yardımcı Fonksiyonlar
# ----------------------------

def clean_text(text: str) -> str:
    """
    Metni temizler:
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


def shorten_title(title: str, max_length: int = 80) -> str:
    """
    Başlığı belirtilen uzunluğa göre kısaltır.
    Args:
        title (str): Kısaltılacak başlık.
        max_length (int): Maksimum uzunluk.
    Returns:
        str: Kısaltılmış başlık.
    """
    if len(title) > max_length:
        return title[:max_length]
    return title


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
    reflowed_text = detect_columns_and_reflow(cleaned_text)
    print("\nAkış Düzenlenmiş Metin:")
    print(reflowed_text[:500])  # İlk 500 karakter

    # Bölüm haritalama
    mapped_sections = map_scientific_sections(cleaned_text)
    print("\nHaritalanmış Bölümler:")
    for section, info in mapped_sections.items():
        if info:
            print(f"{section}: {info['start']}-{info['end']} ({len(info['content'])} karakter)")

    # Metni parçalara ayırma
    chunks = split_text_into_chunks(cleaned_text, chunk_size=100)
    print("\nMetin Parçaları:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk[:50]}...")
        
#         Modül Açıklaması
# split_text_into_chunks :
# Metni önce paragraflara böler, ardından her paragrafı belirli kelime sayısına göre küçük parçalara ayırır.
# Orijinal paragraf yapısını korur ve wrap olmadan işlem yapar.
# detect_columns_and_reflow :
# Bilimsel dokümanlarda bulunan sütunlu yapıyı tespit eder.
# Sütunları birleştirerek metnin akışını düzeltir.
# map_scientific_sections :
# Bilimsel dokümanların ana bölümlerini (örneğin: Abstract, Giriş, Yöntemler, Bulgular, Tartışma, Sonuç) haritalar.
# Ek olarak, içerik tablosu, tablolar, çizelgeler gibi ek bölümleri de tespit eder.
# clean_text :
# Ham metni temizler:
# \r\n yerine \n
# Fazla boşluk ve satır aralıklarını azaltır.
# Paragrafların yapısını korur.
# shorten_title :
# Başlıkları belirli bir uzunluğa göre kısaltır.


# Özellikler
# Paragraf Koruma : Metin parçalama sırasında paragraf yapısını korur.
# Sütun Tespiti : Bilimsel dokümanlarda bulunan sütunlu yapıyı tespit eder ve akışı düzeltir.
# Bölüm Haritalama : Ana bölümleri ve ek bölümleri haritalar.
# Esneklik : Farklı boyutlarda parçalama ve özelleştirme seçenekleri sunar.
