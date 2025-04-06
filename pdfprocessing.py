# pdfprocessing.py

import fitz
import re
from configmodule import Config
from helpermodule import cleanadvancedtext

config = Config()

class PDFProcessor:
    def __init__(self):
        self.regex_section_patterns = config.REGEXSECTIONPATTERNS
        self.table_detection_patterns = config.TABLEDETECTIONPATTERNS

    def extract_text_from_pdf(self, pdf_path):
        """
        PDF dosyasından metin çıkarır.

        Args:
            pdf_path (str): PDF dosyasının yolu.

        Returns:
            str: Çıkarılan metin.
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            config.logger.error(f"PDF metin çıkarma hatası: {e}")
            return None

    def detect_columns(self, text):
        """
        Metindeki sütun yapısını tespit eder.

        Args:
            text (str): Metin.

        Returns:
            list: Sütunların metin parçaları.
        """
        # TODO: Sütun tespiti algoritması buraya eklenebilir
        return [text]  # Şimdilik tüm metni tek sütun olarak döndür

    def map_scientific_sections(self, text):
        """
        Metindeki bilimsel bölümleri haritalar.

        Args:
            text (str): Metin.

        Returns:
            dict: Bölüm adları ve metin parçaları.
        """
        sections = {}
        for pattern in self.regex_section_patterns:
            match = re.search(pattern, text)
            if match:
                section_name = match.group().strip()
                section_text = text[match.end():]
                sections[section_name] = section_text
                text = section_text
        return sections

    def reflow_columns(self, text):
        """
        Metni tek akışa dönüştürür.

        Args:
            text (str): Metin.

        Returns:
            str: Tek akışa dönüştürülmüş metin.
        """
        # HTML ve Markdown etiketlerini temizle
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        # Ekstra boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        return text

    def extract_tables(self, text):
        """
        Metinden tabloları çıkarır.

        Args:
            text (str): Metin.

        Returns:
            list: Tablo metin parçaları.
        """
        tables = []
        for pattern in self.table_detection_patterns:
            matches = re.findall(pattern, text)
            tables.extend(matches)
        return tables

    def process_pdf(self, pdf_path):
        """
        PDF dosyasını işler.

        Args:
            pdf_path (str): PDF dosyasının yolu.

        Returns:
            dict: Çıkarılan metin, bölümler, tablolar.
        """
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None

        # Metni temizle
        cleaned_text = cleanadvancedtext(text)

        # Bölümleri haritala
        sections = self.map_scientific_sections(cleaned_text)

        # Sütunları tespit et
        columns = self.detect_columns(cleaned_text)

        # Tabloları çıkar
        tables = self.extract_tables(cleaned_text)

        # Metni reflow et
        reflowed_text = self.reflow_columns(cleaned_text)

        return {
            "text": reflowed_text,
            "sections": sections,
            "columns": columns,
            "tables": tables
        }

# PDFProcessor sınıfını kullanmak için:
processor = PDFProcessor()
pdf_path = "ornek.pdf"
sonuclar = processor.process_pdf(pdf_path)
if sonuclar:
    print(sonuclar)
else:
    print("PDF işlenirken hata oluştu.")
