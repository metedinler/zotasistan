# processingmanager.py

import os
import logging
from pdfprocessing import PDFProcessor
from embeddingmodule import EmbeddingManager
from filesavemodule import save_tables_to_chromadb, save_clean_text_to_chromadb, save_references_to_chromadb
from configmodule import Config

config = Config()
logger = logging.getLogger(__name__)

class ProcessingManager:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_manager = EmbeddingManager()

    def process_pdf(self, pdf_path):
        """
        PDF dosyasını işler ve metin, tablolar, referanslar gibi verileri çıkarır.

        Args:
            pdf_path (str): PDF dosyasının yolu.

        Returns:
            dict: Çıkarılan metin, tablolar, referanslar.
        """
        try:
            # PDF'den metin çıkar
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not text:
                logger.error(f"PDF'den metin çıkarma başarısız: {pdf_path}")
                return None

            # Metni temizle
            cleaned_text = self.pdf_processor.clean_advanced_text(text)

            # Tabloları çıkar
            tables = self.pdf_processor.extract_tables(cleaned_text)

            # Referansları çıkar
            references = self.pdf_processor.extract_references(cleaned_text)

            # Bilimsel bölümleri haritala
            sections = self.pdf_processor.map_scientific_sections(cleaned_text)

            # Metni reflow et
            reflowed_text = self.pdf_processor.reflow_columns(cleaned_text)

            # Embedding oluştur
            embeddings = self.embedding_manager.process_large_text(reflowed_text, pdf_path)

            return {
                "text": reflowed_text,
                "tables": tables,
                "references": references,
                "sections": sections,
                "embeddings": embeddings
            }
        except Exception as e:
            logger.error(f"PDF işlenirken hata oluştu: {e}")
            return None

    def save_data_to_chromadb(self, data, pdf_id):
        """
        Çıkarılan verileri ChromaDB'ye kaydeder.

        Args:
            data (dict): Kaydedilecek veriler (tablolar, temiz metin, referanslar, embeddingler).
            pdf_id (str): PDF dosya kimliği.
        """
        try:
            # Tabloları kaydeder
            if data["tables"]:
                save_tables_to_chromadb(data["tables"], pdf_id)

            # Temiz metni kaydeder
            if data["text"]:
                save_clean_text_to_chromadb(data["text"], pdf_id)

            # Referansları kaydeder
            if data["references"]:
                save_references_to_chromadb(data["references"], pdf_id)

            # Embeddingleri kaydeder
            if data["embeddings"]:
                self.embedding_manager.save_embeddings(data["embeddings"], f"{pdf_id}_embeddings.json")

            logger.info(f"✅ Veriler ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            logger.error(f"Veriler ChromaDB'ye kaydedilemedi: {e}")

# ProcessingManager sınıfını kullanmak için:
processing_manager = ProcessingManager()
pdf_path = "ornek.pdf"
sonuclar = processing_manager.process_pdf(pdf_path)
if sonuclar:
    pdf_id = os.path.basename(pdf_path).split(".")[0]
    processing_manager.save_data_to_chromadb(sonuclar, pdf_id)
else:
    print("PDF işlenirken hata oluştu.")
