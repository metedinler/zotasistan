# filesavemodule.py

import json
import os
from configmodule import Config
from chromadb import PersistentClient

config = Config()

class FileSaver:
    def __init__(self):
        self.chroma_client = PersistentClient(path="chroma_db")
        self.collections = {
            "clean_text": self.chroma_client.get_or_create_collection(name="clean_text"),
            "tables": self.chroma_client.get_or_create_collection(name="tables"),
            "references": self.chroma_client.get_or_create_collection(name="references"),
            "embeddings": self.chroma_client.get_or_create_collection(name="embeddings")
        }

    def save_clean_text_to_chromadb(self, text, pdf_id):
        """
        Temiz metni ChromaDB'ye kaydeder.

        Args:
            text (str): Kaydedilecek temiz metin.
            pdf_id (str): PDF dosya kimliği.
        """
        try:
            self.collections["clean_text"].add(
                documents=[text],
                metadatas=[{"pdf_id": pdf_id}],
                ids=[f"{pdf_id}_clean_text"]
            )
            config.logger.info(f"✅ Temiz metin ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            config.logger.error(f"Temiz metin ChromaDB'ye kaydedilemedi: {e}")

    def save_tables_to_chromadb(self, tables, pdf_id):
        """
        Tabloları ChromaDB'ye kaydeder.

        Args:
            tables (list): Kaydedilecek tablo metinlerinin listesi.
            pdf_id (str): PDF dosya kimliği.
        """
        try:
            self.collections["tables"].add(
                documents=tables,
                metadatas=[{"pdf_id": pdf_id}],
                ids=[f"{pdf_id}_table_{i}" for i in range(len(tables))]
            )
            config.logger.info(f"✅ Tablolar ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            config.logger.error(f"Tablolar ChromaDB'ye kaydedilemedi: {e}")

    def save_references_to_chromadb(self, references, pdf_id):
        """
        Kaynakçaları ChromaDB'ye kaydeder.

        Args:
            references (list): Kaydedilecek kaynakça metinlerinin listesi.
            pdf_id (str): PDF dosya kimliği.
        """
        try:
            self.collections["references"].add(
                documents=references,
                metadatas=[{"pdf_id": pdf_id}],
                ids=[f"{pdf_id}_reference_{i}" for i in range(len(references))]
            )
            config.logger.info(f"✅ Kaynakçalar ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            config.logger.error(f"Kaynakçalar ChromaDB'ye kaydedilemedi: {e}")

    def save_embeddings_to_chromadb(self, embeddings, pdf_id):
        """
        Embedding'leri ChromaDB'ye kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding vektörlerinin listesi.
            pdf_id (str): PDF dosya kimliği.
        """
        try:
            self.collections["embeddings"].add(
                vectors=embeddings,
                metadatas=[{"pdf_id": pdf_id}],
                ids=[f"{pdf_id}_embedding_{i}" for i in range(len(embeddings))]
            )
            config.logger.info(f"✅ Embedding'ler ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            config.logger.error(f"Embedding'ler ChromaDB'ye kaydedilemedi: {e}")

    def save_to_json(self, data, output_path):
        """
        Verileri JSON formatında kaydeder.

        Args:
            data (dict or list): Kaydedilecek veriler.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            config.logger.info(f"✅ Veriler JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Veriler JSON formatında kaydedilemedi: {e}")

# FileSaver sınıfını kullanmak için:
file_saver = FileSaver()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
file_saver.save_clean_text_to_chromadb(text, pdf_id)
