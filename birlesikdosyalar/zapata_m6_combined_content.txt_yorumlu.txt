# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results


# C:/Users/mete/Zotero/zotasistan/zapata_m6\birlestirilmis program\zzzapata_m6.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")
                    


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

# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations
    
# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")

# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

    # helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


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

# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")

    # veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")

  # veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()

# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()   
# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results

# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results

# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results


# C:/Users/mete/Zotero/zotasistan/zapata_m6\birlesikdosyalar\zapata_m6_combined_content.txt_yorumlu.py
# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results


# C:/Users/mete/Zotero/zotasistan/zapata_m6\birlestirilmis program\zzzapata_m6.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")
                    


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

# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations
    
# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")

# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

    # helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


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

# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")

    # veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")

  # veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()

# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()   
# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results

# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results



# C:/Users/mete/Zotero/zotasistan/zapata_m6\birlesikdosyalar\zapata_m6_combined_content.txt_yorumsuz.py
import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
config = Config()
class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }
    def get_sentence_transformer(self, model_key):
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")
        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None
    def embed_text_with_model(self, text, model_key):
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None
        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None
    def get_available_models(self):
        return list(self.model_list.keys())
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS
    def split_into_sentences(self, text):
        return sent_tokenize(text)
    def extract_citations_from_sentence(self, sentence):
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations
    def match_citation_with_references(self, citation, references):
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}
    def get_section_for_sentence(self, sentence, sections):
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"
    def map_citations(self, text, references, sections):
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings
    def save_citation_mapping(self, citation_mappings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager
class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
    def cluster_embeddings(self, embeddings, n_clusters=5):
        try:
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)
            pca = PCA(n_components=0.95)  
            pca_embeddings = pca.fit_transform(scaled_embeddings)
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")
            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None
    def visualize_clusters(self, embeddings, cluster_labels):
        try:
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
clustering_manager = ClusteringManager()
embeddings = []  
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")

import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb
class Config:
    def __init__(self):
        load_dotenv()
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.ensure_directories()
    def ensure_directories(self):
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def split_text(self, text, chunk_size=256, method="words"):
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
    def save_embeddings(self, embeddings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

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
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            config.logger.info(f"✅ Veriler JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Veriler JSON formatında kaydedilemedi: {e}")
file_saver = FileSaver()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
file_saver.save_clean_text_to_chromadb(text, pdf_id)

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config
config = Config()
class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)
    def select_pdf_file(self):
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")
    def start_processing(self):
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return
            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")
            results = self.processing_manager.process_pdf(self.pdf_path)
            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class HelperFunctions:
    def __init__(self):
        self.logger = config.logger
    def memory_usage(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    def shorten_title(self, title, max_length=50):
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title
    def clean_advanced_text(self, text):
        text = re.sub(r'<.*?>', '', text)  
        text = re.sub(r'\[.*?\]', '', text)  
        text = re.sub(r'\s+', ' ', text)  
        return text.strip()
    def fuzzy_match(self, text1, text2):
        return fuzz.partial_ratio(text1.lower(), text2.lower())
    def stack_load(self, stack_file):
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []
    def stack_update(self, stack_file, new_item):
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager
config = Config()
logger = logging.getLogger(__name__)
def process_pdf_workflow(pdf_path):
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return
        pdf_id = os.path.basename(pdf_path).split(".")[0]
        processing_manager.save_data_to_chromadb(results, pdf_id)
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")
            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")
    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")
def main():
    try:
        logger.info("Program başlatılıyor...")
        app = GUIManager()
        app.mainloop()
    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")
if __name__ == "__main__":
    main()

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
        return [text]  
    def map_scientific_sections(self, text):
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
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    def extract_tables(self, text):
        tables = []
        for pattern in self.table_detection_patterns:
            matches = re.findall(pattern, text)
            tables.extend(matches)
        return tables
    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        cleaned_text = cleanadvancedtext(text)
        sections = self.map_scientific_sections(cleaned_text)
        columns = self.detect_columns(cleaned_text)
        tables = self.extract_tables(cleaned_text)
        reflowed_text = self.reflow_columns(cleaned_text)
        return {
            "text": reflowed_text,
            "sections": sections,
            "columns": columns,
            "tables": tables
        }
processor = PDFProcessor()
pdf_path = "ornek.pdf"
sonuclar = processor.process_pdf(pdf_path)
if sonuclar:
    print(sonuclar)
else:
    print("PDF işlenirken hata oluştu.")

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
        try:
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not text:
                logger.error(f"PDF'den metin çıkarma başarısız: {pdf_path}")
                return None
            cleaned_text = self.pdf_processor.clean_advanced_text(text)
            tables = self.pdf_processor.extract_tables(cleaned_text)
            references = self.pdf_processor.extract_references(cleaned_text)
            sections = self.pdf_processor.map_scientific_sections(cleaned_text)
            reflowed_text = self.pdf_processor.reflow_columns(cleaned_text)
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
        try:
            if data["tables"]:
                save_tables_to_chromadb(data["tables"], pdf_id)
            if data["text"]:
                save_clean_text_to_chromadb(data["text"], pdf_id)
            if data["references"]:
                save_references_to_chromadb(data["references"], pdf_id)
            if data["embeddings"]:
                self.embedding_manager.save_embeddings(data["embeddings"], f"{pdf_id}_embeddings.json")
            logger.info(f"✅ Veriler ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            logger.error(f"Veriler ChromaDB'ye kaydedilemedi: {e}")
processing_manager = ProcessingManager()
pdf_path = "ornek.pdf"
sonuclar = processing_manager.process_pdf(pdf_path)
if sonuclar:
    pdf_id = os.path.basename(pdf_path).split(".")[0]
    processing_manager.save_data_to_chromadb(sonuclar, pdf_id)
else:
    print("PDF işlenirken hata oluştu.")

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
    def save_embeddings(self, embeddings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config
config = Config()
class VisualizationManager:
    def __init__(self):
        self.logger = config.logger
    def visualize_citation_graph(self, graph, output_path=None):
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")
    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")
    def save_graph_as_vosviewer(self, graph, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")
viz_manager = VisualizationManager()
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")
viz_manager.visualize_citation_graph(graph)
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")

import json
import networkx as nx
from configmodule import Config
config = Config()
class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger
    def create_citation_graph(self, citation_mappings):
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None
    def save_graph_to_file(self, graph, output_path):
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")
    def analyze_citation_graph(self, graph):
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None
    def save_analysis_to_json(self, analysis_results, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")
processor = CitationChainProcessor()
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]
graph = processor.create_citation_graph(citation_mappings)
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")
analysis_results = processor.analyze_citation_graph(graph)
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config
config = Config()
class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None
    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )
            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None
    def evaluate_model(self, trainer):
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None
if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)
    if tokenized_data:
        trainer = fine_tuner.train_model(tokenized_data)
        if trainer:
            evaluation_results = fine_tuner.evaluate_model(trainer)

import re
import requests
from configmodule import Config
config = Config()
class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }
    def get_metadata(self, item_key):
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None
    def analyze_references(self, reference_list):
        analysis_results = []
        for reference in reference_list:
            try:
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results
    def map_citations(self, text, references):
        citations = {}
        return citations
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")

import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb
class Config:
    def __init__(self):
        load_dotenv()
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.ensure_directories()
    def ensure_directories(self):
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")
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
        return [text]  
    def map_scientific_sections(self, text):
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
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    def extract_tables(self, text):
        tables = []
        for pattern in self.table_detection_patterns:
            matches = re.findall(pattern, text)
            tables.extend(matches)
        return tables
    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        cleaned_text = cleanadvancedtext(text)
        sections = self.map_scientific_sections(cleaned_text)
        columns = self.detect_columns(cleaned_text)
        tables = self.extract_tables(cleaned_text)
        reflowed_text = self.reflow_columns(cleaned_text)
        return {
            "text": reflowed_text,
            "sections": sections,
            "columns": columns,
            "tables": tables
        }
import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def split_text(self, text, chunk_size=256, method="words"):
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
config = Config()
class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }
    def get_sentence_transformer(self, model_key):
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")
        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None
    def embed_text_with_model(self, text, model_key):
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None
        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None
    def get_available_models(self):
        return list(self.model_list.keys())
import re
import requests
from configmodule import Config
config = Config()
class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }
    def get_metadata(self, item_key):
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None
    def analyze_references(self, reference_list):
        analysis_results = []
        for reference in reference_list:
            try:
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results
    def map_citations(self, text, references):
        citations = {}
        return citations
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")
import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
    def save_embeddings(self, embeddings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")
import re
import logging
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class HelperFunctions:
    def __init__(self):
        self.logger = config.logger
    def memory_usage(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    def shorten_title(self, title, max_length=50):
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title
    def clean_advanced_text(self, text):
        text = re.sub(r'<.*?>', '', text)  
        text = re.sub(r'\[.*?\]', '', text)  
        text = re.sub(r'\s+', ' ', text)  
        return text.strip()
    def fuzzy_match(self, text1, text2):
        return fuzz.partial_ratio(text1.lower(), text2.lower())
    def stack_load(self, stack_file):
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []
    def stack_update(self, stack_file, new_item):
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")
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
        return [text]  
    def map_scientific_sections(self, text):
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
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    def extract_tables(self, text):
        tables = []
        for pattern in self.table_detection_patterns:
            matches = re.findall(pattern, text)
            tables.extend(matches)
        return tables
    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        cleaned_text = cleanadvancedtext(text)
        sections = self.map_scientific_sections(cleaned_text)
        columns = self.detect_columns(cleaned_text)
        tables = self.extract_tables(cleaned_text)
        reflowed_text = self.reflow_columns(cleaned_text)
        return {
            "text": reflowed_text,
            "sections": sections,
            "columns": columns,
            "tables": tables
        }
processor = PDFProcessor()
pdf_path = "ornek.pdf"
sonuclar = processor.process_pdf(pdf_path)
if sonuclar:
    print(sonuclar)
else:
    print("PDF işlenirken hata oluştu.")
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
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            config.logger.info(f"✅ Veriler JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Veriler JSON formatında kaydedilemedi: {e}")
file_saver = FileSaver()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
file_saver.save_clean_text_to_chromadb(text, pdf_id)
import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS
    def split_into_sentences(self, text):
        return sent_tokenize(text)
    def extract_citations_from_sentence(self, sentence):
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations
    def match_citation_with_references(self, citation, references):
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}
    def get_section_for_sentence(self, sentence, sections):
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"
    def map_citations(self, text, references, sections):
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings
    def save_citation_mapping(self, citation_mappings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager
class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
    def cluster_embeddings(self, embeddings, n_clusters=5):
        try:
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)
            pca = PCA(n_components=0.95)  
            pca_embeddings = pca.fit_transform(scaled_embeddings)
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")
            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None
    def visualize_clusters(self, embeddings, cluster_labels):
        try:
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
clustering_manager = ClusteringManager()
embeddings = []  
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")
import json
import networkx as nx
from configmodule import Config
config = Config()
class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger
    def create_citation_graph(self, citation_mappings):
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None
    def save_graph_to_file(self, graph, output_path):
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")
    def analyze_citation_graph(self, graph):
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None
    def save_analysis_to_json(self, analysis_results, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")
processor = CitationChainProcessor()
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]
graph = processor.create_citation_graph(citation_mappings)
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")
analysis_results = processor.analyze_citation_graph(graph)
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")
import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config
config = Config()
class VisualizationManager:
    def __init__(self):
        self.logger = config.logger
    def visualize_citation_graph(self, graph, output_path=None):
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")
    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")
    def save_graph_as_vosviewer(self, graph, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")
viz_manager = VisualizationManager()
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")
viz_manager.visualize_citation_graph(graph)
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")
import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config
config = Config()
class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None
    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )
            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None
    def evaluate_model(self, trainer):
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None
if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)
    if tokenized_data:
        trainer = fine_tuner.train_model(tokenized_data)
        if trainer:
            evaluation_results = fine_tuner.evaluate_model(trainer)
import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config
config = Config()
class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)
    def select_pdf_file(self):
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")
    def start_processing(self):
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return
            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")
            results = self.processing_manager.process_pdf(self.pdf_path)
            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()
import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager
config = Config()
logger = logging.getLogger(__name__)
def process_pdf_workflow(pdf_path):
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return
        pdf_id = os.path.basename(pdf_path).split(".")[0]
        processing_manager.save_data_to_chromadb(results, pdf_id)
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")
            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")
    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")
def main():
    try:
        logger.info("Program başlatılıyor...")
        app = GUIManager()
        app.mainloop()
    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")
if __name__ == "__main__":
    main()   
import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
config = Config()
class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }
    def get_sentence_transformer(self, model_key):
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")
        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None
    def embed_text_with_model(self, text, model_key):
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None
        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None
    def get_available_models(self):
        return list(self.model_list.keys())
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS
    def split_into_sentences(self, text):
        return sent_tokenize(text)
    def extract_citations_from_sentence(self, sentence):
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations
    def match_citation_with_references(self, citation, references):
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}
    def get_section_for_sentence(self, sentence, sections):
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"
    def map_citations(self, text, references, sections):
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings
    def save_citation_mapping(self, citation_mappings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager
class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
    def cluster_embeddings(self, embeddings, n_clusters=5):
        try:
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)
            pca = PCA(n_components=0.95)  
            pca_embeddings = pca.fit_transform(scaled_embeddings)
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")
            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None
    def visualize_clusters(self, embeddings, cluster_labels):
        try:
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
clustering_manager = ClusteringManager()
embeddings = []  
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")

import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb
class Config:
    def __init__(self):
        load_dotenv()
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.ensure_directories()
    def ensure_directories(self):
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def split_text(self, text, chunk_size=256, method="words"):
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
    def save_embeddings(self, embeddings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

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
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            config.logger.info(f"✅ Veriler JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Veriler JSON formatında kaydedilemedi: {e}")
file_saver = FileSaver()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
file_saver.save_clean_text_to_chromadb(text, pdf_id)

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config
config = Config()
class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)
    def select_pdf_file(self):
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")
    def start_processing(self):
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return
            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")
            results = self.processing_manager.process_pdf(self.pdf_path)
            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class HelperFunctions:
    def __init__(self):
        self.logger = config.logger
    def memory_usage(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    def shorten_title(self, title, max_length=50):
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title
    def clean_advanced_text(self, text):
        text = re.sub(r'<.*?>', '', text)  
        text = re.sub(r'\[.*?\]', '', text)  
        text = re.sub(r'\s+', ' ', text)  
        return text.strip()
    def fuzzy_match(self, text1, text2):
        return fuzz.partial_ratio(text1.lower(), text2.lower())
    def stack_load(self, stack_file):
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []
    def stack_update(self, stack_file, new_item):
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager
config = Config()
logger = logging.getLogger(__name__)
def process_pdf_workflow(pdf_path):
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return
        pdf_id = os.path.basename(pdf_path).split(".")[0]
        processing_manager.save_data_to_chromadb(results, pdf_id)
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")
            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")
    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")
def main():
    try:
        logger.info("Program başlatılıyor...")
        app = GUIManager()
        app.mainloop()
    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")
if __name__ == "__main__":
    main()

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
        return [text]  
    def map_scientific_sections(self, text):
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
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    def extract_tables(self, text):
        tables = []
        for pattern in self.table_detection_patterns:
            matches = re.findall(pattern, text)
            tables.extend(matches)
        return tables
    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        cleaned_text = cleanadvancedtext(text)
        sections = self.map_scientific_sections(cleaned_text)
        columns = self.detect_columns(cleaned_text)
        tables = self.extract_tables(cleaned_text)
        reflowed_text = self.reflow_columns(cleaned_text)
        return {
            "text": reflowed_text,
            "sections": sections,
            "columns": columns,
            "tables": tables
        }
processor = PDFProcessor()
pdf_path = "ornek.pdf"
sonuclar = processor.process_pdf(pdf_path)
if sonuclar:
    print(sonuclar)
else:
    print("PDF işlenirken hata oluştu.")

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
        try:
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not text:
                logger.error(f"PDF'den metin çıkarma başarısız: {pdf_path}")
                return None
            cleaned_text = self.pdf_processor.clean_advanced_text(text)
            tables = self.pdf_processor.extract_tables(cleaned_text)
            references = self.pdf_processor.extract_references(cleaned_text)
            sections = self.pdf_processor.map_scientific_sections(cleaned_text)
            reflowed_text = self.pdf_processor.reflow_columns(cleaned_text)
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
        try:
            if data["tables"]:
                save_tables_to_chromadb(data["tables"], pdf_id)
            if data["text"]:
                save_clean_text_to_chromadb(data["text"], pdf_id)
            if data["references"]:
                save_references_to_chromadb(data["references"], pdf_id)
            if data["embeddings"]:
                self.embedding_manager.save_embeddings(data["embeddings"], f"{pdf_id}_embeddings.json")
            logger.info(f"✅ Veriler ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            logger.error(f"Veriler ChromaDB'ye kaydedilemedi: {e}")
processing_manager = ProcessingManager()
pdf_path = "ornek.pdf"
sonuclar = processing_manager.process_pdf(pdf_path)
if sonuclar:
    pdf_id = os.path.basename(pdf_path).split(".")[0]
    processing_manager.save_data_to_chromadb(sonuclar, pdf_id)
else:
    print("PDF işlenirken hata oluştu.")

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
    def save_embeddings(self, embeddings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config
config = Config()
class VisualizationManager:
    def __init__(self):
        self.logger = config.logger
    def visualize_citation_graph(self, graph, output_path=None):
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")
    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")
    def save_graph_as_vosviewer(self, graph, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")
viz_manager = VisualizationManager()
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")
viz_manager.visualize_citation_graph(graph)
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")

import json
import networkx as nx
from configmodule import Config
config = Config()
class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger
    def create_citation_graph(self, citation_mappings):
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None
    def save_graph_to_file(self, graph, output_path):
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")
    def analyze_citation_graph(self, graph):
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None
    def save_analysis_to_json(self, analysis_results, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")
processor = CitationChainProcessor()
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]
graph = processor.create_citation_graph(citation_mappings)
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")
analysis_results = processor.analyze_citation_graph(graph)
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config
config = Config()
class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None
    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )
            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None
    def evaluate_model(self, trainer):
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None
if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)
    if tokenized_data:
        trainer = fine_tuner.train_model(tokenized_data)
        if trainer:
            evaluation_results = fine_tuner.evaluate_model(trainer)

import re
import requests
from configmodule import Config
config = Config()
class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }
    def get_metadata(self, item_key):
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None
    def analyze_references(self, reference_list):
        analysis_results = []
        for reference in reference_list:
            try:
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results
    def map_citations(self, text, references):
        citations = {}
        return citations
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")
import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
config = Config()
class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }
    def get_sentence_transformer(self, model_key):
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")
        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None
    def embed_text_with_model(self, text, model_key):
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None
        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None
    def get_available_models(self):
        return list(self.model_list.keys())
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS
    def split_into_sentences(self, text):
        return sent_tokenize(text)
    def extract_citations_from_sentence(self, sentence):
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations
    def match_citation_with_references(self, citation, references):
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}
    def get_section_for_sentence(self, sentence, sections):
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"
    def map_citations(self, text, references, sections):
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings
    def save_citation_mapping(self, citation_mappings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager
class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
    def cluster_embeddings(self, embeddings, n_clusters=5):
        try:
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)
            pca = PCA(n_components=0.95)  
            pca_embeddings = pca.fit_transform(scaled_embeddings)
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")
            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None
    def visualize_clusters(self, embeddings, cluster_labels):
        try:
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")
clustering_manager = ClusteringManager()
embeddings = []  
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")

import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb
class Config:
    def __init__(self):
        load_dotenv()
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.ensure_directories()
    def ensure_directories(self):
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def split_text(self, text, chunk_size=256, method="words"):
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
    def save_embeddings(self, embeddings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

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
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            config.logger.info(f"✅ Veriler JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Veriler JSON formatında kaydedilemedi: {e}")
file_saver = FileSaver()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
file_saver.save_clean_text_to_chromadb(text, pdf_id)

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config
config = Config()
class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)
    def select_pdf_file(self):
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")
    def start_processing(self):
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return
            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")
            results = self.processing_manager.process_pdf(self.pdf_path)
            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config
config = Config()
class HelperFunctions:
    def __init__(self):
        self.logger = config.logger
    def memory_usage(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    def shorten_title(self, title, max_length=50):
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title
    def clean_advanced_text(self, text):
        text = re.sub(r'<.*?>', '', text)  
        text = re.sub(r'\[.*?\]', '', text)  
        text = re.sub(r'\s+', ' ', text)  
        return text.strip()
    def fuzzy_match(self, text1, text2):
        return fuzz.partial_ratio(text1.lower(), text2.lower())
    def stack_load(self, stack_file):
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []
    def stack_update(self, stack_file, new_item):
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager
config = Config()
logger = logging.getLogger(__name__)
def process_pdf_workflow(pdf_path):
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return
        pdf_id = os.path.basename(pdf_path).split(".")[0]
        processing_manager.save_data_to_chromadb(results, pdf_id)
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")
            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")
    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")
def main():
    try:
        logger.info("Program başlatılıyor...")
        app = GUIManager()
        app.mainloop()
    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")
if __name__ == "__main__":
    main()

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
        return [text]  
    def map_scientific_sections(self, text):
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
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    def extract_tables(self, text):
        tables = []
        for pattern in self.table_detection_patterns:
            matches = re.findall(pattern, text)
            tables.extend(matches)
        return tables
    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        cleaned_text = cleanadvancedtext(text)
        sections = self.map_scientific_sections(cleaned_text)
        columns = self.detect_columns(cleaned_text)
        tables = self.extract_tables(cleaned_text)
        reflowed_text = self.reflow_columns(cleaned_text)
        return {
            "text": reflowed_text,
            "sections": sections,
            "columns": columns,
            "tables": tables
        }
processor = PDFProcessor()
pdf_path = "ornek.pdf"
sonuclar = processor.process_pdf(pdf_path)
if sonuclar:
    print(sonuclar)
else:
    print("PDF işlenirken hata oluştu.")

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
        try:
            text = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if not text:
                logger.error(f"PDF'den metin çıkarma başarısız: {pdf_path}")
                return None
            cleaned_text = self.pdf_processor.clean_advanced_text(text)
            tables = self.pdf_processor.extract_tables(cleaned_text)
            references = self.pdf_processor.extract_references(cleaned_text)
            sections = self.pdf_processor.map_scientific_sections(cleaned_text)
            reflowed_text = self.pdf_processor.reflow_columns(cleaned_text)
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
        try:
            if data["tables"]:
                save_tables_to_chromadb(data["tables"], pdf_id)
            if data["text"]:
                save_clean_text_to_chromadb(data["text"], pdf_id)
            if data["references"]:
                save_references_to_chromadb(data["references"], pdf_id)
            if data["embeddings"]:
                self.embedding_manager.save_embeddings(data["embeddings"], f"{pdf_id}_embeddings.json")
            logger.info(f"✅ Veriler ChromaDB'ye kaydedildi: {pdf_id}")
        except Exception as e:
            logger.error(f"Veriler ChromaDB'ye kaydedilemedi: {e}")
processing_manager = ProcessingManager()
pdf_path = "ornek.pdf"
sonuclar = processing_manager.process_pdf(pdf_path)
if sonuclar:
    pdf_id = os.path.basename(pdf_path).split(".")[0]
    processing_manager.save_data_to_chromadb(sonuclar, pdf_id)
else:
    print("PDF işlenirken hata oluştu.")

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels
config = Config()
class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  
    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        if model_priority is None:
            model_priority = self.model_priority
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  
            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  
        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}
    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  
        return embeddings
    def save_embeddings(self, embeddings, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config
config = Config()
class VisualizationManager:
    def __init__(self):
        self.logger = config.logger
    def visualize_citation_graph(self, graph, output_path=None):
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")
    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")
    def save_graph_as_vosviewer(self, graph, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")
viz_manager = VisualizationManager()
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")
viz_manager.visualize_citation_graph(graph)
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")

import json
import networkx as nx
from configmodule import Config
config = Config()
class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger
    def create_citation_graph(self, citation_mappings):
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None
    def save_graph_to_file(self, graph, output_path):
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")
    def analyze_citation_graph(self, graph):
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None
    def save_analysis_to_json(self, analysis_results, output_path):
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")
processor = CitationChainProcessor()
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]
graph = processor.create_citation_graph(citation_mappings)
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")
analysis_results = processor.analyze_citation_graph(graph)
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config
config = Config()
class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None
    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )
            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None
    def evaluate_model(self, trainer):
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None
if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)
    if tokenized_data:
        trainer = fine_tuner.train_model(tokenized_data)
        if trainer:
            evaluation_results = fine_tuner.evaluate_model(trainer)

import re
import requests
from configmodule import Config
config = Config()
class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }
    def get_metadata(self, item_key):
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None
    def analyze_references(self, reference_list):
        analysis_results = []
        for reference in reference_list:
            try:
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results
    def map_citations(self, text, references):
        citations = {}
        return citations
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\birlestirilmis program\zzzapata_m6.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")
                    


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

# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations
    
# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")

# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")

    # helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


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

# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")

    # veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")

  # veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()

# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()   
# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results

# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results

# C:/Users/mete/Zotero/zotasistan/zapata_m6\alternativeembeddingmodule.py
# alternativeembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config

config = Config()

class AlternativeEmbeddingManager:
    def __init__(self):
        self.model_list = {
            "contrieverlarge": "facebook/contriever-large",
            "specterlarge": "allenai/specter-large",
            "allmpnet": "sentence-transformers/all-mpnet-base-v2",
            "paraphrasempnet": "sentence-transformers/paraphrase-mpnet-base-v2",
            "stsbrobertalarge": "sentence-transformers/stsb-roberta-large",
            "labse": "sentence-transformers/LaBSE",
            "universalsentenceencoder": "universal-sentence-encoder",
            "universalsentenceencoderlite": "universal-sentence-encoder-lite"
        }

    def get_sentence_transformer(self, model_key):
        """
        Belirtilen model anahtarına göre SentenceTransformer modelini yükler.

        Args:
            model_key (str): MODELLIST içinde yer alan model anahtarı.

        Returns:
            SentenceTransformer veya None: Yüklenen model, yüklenemezse None.
        """
        model_name = self.model_list.get(model_key)
        if not model_name:
            raise ValueError(f"Geçersiz model anahtarı: {model_key}")

        try:
            model = SentenceTransformer(model_name)
            config.logger.info(f"{model_key} modeli yüklendi: {model_name}")
            return model
        except Exception as e:
            config.logger.error(f"Model yükleme hatası: {e}")
            return None

    def embed_text_with_model(self, text, model_key):
        """
        Verilen metni belirtilen SentenceTransformer modeli ile embedding'e dönüştürür.

        Args:
            text (str): Embedding oluşturulacak metin.
            model_key (str): Kullanılacak model anahtarı.

        Returns:
            np.ndarray veya None: Oluşturulan embedding vektörü, hata durumunda None.
        """
        model = self.get_sentence_transformer(model_key)
        if not model:
            return None

        try:
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            config.logger.error(f"Embedding oluşturma hatası: {e}")
            return None

    def get_available_models(self):
        """
        Kullanılabilir SentenceTransformer modellerinin listesini döndürür.

        Returns:
            list: Kullanılabilir model anahtarlarının listesi.
        """
        return list(self.model_list.keys())

# AlternativeEmbeddingManager sınıfını kullanmak için:
alternative_embedding_manager = AlternativeEmbeddingManager()
model_key = "contrieverlarge"
text = "Örnek metin..."
embeddings = alternative_embedding_manager.embed_text_with_model(text, model_key)
if embeddings is not None:
    print(embeddings)
else:
    print("Embedding oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\citationmappingmodule.py
# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\clustering_module.py
# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\configmodule.py
# configmodule.py
import os
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import chromadb

class Config:
    def __init__(self):
        load_dotenv()

        # Temel Dizinler
        self.KAYNAK_DIZIN = os.getenv("KAYNAK_DIZIN", r"C:\Users\mete\Zotero\zotai")  # Ana dizin
        self.STORAGE_DIR = Path(os.getenv("STORAGE_DIR", r"C:\Users\mete\Zotero\storage"))  # PDF saklama dizini
        self.SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", r"C:\Users\mete\Zotero\zotai\egitimpdf"))  # İşlenmiş PDF dizini
        self.HEDEF_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizMetin"))
        self.TEMIZ_TABLO_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizTablo"))
        self.TEMIZ_KAYNAKCA_DIZIN = Path(os.path.join(self.KAYNAK_DIZIN, "TemizKaynakca"))
        self.PDF_DIR = self.SUCCESS_DIR / "pdfler"
        self.EMBEDDING_PARCA_DIR = self.SUCCESS_DIR / "embedingparca"

        # API Anahtarları
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ZOTERO_API_KEY = os.getenv('ZOTERO_API_KEY')
        self.ZOTERO_USER_ID = os.getenv('ZOTERO_USER_ID')
        self.ZOTERO_API_URL = f"https://api.zotero.org/users/{self.ZOTERO_USER_ID}/items"

        # Loglama Ayarları
        self.LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "islem_logu.json")
        self.HATA_LOG_DOSYASI = os.path.join(self.KAYNAK_DIZIN, "hata_logu.json")

        # Renkli Loglama Yapılandırması
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        file_handler = logging.FileHandler('pdf_processing.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # ChromaDB Yapılandırması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        #self.collection = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        #self.bib_collection = self.chroma_client.get_or_create_collection(name="pdf_bibliography")

        # Dizinleri oluştur
        self.ensure_directories()

    def ensure_directories(self):
        """Gerekli dizinlerin oluşturulması"""
        for directory in [self.PDF_DIR, self.EMBEDDING_PARCA_DIR, self.HEDEF_DIZIN, self.TEMIZ_TABLO_DIZIN, self.TEMIZ_KAYNAKCA_DIZIN]:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ {directory} dizini oluşturuldu.")
                except Exception as e:
                    self.logger.error(f"❌ {directory} dizini oluşturulamadı: {e}")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\embeddingmodule.py
# embeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class EmbeddingManager:
    def __init__(self):
        self.openai_client = None  # OpenAI API istemcisini buradan oluşturabilirsiniz
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def split_text(self, text, chunk_size=256, method="words"):
        """
        Metni belirlenen chunk_size'a göre parçalar.

        Args:
            text (str): Paralanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        if "text-embedding-ada-002" not in self.circuit_breaker or not self.circuit_breaker["text-embedding-ada-002"]:
            try:
                # OpenAI API ile embedding oluşturma
                response = self.openai_client.embeddings.create(input_text=text, model="text-embedding-ada-002")
                return {"embedding": response.data[0].embedding, "model": "text-embedding-ada-002"}
            except Exception as e:
                config.logger.warning(f"OpenAI modeli başarısız: {e}. Alternatif modellere geçiliyor.")
                self.circuit_breaker["text-embedding-ada-002"] = True

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embedtextwithmodel(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# EmbeddingManager sınıfını kullanmak için:
embedding_manager = EmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\filesavemodule.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\guimodule.py
# guimodule.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
from processingmanager import ProcessingManager
from configmodule import Config

config = Config()

class GUIManager(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere ayarları
        self.title("PDF İşleme Aracı")
        self.geometry("800x600")
        self.processing_manager = ProcessingManager()

        # Tema ayarları
        theme = config.GUITHEME or "light"
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")

        # Ana çerçeve
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Başlık
        self.title_label = ctk.CTkLabel(self.main_frame, text="PDF İşleme Aracı", font=("Arial", 24))
        self.title_label.pack(pady=10)

        # PDF seçme düğmesi
        self.select_pdf_button = ctk.CTkButton(
            self.main_frame, text="PDF Seç", command=self.select_pdf_file
        )
        self.select_pdf_button.pack(pady=10)

        # İşlem başlatma düğmesi
        self.start_process_button = ctk.CTkButton(
            self.main_frame, text="İşlemi Başlat", command=self.start_processing, state="disabled"
        )
        self.start_process_button.pack(pady=10)

        # Durum etiketi
        self.status_label = ctk.CTkLabel(self.main_frame, text="Durum: Bekleniyor...", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def select_pdf_file(self):
        """
        PDF dosyasını seçmek için bir dosya seçici açar.
        """
        file_path = filedialog.askopenfilename(
            title="PDF Dosyası Seç",
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if file_path:
            self.pdf_path = file_path
            self.status_label.configure(text=f"Seçilen Dosya: {file_path}")
            self.start_process_button.configure(state="normal")
            config.logger.info(f"PDF dosyası seçildi: {file_path}")
        else:
            messagebox.showwarning("Uyarı", "Bir PDF dosyası seçmediniz.")

    def start_processing(self):
        """
        Seçilen PDF dosyasını işler ve sonuçları kaydeder.
        """
        try:
            if not hasattr(self, "pdf_path"):
                messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin.")
                return

            self.status_label.configure(text="İşlem başlatılıyor...")
            config.logger.info(f"İşlem başlatıldı: {self.pdf_path}")

            # PDF işleme işlemi
            results = self.processing_manager.process_pdf(self.pdf_path)

            if results:
                pdf_id = self.pdf_path.split("/")[-1].split(".")[0]
                self.processing_manager.save_data_to_chromadb(results, pdf_id)
                messagebox.showinfo("Başarılı", "PDF başarıyla işlendi ve veriler kaydedildi.")
                config.logger.info(f"İşlem tamamlandı: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem tamamlandı.")
            else:
                messagebox.showerror("Hata", "PDF işlenirken bir hata oluştu.")
                config.logger.error(f"PDF işlenemedi: {self.pdf_path}")
                self.status_label.configure(text="Durum: İşlem başarısız.")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
            config.logger.error(f"İşlem sırasında hata oluştu: {e}")

# GUI'yi çalıştırmak için:
if __name__ == "__main__":
    app = GUIManager()
    app.mainloop()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\helpermodule.py
# helpermodule.py

import re
import logging
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class HelperFunctions:
    def __init__(self):
        self.logger = config.logger

    def memory_usage(self):
        """
        Bellek kullanımını MB cinsinden ölçer.

        Returns:
            float: Bellek kullanımı (MB).
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def shorten_title(self, title, max_length=50):
        """
        Uzun başlıkları belirtilen karakter sayısına indirger.

        Args:
            title (str): Başlık metni.
            max_length (int): Maksimum karakter sayısı. Varsayılan 50.

        Returns:
            str: Kısaltılmış başlık.
        """
        if len(title) > max_length:
            return title[:max_length - 3] + "..."
        return title

    def clean_advanced_text(self, text):
        """
        Metni temizler; HTML/Mardown etiketlerini, sayfa başı/sonu ifadelerini ve ekstra boşlukları kaldırır.

        Args:
            text (str): Temizlenecek metin.

        Returns:
            str: Temizlenmiş metin.
        """
        text = re.sub(r'<.*?>', '', text)  # HTML etiketleri temizle
        text = re.sub(r'\[.*?\]', '', text)  # Markdown etiketleri temizle
        text = re.sub(r'\s+', ' ', text)  # Ekstra boşlukları temizle
        return text.strip()

    def fuzzy_match(self, text1, text2):
        """
        İki metin arasındaki benzerlik oranını RapidFuzz kullanarak hesaplar.

        Args:
            text1 (str): İlk metin.
            text2 (str): İkinci metin.

        Returns:
            float: Benzerlik oranı (0-100).
        """
        return fuzz.partial_ratio(text1.lower(), text2.lower())

    def stack_load(self, stack_file):
        """
        Stack dosyasını yükler.

        Args:
            stack_file (str): Stack dosyasının yolu.

        Returns:
            list: Stack dosyasındaki işlemlerin listesi.
        """
        try:
            with open(stack_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Stack dosyası yüklenemedi: {e}")
            return []

    def stack_update(self, stack_file, new_item):
        """
        Stack dosyasını günceller.

        Args:
            stack_file (str): Stack dosyasının yolu.
            new_item (dict): Eklenmek istenen yeni işlem.
        """
        try:
            stack_data = self.stack_load(stack_file)
            stack_data.append(new_item)
            with open(stack_file, "w") as f:
                json.dump(stack_data, f)
            self.logger.info(f"Stack dosyası güncellendi: {stack_file}")
        except Exception as e:
            self.logger.error(f"Stack dosyası güncellenemedi: {e}")

# HelperFunctions sınıfını kullanmak için:
helper_functions = HelperFunctions()
text = "Örnek metin..."
cleaned_text = helper_functions.clean_advanced_text(text)
if cleaned_text:
    print(cleaned_text)
else:
    print("Metin temizlenemedi.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\main.py
# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()


# C:/Users/mete/Zotero/zotasistan/zapata_m6\pdfprocessing.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\processingmanager.py
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


# C:/Users/mete/Zotero/zotasistan/zapata_m6\robustembeddingmodule.py
# robustembeddingmodule.py

import numpy as np
from sentencetransformers import SentenceTransformer
from configmodule import Config
from alternativeembeddingmodule import getsentencetransformer, embedtextwithmodel, getavailablemodels

config = Config()

class RobustEmbeddingManager:
    def __init__(self):
        self.model_priority = ["contrieverlarge", "specterlarge", "allmpnet"]
        self.circuit_breaker = {}  # Hata durumunda modeli devre dışı bırakmak için
        self.max_retries = 3
        self.backoff_factor = 1.5
        self.rate_limit_delay = 1  # API rate limit koruması için sabit gecikme

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.5):
        """
        Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.

        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parçası sayısı.
            model_priority (list): Kullanılacak model sırası. Varsayılan olarak DEFAULTMODELPRIORITY.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.

        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker aktif olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    model = getsentencetransformer(model_key)
                    if model:
                        embedding = model.encode(text)
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor * attempt
                    config.logger.error(f"{model_key} ile embedding başarısız! Deneme {attempt}/{max_retries}. Hata: {e}")
                    time.sleep(wait_time)
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        config.logger.critical(f"Embedding işlemi tamamen başarısız oldu! PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks}")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.

        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı. Varsayılan 256.
            method (str): Parçalama yöntemi. "words" kelime bazında, "paragraphs" paragraf bazında parçalama.

        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.robust_embed_text, chunk, pdf_id, i, total_chunks) for i, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(self.rate_limit_delay)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        Embedding verilerini JSON formatında kaydeder.

        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"Embedding verileri kaydedilemedi: {e}")

# RobustEmbeddingManager sınıfını kullanmak için:
robust_embedding_manager = RobustEmbeddingManager()
text = "Örnek metin..."
pdf_id = "örnek_pdf_id"
embeddings = robust_embedding_manager.process_large_text(text, pdf_id)
if embeddings:
    output_path = "embeddings.json"
    robust_embedding_manager.save_embeddings(embeddings, output_path)
else:
    print("Embedding işlemi başarısız.")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_gorsellestirme.py
# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\veri_isleme.py
# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")


# C:/Users/mete/Zotero/zotasistan/zapata_m6\yapay_zeka_finetuning.py
# yapay_zeka_finetuning.py

import os
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from configmodule import Config

config = Config()

class FineTuningManager:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Fine-tuning işlemleri için gerekli olan modeli ve tokenizer'ı yükler.

        Args:
            model_name (str): Kullanılacak önceden eğitilmiş modelin adı.
            num_labels (int): Çıkış katmanındaki etiket sayısı.
        """
        self.logger = config.logger
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_data(self, dataset_path, text_column="text", label_column="label"):
        """
        Veriyi işler ve model için uygun hale getirir.

        Args:
            dataset_path (str): Veri kümesinin yolu.
            text_column (str): Metin sütununun adı.
            label_column (str): Etiket sütununun adı.

        Returns:
            DatasetDict: Eğitim ve doğrulama için işlenmiş veri kümesi.
        """
        try:
            raw_datasets = load_dataset("csv", data_files={"train": os.path.join(dataset_path, "train.csv"),
                                                           "validation": os.path.join(dataset_path, "validation.csv")})
            
            def tokenize_function(examples):
                return self.tokenizer(examples[text_column], truncation=True, padding=True)

            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
            self.logger.info("✅ Veri başarıyla işlenip tokenize edildi.")
            return tokenized_datasets
        except Exception as e:
            self.logger.error(f"Veri işleme sırasında hata oluştu: {e}")
            return None

    def train_model(self, tokenized_datasets, output_dir="fine_tuned_model", batch_size=16, epochs=3):
        """
        Modeli verilen veri kümesiyle eğitir.

        Args:
            tokenized_datasets (DatasetDict): İşlenmiş veri kümesi.
            output_dir (str): Eğitilmiş modelin kaydedileceği dizin.
            batch_size (int): Mini-batch boyutu.
            epochs (int): Eğitim döngüsü sayısı.

        Returns:
            Trainer: Eğitim işlemini gerçekleştiren Trainer nesnesi.
        """
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_dir=os.path.join(output_dir, "logs"),
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                tokenizer=self.tokenizer
            )

            trainer.train()
            self.logger.info("✅ Model başarıyla eğitildi.")
            return trainer
        except Exception as e:
            self.logger.error(f"Model eğitimi sırasında hata oluştu: {e}")
            return None

    def evaluate_model(self, trainer):
        """
        Eğitilmiş modeli doğrulama veri kümesi üzerinde değerlendirir.

        Args:
            trainer (Trainer): Eğitilmiş Trainer nesnesi.

        Returns:
            dict: Değerlendirme sonuçları.
        """
        try:
            evaluation_results = trainer.evaluate()
            self.logger.info(f"✅ Model değerlendirme sonuçları: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Model değerlendirme sırasında hata oluştu: {e}")
            return None

# FineTuningManager sınıfını kullanmak için örnek:

if __name__ == "__main__":
    fine_tuner = FineTuningManager(model_name="bert-base-uncased", num_labels=2)

    # Veri kümesini işleyin
    dataset_path = "datasets"
    tokenized_data = fine_tuner.preprocess_data(dataset_path)

    if tokenized_data:
        # Modeli eğitin
        trainer = fine_tuner.train_model(tokenized_data)

        if trainer:
            # Modeli değerlendirin
            evaluation_results = fine_tuner.evaluate_model(trainer)


# C:/Users/mete/Zotero/zotasistan/zapata_m6\zoteromodule.py
# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results

