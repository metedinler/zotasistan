# module_embedding_and_vectorization

import os
import json
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import psutil
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# ----------------------------
# Ortam Ayarları ve Loglama
# ----------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_DIR = os.getenv("EMBEDDING_DIR", "embeddings")
LOG_FILE = os.getenv("LOG_FILE", "embedding_log.json")

# Logging yapılandırması
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('embedding_processing.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# ----------------------------
# Yardımcı Fonksiyonlar
# ----------------------------

def memory_usage():
    """Anlık bellek kullanımını gösterir."""
    return f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"


def save_text_file(directory, filename, content):
    """Belirtilen dizine, verilen dosya adıyla içerik kaydeder."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as f:
        f.write(content)


def load_json(file_path):
    """JSON dosyasını yükler."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_json(file_path, data):
    """Veriyi JSON formatında kaydeder."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ----------------------------
# Embedding ve Vectorization Fonksiyonları
# ----------------------------

def embed_text_chunk(client, text_chunk):
    """
    OpenAI API kullanarak bir metin parçası için embedding oluşturur.
    Args:
        client (OpenAI): OpenAI istemcisi.
        text_chunk (str): Embedding oluşturulacak metin parçası.
    Returns:
        list: Embedding vektörü.
    """
    try:
        response = client.embeddings.create(
            input=text_chunk,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Embedding oluşturulamadı: {e}")
        return None


def embed_text_chunks(text_chunks):
    """
    Metin parçaları için embedding'ler oluşturur.
    Args:
        text_chunks (list): Embedding oluşturulacak metin parçaları listesi.
    Returns:
        list: Embedding vektörleri listesi.
    """
    embeddings = []
    client = OpenAI(api_key=OPENAI_API_KEY)
    for chunk in text_chunks:
        embedding = embed_text_chunk(client, chunk)
        if embedding:
            embeddings.append(embedding)
        else:
            embeddings.append([0.0] * 1536)  # Default embedding (sıfır vektör)
    return embeddings


def vectorize_text(texts):
    """
    Metinler için TF-IDF vektörlerini oluşturur.
    Args:
        texts (list): Vektörleştirilecek metinler listesi.
    Returns:
        sparse matrix: TF-IDF matrisi.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    return X


def calculate_similarity(query, texts):
    """
    Bir sorgu ile metinler arasındaki benzerlik skorlarını hesaplar.
    Args:
        query (str): Sorgu metni.
        texts (list): Benzerlik skoru hesaplanacak metinler listesi.
    Returns:
        list: Benzerlik skorları.
    """
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(X, query_vec).flatten()
    return similarities


# ----------------------------
# Ana İşlem Fonksiyonları
# ----------------------------

def process_embedding_for_file(file_data):
    """
    Bir dosya için embedding işlemi gerçekleştirir.
    Args:
        file_data (dict): Dosya bilgileri ve metin parçaları.
    Returns:
        dict: İşlenmiş dosya bilgisi ve embedding sonuçları.
    """
    try:
        title = file_data.get("title", "unknown")
        text_chunks = file_data.get("chunks", [])
        embeddings = embed_text_chunks(text_chunks)

        # Embedding'leri kaydet
        for idx, embedding in enumerate(embeddings):
            embed_filename = f"{title}_chunk_{idx}.embed.json"
            save_json(os.path.join(EMBEDDING_DIR, embed_filename), {"embedding": embedding})

        # Sonuçları döndür
        return {
            "title": title,
            "embeddings": embeddings,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ {title} için embedding işlemi başarısız: {e}")
        traceback.print_exc()
        return {
            "title": title,
            "embeddings": [],
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def batch_process_embeddings(file_list):
    """
    Birden fazla dosya için embedding işlemini paralel olarak gerçekleştirir.
    Args:
        file_list (list): Dosya bilgileri ve metin parçaları içeren liste.
    Returns:
        list: İşlenmiş dosya sonuçları.
    """
    results = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(process_embedding_for_file, file_data): file_data for file_data in file_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding İşlemleri"):
            result = future.result()
            results.append(result)
    return results


# ----------------------------
# Örnek Kullanım
# ----------------------------

if __name__ == "__main__":
    # Örnek dosya listesi
    example_files = [
        {
            "title": "ornek_dosya_1",
            "chunks": [
                "Bu bir örnek metin parçasıdır.",
                "Diğer bir örnek metin parçası."
            ]
        },
        {
            "title": "ornek_dosya_2",
            "chunks": [
                "Bilimsel dokümanlar için embedding oluşturuluyor.",
                "Metin analizi ve kümelenme işlemleri için kullanılır."
            ]
        }
    ]

    # Batch işlem başlat
    processed_results = batch_process_embeddings(example_files)

    # Sonuçları kaydet
    save_json(LOG_FILE, processed_results)
    print("✅ Tüm embedding işlemleri tamamlandı ve sonuçlar kaydedildi.")
    
    
#     Modül Açıklaması
# embed_text_chunk :
# OpenAI API kullanarak tek bir metin parçası için embedding oluşturur.
# Hata durumunda sıfır vektör döner.
# embed_text_chunks :
# Birden fazla metin parçası için embedding'ler oluşturur.
# Her bir metin parçası için embedding oluşturup listeye ekler.
# vectorize_text :
# Metinler için TF-IDF vektörlerini oluşturur.
# Kümeleme ve benzerlik hesaplamaları için kullanılabilir.
# calculate_similarity :
# Bir sorgu ile metinler arasındaki benzerlik skorlarını hesaplar.
# Cosine similarity kullanarak benzerlik skorlarını döndürür.
# process_embedding_for_file :
# Tek bir dosya için embedding işlemini gerçekleştirir.
# Embedding'leri diskte JSON formatında kaydeder.
# batch_process_embeddings :
# Birden fazla dosya için embedding işlemini paralel olarak gerçekleştirir.
# ProcessPoolExecutor kullanarak performansı artırır.
# Özellikler
# Paralel İşleme : Çok çekirdekli işlemci desteğiyle birden fazla dosya aynı anda işlenebilir.
# Hata Yönetimi : Her işlemde hata kontrolü yapılır ve log kaydı tutulur.
# TF-IDF ve Cosine Similarity : Metinler arasında benzerlik hesaplamaları için güçlü araçlar sağlar.
# OpenAI Entegrasyonu : Gelişmiş embedding modelleriyle yüksek kaliteli vektörler oluşturur.