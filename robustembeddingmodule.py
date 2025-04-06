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
