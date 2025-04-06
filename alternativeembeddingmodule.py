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
