import os
import pandas as pd
from config_module import config
from openai import OpenAI
from transformers import LlamaTokenizer

def split_text(text, chunk_size=256):
    """
    Metni, her biri chunk_size kelime içeren parçalara böler.
    
    Args:
        text (str): İşlenecek metin.
        chunk_size (int): Her parçanın içerdiği maksimum kelime sayısı (varsayılan: 256).
    
    Returns:
        list: Parçalara ayrılmış metin parçalarının listesi.
    """
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text):
    """
    OpenAI API kullanarak verilen metin için embedding oluşturur.
    Model: "text-embedding-ada-002"
    
    Args:
        text (str): Embedding oluşturulacak metin.
    
    Returns:
        list veya None: Oluşturulan embedding vektörü (örneğin, 1536 boyutlu liste) veya hata durumunda None.
    """
    try:
        client_instance = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client_instance.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        config.logger.error(f"Embedding oluşturulamadı: {e}")
        return None

def fine_tuning_preparation():
    """
    Fine-tuning için veri hazırlığı yapar.
    Eğer "fine_tuning_dataset.csv" mevcutsa, buradan metin verileri alınır,
    aksi halde örnek metinlerle çalışılır.
    
    Her metin, LLaMA Tokenizer kullanılarak tokenize edilir ve token sayıları hesaplanır.
    
    Returns:
        tuple: (summary, tokenized_data)
            - summary: Token sayıları ve istatistiklerini içeren özet metin.
            - tokenized_data: Her metnin tokenize edilmiş verilerinin listesi.
    """
    try:
        if os.path.exists("fine_tuning_dataset.csv"):
            df = pd.read_csv("fine_tuning_dataset.csv")
            texts = df["icerik"].tolist()
        else:
            texts = ["Örnek metin 1", "Örnek metin 2"]
        if not texts:
            config.logger.error("Fine-tuning için yeterli veri bulunamadı.")
            return None
        
        # LLaMA Tokenizer'ı yükle
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        tokenized_data = [tokenizer(text, truncation=True, max_length=512) for text in texts]
        token_counts = [len(item["input_ids"]) for item in tokenized_data]
        summary = (
            f"Toplam kayıt: {len(tokenized_data)}\n"
            f"Ortalama token sayısı: {sum(token_counts)/len(token_counts):.2f}\n"
            f"En fazla token: {max(token_counts)}\n"
            f"En az token: {min(token_counts)}\n"
        )
        return summary, tokenized_data
    except Exception as e:
        config.logger.error(f"Fine-tuning hazırlık hatası: {e}")
        return None
