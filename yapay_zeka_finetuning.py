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
