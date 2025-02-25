import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını belirtilen yoldan yükle
load_dotenv("C:/Users/mete/Zotero/zotasistan/.env")

class Config:
    """
    Uygulama genelinde kullanılacak yapılandırma ayarlarını yönetir.
    - Ortam değişkenlerini (.env) yükler.
    - Gerekli dizinleri oluşturur.
    - Merkezi loglama sistemini yapılandırır.
    """
    # Base directory (proje kök dizini olarak kullanılabilir)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Dizin Yapıları (Varsayılan değerler .env üzerinden de ayarlanabilir)
    STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
    SUCCESS_DIR = Path(os.getenv("SUCCESS_DIR", BASE_DIR / "processed" / "success"))
    CITATIONS_DIR = Path(os.getenv("CITATIONS_DIR", BASE_DIR / "processed" / "citations"))
    TABLES_DIR = Path(os.getenv("TABLES_DIR", BASE_DIR / "processed" / "tables"))
    CLEAN_TEXT_DIR = Path(os.getenv("TEMIZMETIN_DIR", BASE_DIR / "processed" / "clean_text"))
    EMBEDDINGS_DIR = Path(os.getenv("EMBEDDING_PARCA_DIZIN", BASE_DIR / "processed" / "embeddings"))
    TEMP_DIR = Path(os.getenv("TEMP_DIR", BASE_DIR / "temp"))
    LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))

    # Gerekli dizinlerin oluşturulması
    for directory in [STORAGE_DIR, SUCCESS_DIR, CITATIONS_DIR, TABLES_DIR, CLEAN_TEXT_DIR, EMBEDDINGS_DIR, TEMP_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Log Dosyası
    LOG_FILE = LOG_DIR / "app.log"

    # Loglama yapılandırması
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.DEBUG,  # İsteğe bağlı olarak os.getenv("LOG_LEVEL", "DEBUG") kullanılabilir
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    logger = logging.getLogger(__name__)

    # PDF Metin Çıkarma Yöntemi (.env’den okunur)
    PDF_TEXT_EXTRACTION_METHOD = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()

    # Zotero API Ayarları
    ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
    ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
    ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE", "user")  # user veya group

    # OpenAI API ve Embedding Ayarları
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Alternatif Embedding Modelleri (env üzerinden ayarlanabilir)
    EMBEDDING_MODELS = {
        "contriever_large": os.getenv("CONTRIEVER_LARGE_MODEL", "facebook/contriever-large"),
        "specter_large": os.getenv("SPECTER_LARGE_MODEL", "allenai-specter-large"),
        "stsb_roberta_large": os.getenv("STSB_ROBERTA_LARGE_MODEL", "stsb-roberta-large"),
        "labse": os.getenv("LABSE_MODEL", "LaBSE"),
        "universal_sentence_encoder": os.getenv("USE_MODEL", "universal-sentence-encoder"),
    }

    # Chunk ve Büyük Dosya İşleme Ayarları
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
    LARGE_FILE_SPLIT_SIZE = int(os.getenv("LARGE_FILE_SPLIT_SIZE", 10000))

    # NLP ve Regex Ayarları (Bilimsel bölümler için)
    REGEX_SECTION_PATTERNS = {
        "Abstract": r"(?:^|\n)(Abstract|Özet)(?::)?\s*\n",
        "Introduction": r"(?:^|\n)(Introduction|Giriş)(?::)?\s*\n",
        "Methods": r"(?:^|\n)(Methods|Materials and Methods|Yöntemler|Metot)(?::)?\s*\n",
        "Results": r"(?:^|\n)(Results|Bulgular)(?::)?\s*\n",
        "Discussion": r"(?:^|\n)(Discussion|Tartışma)(?::)?\s*\n",
        "Conclusion": r"(?:^|\n)(Conclusion|Sonuç)(?::)?\s*\n"
    }

    # Tablo tespiti için regex desenleri
    TABLE_DETECTION_PATTERNS = [
        r"(?:^|\n)(Tablo|Table|Çizelge|Chart)\s*\d+",
        r"(?:^|\n)(Şekil|Figure|Fig)\s*\d+"
    ]

    # Paralel İşlem ve Hata Yönetimi Ayarları
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", 1.0))

    # GUI Ayarları
    GUI_THEME = os.getenv("GUI_THEME", "light")

config = Config()
#---------------------------------------config modulu sonu------------------------------------------------
import re
import requests
from config_module import config

class ZoteroEntegratoru:
    """
    Zotero API entegrasyonunu yöneten sınıf.
    
    Bu sınıf:
    - Zotero API'si aracılığıyla belirtilen item key'e göre bibliyometrik verileri çeker.
    - Verilen referans listesini analiz edip, basit regex kullanarak yazar isimlerini çıkarmayı sağlar.
    """
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
        self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    
    def meta_veri_al(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri çeker.
        
        Args:
            item_key (str): Zotero item key.
            
        Returns:
            dict or None: Çekilen bibliyometrik veriler; hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri çekildi: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası {response.status_code}: {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def referanslari_analiz_et(self, referans_listesi):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.
        
        Args:
            referans_listesi (list): Kaynakça referans metinlerinin listesi.
            
        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
                  Örnek: [{"orijinal": "Smith, 2020. Title...", "yazar": "Smith"}, ...]
        """
        analiz_sonuc = []
        try:
            for referans in referans_listesi:
                # Basit regex: Referans metninin başındaki ilk kelimeyi yazar olarak kabul ediyoruz.
                match = re.search(r'^([\w\.\-]+)', referans)
                yazar = match.group(1) if match else "Bilinmeyen"
                analiz_sonuc.append({
                    "orijinal": referans,
                    "yazar": yazar
                })
            config.logger.info("Referans analizi tamamlandı.")
            return analiz_sonuc
        except Exception as e:
            config.logger.error(f"Referans analizi hatası: {e}")
            return []
#---------------------------------------zotero modulu sonu------------------------------------------------
import os
import re
from config_module import config

def extract_text_from_pdf(pdf_path, method=None):
    """
    PDF'den ham metni çıkarır.
    Eğer method parametresi verilmezse, .env dosyasından "PDF_TEXT_EXTRACTION_METHOD" okunur (varsayılan: "pdfplumber").
    Eğer pdfplumber ile metin çıkarma hatası alınırsa, otomatik olarak pdfminer yöntemi devreye girer.
    
    Args:
        pdf_path (str or Path): PDF dosyasının yolu.
        method (str, optional): Kullanılacak metin çıkarma yöntemi ("pdfplumber" veya "pdfminer").
        
    Returns:
        str veya None: Çıkarılan metin; hata durumunda None.
    """
    if method is None:
        method = os.getenv("PDF_TEXT_EXTRACTION_METHOD", "pdfplumber").lower()

    text = None
    if method == "pdfplumber":
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n".join(pages_text)
            config.logger.info(f"✅ pdfplumber ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"❌ pdfplumber ile metin çıkarma hatası: {e}. pdfminer deneniyor.")
            return extract_text_from_pdf(pdf_path, method="pdfminer")
    elif method == "pdfminer":
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            config.logger.info(f"✅ pdfminer ile metin çıkarıldı: {pdf_path}")
        except Exception as e:
            config.logger.error(f"❌ pdfminer ile metin çıkarma hatası: {e}")
    else:
        config.logger.error("Geçersiz method belirtildi. 'pdfplumber' veya 'pdfminer' kullanılabilir.")
    return text

def detect_columns(text, min_gap=4):
    """
    Metindeki sütun yapısını tespit eder.
    Belirli bir boşluk sayısına göre (min_gap) metnin sütunlu olup olmadığını belirler.
    
    Args:
        text (str): İşlenecek metin.
        min_gap (int): Bir satırda sütunları ayırmak için gereken minimum boşluk sayısı.
    
    Returns:
        dict: Örnek: {'sutunlu': True} veya {'sutunlu': False}
    """
    lines = text.split('\n')
    column_line_count = sum(1 for line in lines if re.search(r' {' + str(min_gap) + r',}', line))
    return {'sutunlu': column_line_count > len(lines) * 0.2}

def map_scientific_sections_extended(text):
    """
    Bilimsel dokümanların bölümlerini haritalar.
    Örneğin: Abstract, Introduction, Methods, Results, Discussion, Conclusion,
    İçindekiler, Tablolar, Çizelgeler, Resimler/Figürler, İndeks.
    
    Args:
        text (str): İşlenecek ham metin.
        
    Returns:
        dict: Haritalanmış bölümler; her bölüm için "start", "end" ve "content" bilgileri.
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
    # Her bölüm için ilk eşleşmenin başlangıç indeksini alıyoruz.
    for section, pattern in {**section_patterns, **additional_patterns}.items():
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        sections_map[section] = matches[0].start() if matches else None

    # Sadece bulunan bölümleri ayıkla
    detected_sections = {sec: pos for sec, pos in sections_map.items() if pos is not None}
    sorted_sections = sorted(detected_sections.items(), key=lambda x: x[1])
    mapped_sections = {}
    for i, (section, start_idx) in enumerate(sorted_sections):
        end_idx = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        content = text[start_idx:end_idx].strip()
        mapped_sections[section] = {"start": start_idx, "end": end_idx, "content": content}
    # Ek olarak, sütun yapısı bilgisi ekleyelim
    column_info = detect_columns(text)
    mapped_sections["Column Structure"] = column_info

    # Eğer bazı bölümler bulunamazsa, onları None olarak ekleyelim
    for sec in list(section_patterns.keys()) + list(additional_patterns.keys()):
        if sec not in mapped_sections:
            mapped_sections[sec] = None

    return mapped_sections

def map_pdf_before_extraction(pdf_path, method='pdfplumber'):
    """
    PDF'den metin çıkarılmadan önce yapısal analiz yapar ve bilimsel bölümleri haritalar.
    İlk olarak, extract_text_from_pdf çağrılır; ardından elde edilen metin üzerinden
    map_scientific_sections_extended fonksiyonu ile bölümler tespit edilir.
    
    Args:
        pdf_path (str or Path): PDF dosyasının yolu.
        method (str): Kullanılacak metot ("pdfplumber" veya "pdfminer"). Varsayılan: "pdfplumber".
    
    Returns:
        dict veya None: Haritalanmış bölüm bilgileri; metin çıkarılamazsa None.
    """
    text = extract_text_from_pdf(pdf_path, method=method)
    if not text:
        config.logger.error("PDF'den metin çıkarılamadı; haritalama yapılamıyor.")
        return None
    return map_scientific_sections_extended(text)

def reflow_columns(text):
    """
    Sütunlu metni tek akışa dönüştürür.
    - HTML/Markdown etiketlerini temizler.
    - Sayfa başı/sonu bilgilerini (örn. "Page 1", "Sayfa 1") kaldırır.
    - Fazla boşlukları tek boşluk yapar.
    - Kırpılmış kelimelerdeki tire işaretlerini kaldırır.
    
    Args:
        text (str): İşlenecek metin.
    
    Returns:
        str: Temizlenmiş, tek akışa dönüştürülmüş metin.
    """
    # HTML etiketlerini kaldır
    text = re.sub(r"<[^>]+>", " ", text)
    # Markdown link yapılarını kaldır: [Link](url)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    # Sayfa bilgilerini temizle
    text = re.sub(r"(Page|Sayfa)\s*\d+", " ", text, flags=re.IGNORECASE)
    # Satır sonlarını boşlukla değiştir
    text = re.sub(r"\n", " ", text)
    # Fazla boşlukları tek boşluk yap
    text = re.sub(r"\s{2,}", " ", text)
    # Kırpılmış kelimelerdeki tireyi kaldır: "infor- mation" -> "information"
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text.strip()
#---------------------------------------pdf_processing modulu sonu------------------------------------------------
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from config_module import config
from alternative_embedding_module import get_sentence_transformer, embed_text_with_model, get_available_models

# Varsayılan ayarlar
DEFAULT_MODEL_PRIORITY = ["contriever_large", "specter_large", "all_mpnet", "paraphrase_mpnet"]
OPENAI_MODEL = "text-embedding-ada-002"
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
RATE_LIMIT_DELAY = 1  # API rate limit koruması için sabit gecikme

class EmbeddingManager:
    """
    📌 Embedding işlemlerini yöneten sınıf.
    - OpenAI API ve alternatif embedding modellerini destekler.
    - Retry mekanizması ve circuit breaker ile hata toleranslıdır.
    - Birden fazla model desteği ve parçalama özelliği içerir.
    """

    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model_priority = DEFAULT_MODEL_PRIORITY
        self.circuit_breaker = {}  # Circuit breaker durumunu takip etmek için

    def split_text(self, text, chunk_size=256, method="words"):
        """
        📌 Metni belirlenen chunk_size'a göre böler.
        
        Args:
            text (str): Parçalanacak metin.
            chunk_size (int): Her parça için maksimum kelime sayısı (varsayılan: 256).
            method (str): Bölme yöntemi; "words" kelime bazında, "paragraphs" ise paragraf bazında bölme.
            
        Returns:
            list: Parçalara ayrılmış metin parçalarının listesi.
        """
        if method == "paragraphs":
            paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
            return paragraphs
        else:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def robust_embed_text(self, text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR):
        """
        📌 Verilen metni embedding oluştururken hata toleranslı bir mekanizma kullanır.
        
        Args:
            text (str): Embedding oluşturulacak metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_index (int): İşlenen metin parçasının sırası.
            total_chunks (int): Toplam metin parça sayısı.
            model_priority (list, optional): Kullanılacak model sırası. Varsayılan olarak `DEFAULT_MODEL_PRIORITY`.
            max_retries (int): Her model için en fazla kaç kez tekrar deneneceği.
            backoff_factor (float): Hata alındığında bekleme süresini artıran katsayı.
        
        Returns:
            dict: Başarılı embedding vektörü ve kullanılan model bilgisi.
        """
        if model_priority is None:
            model_priority = self.model_priority

        # Öncelikle OpenAI API ile embedding oluşturmaya çalış
        if OPENAI_MODEL not in self.circuit_breaker or not self.circuit_breaker[OPENAI_MODEL]:
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=OPENAI_MODEL
                )
                return {"embedding": response.data[0].embedding, "model": OPENAI_MODEL}
            except Exception as e:
                config.logger.warning(f"⚠️ OpenAI modeli başarısız ({OPENAI_MODEL}), alternatif modellere geçiliyor. Hata: {e}")
                self.circuit_breaker[OPENAI_MODEL] = True  # Circuit breaker devreye girer

        # OpenAI başarısız olduysa, alternatif modellere geç
        for model_key in model_priority:
            if model_key in self.circuit_breaker and self.circuit_breaker[model_key]:
                continue  # Circuit breaker açık olan modeller atlanır

            for attempt in range(1, max_retries + 1):
                try:
                    embedding = embed_text_with_model(text, model_key)
                    if embedding:
                        return {"embedding": embedding, "model": model_key}
                except Exception as e:
                    wait_time = backoff_factor ** attempt
                    config.logger.error(f"❌ {model_key} ile embedding başarısız! ({attempt}/{max_retries}) Hata: {e}")
                    time.sleep(wait_time)  # Backoff delay
                    if attempt == max_retries:
                        self.circuit_breaker[model_key] = True  # Circuit breaker devreye girer

        # Tüm denemeler başarısız olursa None döndür
        config.logger.critical(f"🚨 Embedding işlemi tamamen başarısız oldu! (PDF: {pdf_id}, Chunk: {chunk_index}/{total_chunks})")
        return {"embedding": None, "model": "failed"}

    def process_large_text(self, text, pdf_id, chunk_size=256, method="words"):
        """
        📌 Büyük metinleri parçalara ayırarak her bir parça için embedding oluşturur.
        
        Args:
            text (str): İşlenecek büyük metin.
            pdf_id (str): PDF dosya kimliği.
            chunk_size (int): Her parça için maksimum kelime sayısı (varsayılan: 256).
            method (str): Bölme yöntemi; "words" kelime bazında, "paragraphs" ise paragraf bazında bölme.
        
        Returns:
            list: Oluşturulan embedding vektörlerinin listesi.
        """
        chunks = self.split_text(text, chunk_size, method)
        embeddings = []
        total_chunks = len(chunks)

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.robust_embed_text,
                    chunk,
                    pdf_id,
                    i,
                    total_chunks
                ): chunk for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                result = future.result()
                embeddings.append(result)
                time.sleep(RATE_LIMIT_DELAY)  # API rate limit koruması

        return embeddings

    def save_embeddings(self, embeddings, output_path):
        """
        📌 Embedding verilerini JSON formatında kaydeder.
        
        Args:
            embeddings (list): Kaydedilecek embedding verileri.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f, ensure_ascii=False, indent=4)
            config.logger.info(f"✅ Embedding verileri başarıyla kaydedildi: {output_path}")
        except Exception as e:
            config.logger.error(f"❌ Embedding verileri kaydedilemedi: {e}")
#---------------------------------------embedding modulu sonu------------------------------------------------
import numpy as np
from sentence_transformers import SentenceTransformer
from config_module import config

# Kullanılabilir alternatif embedding modelleri
MODEL_LIST = {
    "contriever_large": "facebook/contriever-large",
    "specter_large": "allenai-specter-large",
    "specter": "allenai/specter",
    "all_mpnet": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase_mpnet": "sentence-transformers/paraphrase-mpnet-base-v2",
    "stsb_roberta_large": "sentence-transformers/stsb-roberta-large",
    "labse": "sentence-transformers/LaBSE",
    "universal_sentence_encoder": "universal-sentence-encoder",  # TF Hub modeli veya benzeri
    "universal_sentence_encoder_lite": "universal-sentence-encoder-lite"
}

def get_sentence_transformer(model_key):
    """
    📌 Belirtilen model anahtarına göre SentenceTransformer modelini yükler.
    
    Args:
        model_key (str): MODEL_LIST içinde yer alan model anahtarı.
    
    Returns:
        SentenceTransformer veya None: Yüklenmiş model, yüklenemezse None.
    """
    model_name = MODEL_LIST.get(model_key)
    if not model_name:
        raise ValueError(f"❌ Geçersiz model anahtarı: {model_key}")
    try:
        model = SentenceTransformer(model_name)
        config.logger.info(f"✅ {model_key} modeli yüklendi (model adı: {model_name}).")
        return model
    except Exception as e:
        config.logger.error(f"❌ Model yüklenirken hata oluştu ({model_key}): {e}")
        return None

def embed_text_with_model(text, model_key):
    """
    📌 Belirtilen alternatif embedding modeli ile metin embedding'i oluşturur.
    
    Args:
        text (str): Embedding oluşturulacak metin.
        model_key (str): Kullanılacak model anahtarı (MODEL_LIST içinde).
    
    Returns:
        list veya None: Embedding vektörü (örneğin, liste formatında) veya hata durumunda None.
    """
    model = get_sentence_transformer(model_key)
    if not model:
        config.logger.error(f"❌ Model {model_key} yüklenemedi, embedding oluşturulamıyor.")
        return None
    try:
        embedding = model.encode(text)
        config.logger.info(f"✅ Embedding oluşturuldu ({model_key}).")
        return embedding.tolist()
    except Exception as e:
        config.logger.error(f"❌ Embedding oluşturulamadı ({model_key}): {e}")
        return None

def get_available_models():
    """
    📌 Kullanılabilir alternatif embedding modellerinin anahtarlarını döndürür.
    
    Returns:
        list: MODEL_LIST içindeki model anahtarlarının listesi.
    """
    return list(MODEL_LIST.keys())
#---------------------------------------alternative_embedding modulu sonu------------------------------------------------
import time
from config_module import config
from alternative_embedding_module import embed_text_with_model, get_available_models

def robust_embed_text(text, pdf_id, chunk_index, total_chunks, model_priority=None, max_retries=3, backoff_factor=1.0):
    """
    Robust embedding oluşturma fonksiyonu.
    
    Bu fonksiyon, verilen metin için öncelikle model_priority listesinde yer alan modelleri
    sırasıyla dener. Her model için maksimum max_retries deneme yapılır; başarısız olursa
    exponential backoff uygulanır ve sonraki modele geçilir.
    
    Args:
        text (str): Embedding oluşturulacak metin.
        pdf_id (str): PDF dosyasının temel ID'si.
        chunk_index (int): İşlenen chunk numarası (0 tabanlı).
        total_chunks (int): Toplam chunk sayısı.
        model_priority (list, optional): Denenecek modellerin listesi. Varsayılan olarak, tüm alternatif modeller.
        max_retries (int, optional): Her model için maksimum deneme sayısı (varsayılan: 3).
        backoff_factor (float, optional): Denemeler arasında kullanılacak exponential backoff katsayısı (varsayılan: 1.0).
    
    Returns:
        list veya None: Başarılı ise embedding vektörü (liste formatında), aksi halde None.
    """
    if model_priority is None:
        model_priority = get_available_models()
    
    for model in model_priority:
        config.logger.info(f"Denenecek model: {model} (PDF: {pdf_id}, Chunk: {chunk_index+1}/{total_chunks})")
        attempt = 0
        while attempt < max_retries:
            try:
                embedding = embed_text_with_model(text, model)
                if embedding is not None:
                    config.logger.info(f"✅ Model {model} başarılı: PDF {pdf_id}, Chunk {chunk_index+1}/{total_chunks}")
                    # Burada, istenirse yerel cache (örn. embedding_cache.json) güncellenebilir.
                    return embedding
            except Exception as e:
                config.logger.error(f"❌ Hata: Model {model}, Deneme {attempt+1}/{max_retries}: {e}")
            attempt += 1
            sleep_time = backoff_factor * (2 ** attempt)
            config.logger.info(f"⏳ Bekleniyor: {sleep_time} saniye")
            time.sleep(sleep_time)
        config.logger.warning(f"⚠️ Model {model} için maksimum deneme sayısına ulaşıldı, sonraki modele geçiliyor.")
    
    config.logger.error(f"❌ Tüm modeller başarısız oldu: PDF {pdf_id}, Chunk {chunk_index+1}/{total_chunks}")
    return None
#---------------------------------------robust_embedding modulu sonu------------------------------------------------

import os
import re
import json
import psutil
import threading
from rapidfuzz import fuzz
from config_module import config

def memory_usage():
    """
    Bellek kullanımını MB cinsinden döndürür.

    Returns:
        float: Mevcut sürecin kullanılan bellek miktarı (MB cinsinden).
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)

def shorten_title(title, max_length=80):
    """
    Uzun başlıkları belirtilen maksimum uzunlukta kısaltır.

    Args:
        title (str): Kısaltılacak başlık.
        max_length (int): Maksimum karakter sayısı (varsayılan: 80).

    Returns:
        str: Kısaltılmış başlık; eğer başlık max_length'ten kısa ise değişiklik yapmaz.
    """
    if len(title) <= max_length:
        return title
    else:
        return title[:max_length] + "..."

def clean_advanced_text(text):
    """
    Gelişmiş metin temizleme fonksiyonu:
    - HTML etiketlerini kaldırır.
    - Markdown link yapılarını temizler.
    - Sayfa başı/sonu ifadelerini (örn. "Page 1", "Sayfa 1") kaldırır.
    - Fazla boşlukları tek boşluk haline getirir.
    - Kırpılmış kelimelerdeki tire işaretlerini düzeltir.

    Args:
        text (str): Temizlenecek ham metin.

    Returns:
        str: Temizlenmiş metin.
    """
    # HTML etiketlerini kaldır
    text = re.sub(r"<[^>]+>", " ", text)
    # Markdown link yapılarını kaldır (örneğin, [Link](url))
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    # Sayfa başı/sonu ifadelerini kaldır (örn. "Page 1" veya "Sayfa 1")
    text = re.sub(r"(Page|Sayfa)\s*\d+", " ", text, flags=re.IGNORECASE)
    # Fazla boşlukları tek boşluk yap
    text = re.sub(r"\s{2,}", " ", text)
    # Kırpılmış kelimelerdeki tireyi kaldır (örneğin, "infor- mation" -> "information")
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    return text.strip()

def fuzzy_match(text1, text2):
    """
    RapidFuzz kütüphanesini kullanarak iki metin arasındaki benzerlik oranını hesaplar.

    Args:
        text1 (str): İlk metin.
        text2 (str): İkinci metin.

    Returns:
        float: Benzerlik oranı (0-100 arası değer).
    """
    return fuzz.ratio(text1, text2)

# --- Stack Yönetimi (İşlem Listesi) ---

# Stack dosyasının yolu config üzerinden belirleniyor.
STACK_DOSYASI = config.STACK_DOSYASI
stack_lock = threading.Lock()

def stack_yukle():
    """
    Stack dosyasını (işlem listesi) yükler.

    Returns:
        list: İşlenen dosyaların listesini içeren liste.
    """
    if os.path.exists(STACK_DOSYASI):
        try:
            with open(STACK_DOSYASI, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            config.logger.error("❌ Stack dosyası bozuk, sıfırlanıyor.")
    return []

def stack_guncelle(dosya_adi, islem):
    """
    Stack dosyasını günceller. İşlem "ekle" ise dosya adını ekler; "sil" ise çıkarır.

    Args:
        dosya_adi (str): Güncellenecek dosya adı.
        islem (str): "ekle" veya "sil".
    """
    with stack_lock:
        stack = stack_yukle()
        if islem == "ekle":
            if dosya_adi not in stack:
                stack.append(dosya_adi)
        elif islem == "sil":
            if dosya_adi in stack:
                stack.remove(dosya_adi)
        with open(STACK_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(stack, f, ensure_ascii=False, indent=2)

#---------------------------------------helper modulu sonu------------------------------------------------

import os
import re
import threading
from datetime import datetime
from pathlib import Path
import multiprocessing
from tqdm import tqdm
import chromadb
from openai import OpenAI

from config_module import config
from zotero_module import dokuman_id_al, fetch_zotero_metadata
from pdf_processing import (
    extract_text_from_pdf,
    reflow_columns,
    map_scientific_sections_extended,
    map_pdf_before_extraction,
    detect_columns,
    extract_references_enhanced
)
from embedding_module import embed_text
from helper_module import stack_yukle, stack_guncelle, shorten_title

class IslemYoneticisi:
    """
    PDF/TXT dosyalarını işleme sürecini yöneten ana sınıf.
    
    İş Akışı:
      1. Dosya tipi (.pdf veya .txt) belirlenir.
      2. İşleme başlamadan önce, dosya adı stack'e eklenir.
      3. PDF ise: 
         - İlk olarak map_pdf_before_extraction ile yapısal haritalama yapılır.
         - extract_text_from_pdf ile ham metin çıkarılır.
         TXT ise: dosya doğrudan okunur ve map_scientific_sections_extended ile bölümler haritalanır.
      4. Çıkarılan metin, reflow_columns ile tek akışa dönüştürülür.
      5. Bilimsel bölümler map_scientific_sections_extended ile yeniden tespit edilir.
      6. Sütun yapısı detect_columns ile belirlenir.
      7. Kaynakça extract_references_enhanced ile çıkarılır.
      8. Zotero entegrasyonu: dokuman_id_al ile temel dosya ID'si alınır, shorten_title ile kısaltılır, fetch_zotero_metadata ile bibliyometrik veriler çekilir.
      9. Temiz metin üzerinden embed_text fonksiyonu ile embedding oluşturulur.
     10. Tüm bu bilgiler, bir sonuç sözlüğünde toplanır.
     11. İşlem sonunda, dosya adı stack'ten kaldırılır ve sayaçlar güncellenir.
    
    Döndürdüğü sonuç sözlüğü, işlenmiş dosyaya ait tüm bilgileri içerir.
    """
    def __init__(self):
        self.stack_lock = threading.Lock()
        self.kume_sonuclari = []  # İşlemden elde edilen sonuçların saklandığı liste
        self.sayaçlar = {
            'toplam': 0,
            'başarılı': 0,
            'hata': 0
        }
        # ChromaDB bağlantısı ve koleksiyonlarının oluşturulması
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.koleksiyon = self.chroma_client.get_or_create_collection(name="pdf_embeddings")
        self.zotero_koleksiyon = self.chroma_client.get_or_create_collection(name="zotero_meta")
        self.secili_dosya = None

    def pdf_txt_isle(self, dosya_yolu):
        try:
            # İşleme başlamadan önce stack güncellemesi
            self.stack_guncelle(dosya_yolu.name, "ekle")
            config.logger.info(f"{dosya_yolu.name} işleme başladı.")
            
            ext = dosya_yolu.suffix.lower()
            if ext == ".pdf":
                # PDF için: önce yapısal haritalama, sonra metin çıkarımı
                harita = map_pdf_before_extraction(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
                ham_metin = extract_text_from_pdf(dosya_yolu, method=config.PDF_TEXT_EXTRACTION_METHOD)
            elif ext == ".txt":
                with open(dosya_yolu, "r", encoding="utf-8") as f:
                    ham_metin = f.read()
                harita = map_scientific_sections_extended(ham_metin)
            else:
                config.logger.error(f"Desteklenmeyen dosya uzantısı: {dosya_yolu}")
                return None

            if not ham_metin:
                raise ValueError("Ham metin çıkarılamadı.")

            # Metni tek akışa dönüştürme (reflow)
            temiz_metin = reflow_columns(ham_metin)

            # Bilimsel bölümlerin haritalanması
            bolum_haritasi = map_scientific_sections_extended(ham_metin)

            # Sütun yapısı tespiti
            sutun_bilgisi = detect_columns(ham_metin)

            # Kaynakça çıkarımı (hata yönetimiyle)
            try:
                references = extract_references_enhanced(ham_metin)
            except Exception as e:
                config.logger.error(f"Kaynakça çıkarım hatası: {e}")
                references = []

            # Zotero entegrasyonu: Dosya temel ID'si alınır ve kısaltılır
            dosya_id = dokuman_id_al(dosya_yolu.name)
            if not dosya_id:
                dosya_id = dosya_yolu.stem
            dosya_id = shorten_title(dosya_id, max_length=80)
            zotero_meta = fetch_zotero_metadata(dosya_id)

            # Embedding oluşturma (temiz metin üzerinden)
            embedding = embed_text(temiz_metin)

            # Sonuç sözlüğü hazırlanıyor
            result = {
                "dosya": dosya_yolu.name,
                "ham_metin": ham_metin,
                "temiz_metin": temiz_metin,
                "harita": harita,
                "bolum_haritasi": bolum_haritasi,
                "sutun_bilgisi": sutun_bilgisi,
                "kaynakca": references,
                "zotero_meta": zotero_meta,
                "embedding": embedding,
                "islem_tarihi": datetime.now().isoformat()
            }

            self.stack_guncelle(dosya_yolu.name, "sil")
            self.sayaçlar['başarılı'] += 1
            config.logger.info(f"{dosya_yolu.name} başarıyla işlendi.")
            return result
        except Exception as e:
            self.sayaçlar['hata'] += 1
            config.logger.error(f"{dosya_yolu.name} işlenirken hata: {e}", exc_info=True)
            return None

    def stack_guncelle(self, dosya_adi, islem):
        """
        Stack güncelleme işlemini helper_module üzerinden gerçekleştirir.
        """
        from helper_module import stack_guncelle
        with self.stack_lock:
            stack_guncelle(dosya_adi, islem)
 
 #---------------------------------------processing_manager modulu sonu------------------------------------------------

import os
import json
import csv
import pandas as pd
from pathlib import Path
from config_module import config

def save_text_file(directory, filename, content):
    """
    Genel metin dosyalarını TXT formatında kaydeder.
    
    Args:
        directory (str or Path): Dosyanın kaydedileceği dizin.
        filename (str): Dosya adı (uzantı eklenmeyecek, fonksiyon ekleyecek).
        content (str): Kaydedilecek metin içeriği.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{filename}.txt"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        config.logger.info(f"TXT dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"TXT dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_json_file(directory, filename, content):
    """
    Veriyi JSON formatında kaydeder.
    
    Args:
        directory (str or Path): Dosyanın kaydedileceği dizin.
        filename (str): Dosya adı (uzantı eklenmeyecek).
        content (dict): Kaydedilecek veri.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{filename}.json"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        config.logger.info(f"JSON dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"JSON dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_clean_text_files(original_filename, clean_text, bib_info):
    """
    Temiz metin dosyasını hem TXT hem de JSON formatında, bibliyometrik bilgilerle birlikte kaydeder.
    
    Dosya isimlendirme: {ID}.clean.txt ve {ID}.clean.meta.json
    ID, dosyanın temel adı (dokuman_id_al + shorten_title ile elde edilir).
    
    Args:
        original_filename (str): İşlenen dosyanın orijinal adı.
        clean_text (str): Temizlenmiş metin.
        bib_info (dict): Bibliyometrik bilgiler.
    
    Returns:
        dict: {"txt": dosya_yolu, "json": dosya_yolu} şeklinde dosya yolları.
    """
    base_name = Path(original_filename).stem
    txt_path = save_text_file(config.CLEAN_TEXT_DIR / "txt", f"{base_name}.clean", clean_text)
    json_path = save_json_file(config.CLEAN_TEXT_DIR / "json", f"{base_name}.clean.meta", bib_info)
    return {"txt": txt_path, "json": json_path}

def save_references_files(original_filename, references, bib_info):
    """
    Kaynakça verilerini farklı formatlarda kaydeder:
      - TXT: {ID}.references.txt
      - JSON: {ID}.references.meta.json
      - VOSviewer: {ID}.vos.references.txt
      - Pajek: {ID}.pjk.references.paj
      - CSV: {ID}.references.csv  (Her satırda bir kaynakça kaydı)
    
    Args:
        original_filename (str): Orijinal dosya adı.
        references (list): Kaynakça metinlerinin listesi.
        bib_info (dict): Bibliyometrik bilgiler.
    
    Returns:
        dict: Kaydedilen dosya yollarını içeren sözlük.
    """
    base_name = Path(original_filename).stem
    ref_txt = save_text_file(config.REFERENCES_DIR / "txt", f"{base_name}.references", "\n".join(references))
    ref_json = save_json_file(config.REFERENCES_DIR / "json", f"{base_name}.references.meta", bib_info)
    # VOSviewer formatı: Basit liste, her satırda bir kaynak
    vos_path = save_text_file(config.REFERENCES_DIR / "vosviewer", f"{base_name}.vos.references", "\n".join(references))
    # Pajek formatı: İlk satırda toplam vertex sayısı, sonraki satırlarda id ve kaynakça metni
    pajek_file = config.REFERENCES_DIR / "pajek" / f"{base_name}.pjk.references.paj"
    (config.REFERENCES_DIR / "pajek").mkdir(parents=True, exist_ok=True)
    try:
        with open(pajek_file, 'w', encoding='utf-8') as f:
            f.write(f"*Vertices {len(references)}\n")
            for idx, ref in enumerate(references, start=1):
                f.write(f'{idx} "{ref}"\n')
        config.logger.info(f"Pajek dosyası kaydedildi: {pajek_file}")
        pajek_path = str(pajek_file)
    except Exception as e:
        config.logger.error(f"Pajek dosyası kaydedilemedi: {pajek_file}, Hata: {e}")
        pajek_path = None

    # CSV formatı: Her kaynakça için bir satır
    csv_file = config.REFERENCES_DIR / "csv" / f"{base_name}.references.csv"
    (config.REFERENCES_DIR / "csv").mkdir(parents=True, exist_ok=True)
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Kaynakça"])
            for ref in references:
                writer.writerow([ref])
        config.logger.info(f"CSV dosyası kaydedildi: {csv_file}")
        csv_path = str(csv_file)
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {csv_file}, Hata: {e}")
        csv_path = None

    return {"txt": ref_txt, "json": ref_json, "vosviewer": vos_path, "pajek": pajek_path, "csv": csv_path}

def save_table_files(original_filename, table_data_list):
    """
    Tabloları JSON, CSV ve Excel formatlarında kaydeder.
    
    Args:
        original_filename (str): Orijinal dosya adı.
        table_data_list (list): Tabloların verilerini içeren liste. Her tablo, 'baslik' ve 'veriler' anahtarlarına sahip sözlük olarak tanımlanmalı.
    
    Returns:
        dict: Kaydedilen dosya yollarını içeren sözlük.
    """
    base_name = Path(original_filename).stem
    table_dir = config.TABLES_DIR
    table_dir.mkdir(parents=True, exist_ok=True)

    # JSON formatında kaydet
    json_path = table_dir / f"{base_name}.tables.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(table_data_list, f, ensure_ascii=False, indent=4)
        config.logger.info(f"Tablolar JSON formatında kaydedildi: {json_path}")
    except Exception as e:
        config.logger.error(f"JSON dosyası kaydedilemedi: {json_path}, Hata: {e}")
    
    # CSV formatında kaydet
    csv_path = table_dir / f"{base_name}.tables.csv"
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Tablo Başlığı", "Tablo İçeriği"])
            for table in table_data_list:
                writer.writerow([table.get("baslik", ""), json.dumps(table.get("veriler", []), ensure_ascii=False)])
        config.logger.info(f"Tablolar CSV formatında kaydedildi: {csv_path}")
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {csv_path}, Hata: {e}")

    # Excel formatında kaydet
    excel_path = table_dir / f"{base_name}.tables.xlsx"
    try:
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        for idx, table in enumerate(table_data_list, start=1):
            df = pd.DataFrame(table.get("veriler", []))
            sheet_name = f"Tablo{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
        config.logger.info(f"Tablolar Excel formatında kaydedildi: {excel_path}")
    except Exception as e:
        config.logger.error(f"Excel dosyası kaydedilemedi: {excel_path}, Hata: {e}")

    return {"json": str(json_path), "csv": str(csv_path), "excel": str(excel_path)}

def save_embedding_file(original_filename, embedding_text, chunk_index):
    """
    Her dosyanın embedding verilerini kaydeder.
    Dosya isimlendirme: {ID}_chunk{chunk_index}.embed.txt
    
    Args:
        original_filename (str): Orijinal dosya adı.
        embedding_text (str): Embedding verisinin kaydedilecek metin hali.
        chunk_index (int): Chunk numarası.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    base_name = Path(original_filename).stem
    embedding_dir = config.EMBEDDINGS_DIR
    embedding_dir.mkdir(parents=True, exist_ok=True)
    file_path = embedding_dir / f"{base_name}_chunk{chunk_index}.embed.txt"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(embedding_text)
        config.logger.info(f"Embedding dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"Embedding dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_chunked_text_files(original_filename, full_text, chunk_size=256):
    """
    Büyük metni, belirlenen chunk boyutuna göre bölerek dosya sistemine kaydeder.
    Dosya isimlendirme: {ID}_part{parça_numarası}.txt
    
    Args:
        original_filename (str): Orijinal dosya adı.
        full_text (str): Tüm metin.
        chunk_size (int): Her chunk için karakter sayısı (varsayılan: 256).
    
    Returns:
        list: Kaydedilen tüm dosya yollarının listesi.
    """
    base_name = Path(original_filename).stem
    chunk_dir = config.CLEAN_TEXT_DIR / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    text_chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    file_paths = []
    for idx, chunk in enumerate(text_chunks, start=1):
        file_path = save_text_file(chunk_dir, f"{base_name}_part{idx}", chunk)
        if file_path:
            file_paths.append(file_path)
    config.logger.info(f"Büyük metin {len(text_chunks)} parçaya bölündü ve kaydedildi.")
    return file_paths
#---------------------------------------file_save_module sonu------------------------------------------------
import re
import json
from pathlib import Path
from rapidfuzz import fuzz
from config_module import config
from helper_module import fuzzy_match  # RapidFuzz tabanlı benzerlik hesaplaması

def split_into_sentences(text):
    """
    📌 Metni cümlelere böler ve her cümleye sıra numarası ekler.
    
    Args:
        text (str): İşlenecek metin.
    
    Returns:
        list: Her biri {'id': cümle numarası, 'text': cümle} içeren sözlüklerin listesi.
    """
    # Noktalama işaretleri sonrasında böl
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [{"id": idx, "text": sentence.strip()} for idx, sentence in enumerate(sentences, start=1) if sentence.strip()]

def extract_citations_from_sentence(sentence):
    """
    📌 Cümledeki atıf ifadelerini tespit eder.
    Regex desenleri ile farklı atıf stillerini yakalar.
    
    Args:
        sentence (str): İşlenecek cümle.
    
    Returns:
        list: Tespit edilen atıf ifadelerinin listesi.
    """
    # Örnek regex desenleri; daha gelişmiş stiller eklenebilir.
    citation_patterns = [
        r"\(([\w\s\.,\-]+, \d{4})\)",   # Örneğin: (Smith, 2020)
        r"\[\d+\]",                    # Örneğin: [12]
        r"\b(?:[A-Z][a-z]+ et al\.?, \d{4})\b"  # Örneğin: Smith et al., 2020
    ]
    citations = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, sentence)
        if matches:
            citations.extend(matches)
    # Tekrarları kaldır
    return list(set(citations))

def match_citation_with_references(citation_marker, references):
    """
    📌 Tespit edilen atıf ifadesini, kaynakça listesiyle eşleştirir.
    Önce tam eşleşme, ardından fuzzy matching (RapidFuzz) kullanılır.
    
    Args:
        citation_marker (str): Atıf ifadesi.
        references (list): Kaynakça metinlerinin listesi.
    
    Returns:
        str veya None: Eşleşen referans bulunursa döner, bulunamazsa None.
    """
    # Tam eşleşme
    for ref in references:
        if citation_marker.lower() in ref.lower():
            return ref
    
    # Fuzzy matching
    best_match = None
    best_score = 0
    for ref in references:
        score = fuzz.ratio(citation_marker.lower(), ref.lower())
        if score > best_score:
            best_match = ref
            best_score = score
    return best_match if best_score >= 85 else None

def get_section_for_sentence(sentence_id, section_info):
    """
    📌 Cümlenin ait olduğu bilimsel bölümü belirler.
    
    Args:
        sentence_id (int): Cümlenin sıra numarası.
        section_info (dict): Bölüm haritalama bilgileri (örneğin, {"Introduction": {"start": 5, "end": 20, ...}, ...}).
    
    Returns:
        str: Cümlenin ait olduğu bölüm veya "Unknown".
    """
    for section, info in section_info.items():
        if info and info.get("start") is not None and info.get("end") is not None:
            if info["start"] <= sentence_id <= info["end"]:
                return section
    return "Unknown"

def map_citations(clean_text, bibliography, section_info):
    """
    📌 Temiz metin içerisindeki tüm cümleleri işleyerek atıf mapping yapısını oluşturur.
    Her cümle için; cümle numarası, metni, tespit edilen atıf(lar), eşleşen referans ve bölüm bilgisi döndürülür.
    
    Args:
        clean_text (str): Temizlenmiş makale metni.
        bibliography (list): Kaynakça referanslarının listesi.
        section_info (dict): Bölüm haritalama bilgileri.
    
    Returns:
        list: Atıf mapping verilerini içeren sözlüklerin listesi.
    """
    mapped_citations = []
    sentences = split_into_sentences(clean_text)
    for sentence_obj in sentences:
        sent_id = sentence_obj["id"]
        text = sentence_obj["text"]
        citations = extract_citations_from_sentence(text)
        for citation in citations:
            matched_ref = match_citation_with_references(citation, bibliography)
            mapped_citations.append({
                "sentence_id": sent_id,
                "sentence": text,
                "citation": citation,
                "matched_reference": matched_ref,
                "section": get_section_for_sentence(sent_id, section_info)
            })
    return mapped_citations

def save_citation_mapping(pdf_id, citation_mapping):
    """
    📌 Oluşturulan Citation Mapping verilerini, bibliyometrik bilgilerle birlikte "ID.citation.json" olarak kaydeder.
    
    Args:
        pdf_id (str): PDF dosya ID'si.
        citation_mapping (list): Citation mapping verilerini içeren liste.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    citation_dir = Path(config.SUCCESS_DIR) / "citations"
    citation_dir.mkdir(parents=True, exist_ok=True)
    file_path = citation_dir / f"{pdf_id}.citation.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(citation_mapping, f, ensure_ascii=False, indent=4)
        config.logger.info(f"✅ Citation Mapping kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"❌ Citation Mapping kaydedilemedi: {file_path}, Hata: {e}")
        return None

def load_citation_mapping(pdf_id):
    """
    📌 Daha önce kaydedilmiş Citation Mapping dosyasını yükler.
    
    Args:
        pdf_id (str): PDF dosya ID'si.
    
    Returns:
        list veya None: Citation mapping verisi, bulunamazsa veya hata olursa None.
    """
    citation_dir = Path(config.SUCCESS_DIR) / "citations"
    file_path = citation_dir / f"{pdf_id}.citation.json"
    if not file_path.exists():
        config.logger.error(f"❌ Citation Mapping dosyası bulunamadı: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        config.logger.error(f"❌ Citation Mapping dosyası yüklenemedi: {file_path}, Hata: {e}")
        return None
    #---------------------------------------citation_mapping modulu sonu------------------------------------------------
import os
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from processing_manager import IslemYoneticisi
from citation_mapping_module import load_citation_mapping
from embedding_module import embed_text  # Temel embedding oluşturma (arama için kullanılabilir)
from clustering_module import perform_clustering  # Kümeleme analizi fonksiyonu
from fine_tuning_module import train_custom_model  # Fine-tuning model eğitimi
from data_query_module import query_data  # Gelişmiş veri sorgulama fonksiyonu
from config_module import config

class AnaArayuz(ctk.CTk):
    def __init__(self, islem_yoneticisi):
        super().__init__()
        self.islem_yoneticisi = islem_yoneticisi
        self.title("📑 Zotero Entegre PDF İşleyici")
        self.geometry("1200x800")
        self._arayuzu_hazirla()

    def _arayuzu_hazirla(self):
        """
        Ana GUI bileşenlerini oluşturur.
        """
        # Üst bölümde dosya seçme ve işlem başlatma butonları
        self.dosya_sec_btn = ctk.CTkButton(self, text="📂 PDF Seç", command=self._dosya_sec)
        self.dosya_sec_btn.pack(pady=10)

        self.baslat_btn = ctk.CTkButton(self, text="🚀 İşlemi Başlat", command=self._islem_baslat)
        self.baslat_btn.pack(pady=10)

        self.citation_btn = ctk.CTkButton(self, text="📖 Atıf Zinciri Görüntüle", command=self._atif_goster)
        self.citation_btn.pack(pady=10)

        # İlave özellikler menüsü
        self.ilave_ozellikler_menusu()

        # Sonuç Ekranı
        self.sonuc_ekrani = ctk.CTkTextbox(self, width=1000, height=500)
        self.sonuc_ekrani.pack(pady=10)

        # Durum Çubuğu (isteğe bağlı)
        self.status_bar = ctk.CTkLabel(self, text="Durum: Hazır", anchor="w")
        self.status_bar.pack(fill="x", padx=10, pady=5)

    def _dosya_sec(self):
        """
        Kullanıcının PDF dosyası seçmesini sağlar.
        """
        dosya_yolu = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if dosya_yolu:
            self.sonuc_ekrani.insert("end", f"\n📄 Seçilen Dosya: {dosya_yolu}\n")
            self.islem_yoneticisi.secili_dosya = dosya_yolu
            self.status_bar.configure(text=f"Seçilen dosya: {Path(dosya_yolu).name}")

    def _islem_baslat(self):
        """
        Seçili PDF dosyası işlenir.
        """
        if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
            messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
            return

        self.sonuc_ekrani.insert("end", "\n⏳ İşlem başlatılıyor...\n")
        self.status_bar.configure(text="İşlem başlatılıyor...")
        basari, sonuc = self.islem_yoneticisi.pdf_txt_isle(Path(self.islem_yoneticisi.secili_dosya))

        if basari:
            self.sonuc_ekrani.insert("end", f"✅ İşlem tamamlandı: {self.islem_yoneticisi.secili_dosya}\n")
            self.status_bar.configure(text="İşlem tamamlandı.")
        else:
            self.sonuc_ekrani.insert("end", f"❌ Hata oluştu: {sonuc}\n")
            self.status_bar.configure(text="Hata oluştu.")

    def _atif_goster(self):
        """
        Seçili PDF dosyasının atıf zincirini görüntüler.
        """
        if not hasattr(self.islem_yoneticisi, "secili_dosya") or not self.islem_yoneticisi.secili_dosya:
            messagebox.showerror("Hata", "Lütfen önce bir PDF dosyası seçin!")
            return

        pdf_id = Path(self.islem_yoneticisi.secili_dosya).stem
        citation_data = load_citation_mapping(pdf_id)

        if citation_data:
            display_text = "\n📚 Atıf Zinciri:\n"
            for entry in citation_data:
                display_text += f"🔹 Cümle {entry['sentence_id']}: {entry['sentence']}\n    Eşleşen Referans: {entry['matched_reference']}\n"
            self.sonuc_ekrani.insert("end", display_text)
            self.status_bar.configure(text="Atıf zinciri görüntülendi.")
        else:
            self.sonuc_ekrani.insert("end", "⚠️ Atıf verisi bulunamadı!\n")
            self.status_bar.configure(text="Atıf verisi bulunamadı.")

    def ilave_ozellikler_menusu(self):
        """
        İlave Özellikler Menüsü: Embedding Arama, Kümeleme Analizi, Fine-Tuning Modeli, Gelişmiş Veri Sorgulama.
        """
        # Embedding Arama
        self.embedding_btn = ctk.CTkButton(self, text="🔍 Embedding Arama", command=self._embedding_arama)
        self.embedding_btn.pack(pady=5)

        # Kümeleme Analizi
        self.kumeleme_btn = ctk.CTkButton(self, text="📊 Kümeleme Analizi", command=self._kumeleme_analiz)
        self.kumeleme_btn.pack(pady=5)

        # Fine-Tuning Modeli
        self.fine_tune_btn = ctk.CTkButton(self, text="🏋‍♂ Fine-Tuning Modeli", command=self._fine_tune_model)
        self.fine_tune_btn.pack(pady=5)

        # Gelişmiş Veri Sorgulama
        self.veri_sorgu_btn = ctk.CTkButton(self, text="🔎 Gelişmiş Veri Sorgulama", command=self._veri_sorgu)
        self.veri_sorgu_btn.pack(pady=5)

    def _embedding_arama(self):
        """
        Kullanıcının girdiği metinle en yakın embeddingleri arar.
        """
        query = self._kullanici_girdisi_al("Embedding Arama", "Aranacak metni girin:")
        if query:
            # search_embedding fonksiyonu, embedding modülünden veya alternatif embedding modülünden çağrılabilir.
            # Örneğin: search_embedding(query) şeklinde.
            try:
                from alternative_embedding_module import get_available_models, embed_text_with_model
                # Örnek: ilk model ile dene, gerçek uygulamada daha gelişmiş bir arama algoritması kullanılabilir.
                model_list = get_available_models()
                embedding = embed_text_with_model(query, model_list[0])
                result_text = f"Arama sonucu (Model: {model_list[0]}): {embedding[:10]} ... (embedding vektörü)"
            except Exception as e:
                result_text = f"Embedding arama hatası: {e}"
            self._sonuc_goster("🔍 Embedding Arama Sonuçları", result_text)

    def _kumeleme_analiz(self):
        """
        Kümeleme analizi yapar ve sonuçları gösterir.
        """
        try:
            clusters = perform_clustering()  # clustering_module.py'deki fonksiyon
            self._sonuc_goster("📊 Kümeleme Analizi Sonuçları", clusters)
        except Exception as e:
            self._sonuc_goster("📊 Kümeleme Analizi Hatası", str(e))

    def _fine_tune_model(self):
        """
        Fine-Tuning Modeli eğitimi başlatır ve sonuçları gösterir.
        """
        try:
            result = train_custom_model()  # fine_tuning_module.py'deki fonksiyon
            self._sonuc_goster("🏋‍♂ Fine-Tuning Sonuçları", result)
        except Exception as e:
            self._sonuc_goster("🏋‍♂ Fine-Tuning Hatası", str(e))

    def _veri_sorgu(self):
        """
        Gelişmiş veri sorgulama yapar.
        """
        query_params = self._kullanici_girdisi_al("Veri Sorgulama", "Sorgu parametrelerini girin:")
        if query_params:
            try:
                results = query_data(query_params)  # data_query_module.py'den fonksiyon
                self._sonuc_goster("🔎 Veri Sorgulama Sonuçları", results)
            except Exception as e:
                self._sonuc_goster("🔎 Veri Sorgulama Hatası", str(e))

    def _sonuc_goster(self, baslik, icerik):
        """
        Sonuçları GUI üzerinde gösterir.
        """
        self.sonuc_ekrani.insert("end", f"\n{baslik}:\n{icerik}\n")

    def _kullanici_girdisi_al(self, baslik, mesaj):
        """
        Kullanıcıdan input almak için diyalog penceresi açar.
        """
        # CustomTkinter'ın input diyalog kutusunu kullanıyoruz.
        input_dialog = ctk.CTkInputDialog(title=baslik, text=mesaj)
        return input_dialog.get_input()

if __name__ == '__main__':
    # İşlem yöneticisini oluştur ve GUI'yi başlat.
    islem_yoneticisi = IslemYoneticisi()
    arayuz = AnaArayuz(islem_yoneticisi)
    arayuz.mainloop()
    #---------------------------------------gui modulu sonu------------------------------------------------
    import os
import multiprocessing
from config_module import config
from processing_manager import IslemYoneticisi
from gui_module import AnaArayuz

if __name__ == '__main__':
    # Windows üzerinde çoklu işlem desteği için gerekli
    multiprocessing.freeze_support()
    
    try:
        # İşlem yöneticisini oluşturuyoruz
        islem_yoneticisi = IslemYoneticisi()
        
        # STORAGE_DIR içindeki dosya sayısını sayaçlara aktaralım
        try:
            total_files = len(os.listdir(config.STORAGE_DIR))
        except Exception as e:
            config.logger.error(f"STORAGE_DIR okunamadı: {e}")
            total_files = 0
        islem_yoneticisi.sayaçlar['toplam'] = total_files
        
        # Ana GUI arayüzünü oluştur ve başlat
        arayuz = AnaArayuz(islem_yoneticisi)
        arayuz.mainloop()
    
    except Exception as e:
        config.logger.critical(f"Ana uygulama hatası: {e}", exc_info=True)
    
    finally:
        # Son istatistik raporu
        print("\nİşlem Tamamlandı!")
        print(f"Toplam Dosya: {islem_yoneticisi.sayaçlar.get('toplam', 0)}")
        print(f"Başarılı: {islem_yoneticisi.sayaçlar.get('başarılı', 0)}")
        print(f"Hatalı: {islem_yoneticisi.sayaçlar.get('hata', 0)}")

#---------------------------------------main modulu sonu------------------------------------------------
import os
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from config_module import config

def perform_clustering(data, n_clusters=5, use_hdbscan=False):
    """
    📌 Verilen metin verileri üzerinde kümeleme analizi yapar.
    
    İş Akışı:
      1. Girdi metinler, TfidfVectorizer kullanılarak vektörleştirilir (maksimum 1000 özellik).
      2. Eğer use_hdbscan True ise HDBSCAN kullanılarak kümeleme yapılmaya çalışılır.
         - HDBSCAN kurulu değilse hata loglanır ve KMeans'e geçilir.
      3. Varsayılan olarak KMeans, n_clusters parametresi ile belirlenen küme sayısına göre kümeleme yapar.
      4. Küme etiketleri (labels) ve KMeans için merkezler (centers) elde edilir.
      5. Sonuçlar, bir sözlük şeklinde (labels, centers, orijinal veriler) döndürülür.
    
    Args:
        data (list): Kümeleme yapılacak metin verilerinin listesi.
        n_clusters (int, optional): KMeans algoritması için küme sayısı. Varsayılan 5.
        use_hdbscan (bool, optional): True ise HDBSCAN algoritması kullanılmaya çalışılır. Varsayılan False.
    
    Returns:
        dict or None: Kümeleme analizi sonucunu içeren sözlük ({"labels": ..., "centers": ..., "data": ...})
                      veya hata durumunda None.
    """
    if not data or not isinstance(data, list):
        config.logger.error("Kümeleme için geçerli veri sağlanamadı.")
        return None

    try:
        # Metin verilerini TF-IDF vektörlerine dönüştür
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data)
        config.logger.info("TF-IDF vektörleştirme tamamlandı.")
        
        if use_hdbscan:
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
                labels = clusterer.fit_predict(X.toarray())
                centers = None  # HDBSCAN merkez bilgisi sağlamaz
                config.logger.info("HDBSCAN ile kümeleme tamamlandı.")
            except ImportError as ie:
                config.logger.error("HDBSCAN kütüphanesi yüklü değil, KMeans kullanılacak. (Hata: %s)", ie)
                use_hdbscan = False
        
        if not use_hdbscan:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(X)
            centers = clusterer.cluster_centers_.tolist()
            config.logger.info("KMeans ile kümeleme tamamlandı.")
        
        result = {
            "labels": labels.tolist() if hasattr(labels, "tolist") else list(labels),
            "centers": centers,
            "data": data
        }
        return result

    except Exception as e:
        config.logger.error("Kümeleme analizi sırasında hata: %s", e, exc_info=True)
        return None

#---------------------------------------clustering_module sonu------------------------------------------------
import os
import json
import csv
import pandas as pd
from pathlib import Path
from config_module import config

def save_text_file(directory, filename, content):
    """
    Genel metin dosyalarını TXT formatında kaydeder.
    
    Args:
        directory (str or Path): Dosyanın kaydedileceği dizin.
        filename (str): Dosya adı (uzantı eklenmeyecek, fonksiyon ekleyecek).
        content (str): Kaydedilecek metin içeriği.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{filename}.txt"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        config.logger.info(f"TXT dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"TXT dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_json_file(directory, filename, content):
    """
    Veriyi JSON formatında kaydeder.
    
    Args:
        directory (str or Path): Dosyanın kaydedileceği dizin.
        filename (str): Dosya adı (uzantı eklenmeyecek).
        content (dict): Kaydedilecek veri.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / f"{filename}.json"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        config.logger.info(f"JSON dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"JSON dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_clean_text_files(original_filename, clean_text, bib_info):
    """
    Temiz metin dosyasını hem TXT hem de JSON formatında, bibliyometrik bilgilerle birlikte kaydeder.
    
    Dosya isimlendirme: {ID}.clean.txt ve {ID}.clean.meta.json
    ID, dosyanın temel adı (dokuman_id_al + shorten_title ile elde edilir).
    
    Args:
        original_filename (str): İşlenen dosyanın orijinal adı.
        clean_text (str): Temizlenmiş metin.
        bib_info (dict): Bibliyometrik bilgiler.
    
    Returns:
        dict: {"txt": dosya_yolu, "json": dosya_yolu} şeklinde dosya yolları.
    """
    base_name = Path(original_filename).stem
    txt_path = save_text_file(config.CLEAN_TEXT_DIR / "txt", f"{base_name}.clean", clean_text)
    json_path = save_json_file(config.CLEAN_TEXT_DIR / "json", f"{base_name}.clean.meta", bib_info)
    return {"txt": txt_path, "json": json_path}

def save_references_files(original_filename, references, bib_info):
    """
    Kaynakça verilerini farklı formatlarda kaydeder:
      - TXT: {ID}.references.txt
      - JSON: {ID}.references.meta.json
      - VOSviewer: {ID}.vos.references.txt
      - Pajek: {ID}.pjk.references.paj
      - CSV: {ID}.references.csv  (Her satırda bir kaynakça kaydı)
    
    Args:
        original_filename (str): Orijinal dosya adı.
        references (list): Kaynakça metinlerinin listesi.
        bib_info (dict): Bibliyometrik bilgiler.
    
    Returns:
        dict: Kaydedilen dosya yollarını içeren sözlük.
    """
    base_name = Path(original_filename).stem
    ref_txt = save_text_file(config.REFERENCES_DIR / "txt", f"{base_name}.references", "\n".join(references))
    ref_json = save_json_file(config.REFERENCES_DIR / "json", f"{base_name}.references.meta", bib_info)
    # VOSviewer formatı: Basit liste, her satırda bir kaynak
    vos_path = save_text_file(config.REFERENCES_DIR / "vosviewer", f"{base_name}.vos.references", "\n".join(references))
    # Pajek formatı: İlk satırda toplam vertex sayısı, sonraki satırlarda id ve kaynakça metni
    pajek_file = config.REFERENCES_DIR / "pajek" / f"{base_name}.pjk.references.paj"
    (config.REFERENCES_DIR / "pajek").mkdir(parents=True, exist_ok=True)
    try:
        with open(pajek_file, 'w', encoding='utf-8') as f:
            f.write(f"*Vertices {len(references)}\n")
            for idx, ref in enumerate(references, start=1):
                f.write(f'{idx} "{ref}"\n')
        config.logger.info(f"Pajek dosyası kaydedildi: {pajek_file}")
        pajek_path = str(pajek_file)
    except Exception as e:
        config.logger.error(f"Pajek dosyası kaydedilemedi: {pajek_file}, Hata: {e}")
        pajek_path = None

    # CSV formatı: Her kaynakça için bir satır
    csv_file = config.REFERENCES_DIR / "csv" / f"{base_name}.references.csv"
    (config.REFERENCES_DIR / "csv").mkdir(parents=True, exist_ok=True)
    try:
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Kaynakça"])
            for ref in references:
                writer.writerow([ref])
        config.logger.info(f"CSV dosyası kaydedildi: {csv_file}")
        csv_path = str(csv_file)
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {csv_file}, Hata: {e}")
        csv_path = None

    return {"txt": ref_txt, "json": ref_json, "vosviewer": vos_path, "pajek": pajek_path, "csv": csv_path}

def save_table_files(original_filename, table_data_list):
    """
    Tabloları JSON, CSV ve Excel formatlarında kaydeder.
    
    Args:
        original_filename (str): Orijinal dosya adı.
        table_data_list (list): Tabloların verilerini içeren liste. Her tablo, 'baslik' ve 'veriler' anahtarlarına sahip sözlük olarak tanımlanmalı.
    
    Returns:
        dict: Kaydedilen dosya yollarını içeren sözlük.
    """
    base_name = Path(original_filename).stem
    table_dir = config.TABLES_DIR
    table_dir.mkdir(parents=True, exist_ok=True)

    # JSON formatında kaydet
    json_path = table_dir / f"{base_name}.tables.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(table_data_list, f, ensure_ascii=False, indent=4)
        config.logger.info(f"Tablolar JSON formatında kaydedildi: {json_path}")
    except Exception as e:
        config.logger.error(f"JSON dosyası kaydedilemedi: {json_path}, Hata: {e}")
    
    # CSV formatında kaydet
    csv_path = table_dir / f"{base_name}.tables.csv"
    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Tablo Başlığı", "Tablo İçeriği"])
            for table in table_data_list:
                writer.writerow([table.get("baslik", ""), json.dumps(table.get("veriler", []), ensure_ascii=False)])
        config.logger.info(f"Tablolar CSV formatında kaydedildi: {csv_path}")
    except Exception as e:
        config.logger.error(f"CSV dosyası kaydedilemedi: {csv_path}, Hata: {e}")

    # Excel formatında kaydet
    excel_path = table_dir / f"{base_name}.tables.xlsx"
    try:
        writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
        for idx, table in enumerate(table_data_list, start=1):
            df = pd.DataFrame(table.get("veriler", []))
            sheet_name = f"Tablo{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
        config.logger.info(f"Tablolar Excel formatında kaydedildi: {excel_path}")
    except Exception as e:
        config.logger.error(f"Excel dosyası kaydedilemedi: {excel_path}, Hata: {e}")

    return {"json": str(json_path), "csv": str(csv_path), "excel": str(excel_path)}

def save_embedding_file(original_filename, embedding_text, chunk_index):
    """
    Her dosyanın embedding verilerini kaydeder.
    Dosya isimlendirme: {ID}_chunk{chunk_index}.embed.txt
    
    Args:
        original_filename (str): Orijinal dosya adı.
        embedding_text (str): Embedding verisinin kaydedilecek metin hali.
        chunk_index (int): Chunk numarası.
    
    Returns:
        str: Oluşturulan dosya yolunu döndürür.
    """
    base_name = Path(original_filename).stem
    embedding_dir = config.EMBEDDINGS_DIR
    embedding_dir.mkdir(parents=True, exist_ok=True)
    file_path = embedding_dir / f"{base_name}_chunk{chunk_index}.embed.txt"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(embedding_text)
        config.logger.info(f"Embedding dosyası kaydedildi: {file_path}")
        return str(file_path)
    except Exception as e:
        config.logger.error(f"Embedding dosyası kaydedilemedi: {file_path}, Hata: {e}")
        return None

def save_chunked_text_files(original_filename, full_text, chunk_size=256):
    """
    Büyük metni, belirlenen chunk boyutuna göre bölerek dosya sistemine kaydeder.
    Dosya isimlendirme: {ID}_part{parça_numarası}.txt
    
    Args:
        original_filename (str): Orijinal dosya adı.
        full_text (str): Tüm metin.
        chunk_size (int): Her chunk için karakter sayısı (varsayılan: 256).
    
    Returns:
        list: Kaydedilen tüm dosya yollarının listesi.
    """
    base_name = Path(original_filename).stem
    chunk_dir = config.CLEAN_TEXT_DIR / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    text_chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    file_paths = []
    for idx, chunk in enumerate(text_chunks, start=1):
        file_path = save_text_file(chunk_dir, f"{base_name}_part{idx}", chunk)
        if file_path:
            file_paths.append(file_path)
    config.logger.info(f"Büyük metin {len(text_chunks)} parçaya bölündü ve kaydedildi.")
    return file_paths

#---------------------------------------fifine tuning modulu sonu------------------------------------------------

import os
from pathlib import Path
import glob
import json
import numpy as np
from config_module import config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def query_data(query_params):
    """
    📌 Gelişmiş veri sorgulama fonksiyonu.
    
    Bu fonksiyon, config.CLEAN_TEXT_DIR / "txt" dizininde bulunan tüm temiz metin dosyalarını
    tarayarak, kullanıcı tarafından girilen sorgu parametrelerine göre cosine similarity hesaplaması yapar.
    
    İş Akışı:
      1. Belirtilen dizindeki tüm .txt dosyaları okunur.
      2. Her dosyadan elde edilen metinler, dosya adı ile birlikte bir corpus oluşturur.
      3. TfidfVectorizer kullanılarak metinler vektörleştirilir.
      4. Kullanıcının sorgusu da aynı şekilde vektörleştirilir.
      5. Cosine similarity hesaplanarak en yüksek skorlu sonuçlar sıralanır.
      6. Her sonuç için dosya adı, benzerlik skoru ve ilk 200 karakterlik bir özet (snippet) oluşturulur.
    
    Args:
        query_params (str): Kullanıcının sorgu olarak girdiği metin.
    
    Returns:
        dict: Sorgu sonuçlarını içeren sözlük. Örnek:
              {
                  "results": [
                      {"file": "file1.txt", "similarity": 0.87, "snippet": "..." },
                      {"file": "file2.txt", "similarity": 0.75, "snippet": "..." },
                      ...
                  ]
              }
              Hata durumunda {"results": []} döndürülür.
    """
    try:
        # Clean text dosyalarının bulunduğu dizin
        txt_dir = Path(config.CLEAN_TEXT_DIR) / "txt"
        if not txt_dir.exists():
            config.logger.error(f"Clean text dizini bulunamadı: {txt_dir}")
            return {"results": []}
        
        # Tüm .txt dosyalarını topla
        file_paths = list(txt_dir.glob("*.txt"))
        corpus = []
        file_names = []
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    corpus.append(content)
                    file_names.append(file_path.name)
            except Exception as e:
                config.logger.error(f"Dosya okunamadı: {file_path} Hata: {e}")
        
        if not corpus:
            config.logger.error("Sorgulama için temiz metin dosyaları bulunamadı.")
            return {"results": []}
        
        # TF-IDF vektörleştirme
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(corpus)
        
        # Sorgu metnini vektörleştir
        query_vec = vectorizer.transform([query_params])
        
        # Cosine similarity hesapla
        similarities = cosine_similarity(X, query_vec).flatten()
        
        # Benzerlik skoruna göre en yüksek sonuçları sırala
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            sim_score = float(similarities[idx])
            if sim_score < 0.1:
                continue  # Çok düşük benzerlik, atla.
            snippet = corpus[idx][:200].replace("\n", " ")  # İlk 200 karakter snippet olarak alınır
            results.append({
                "file": file_names[idx],
                "similarity": round(sim_score, 4),
                "snippet": snippet
            })
        
        config.logger.info(f"Veri sorgulama tamamlandı, {len(results)} sonuç bulundu.")
        return {"results": results}
    except Exception as e:
        config.logger.error(f"Veri sorgulama sırasında hata: {e}", exc_info=True)
        return {"results": []}
    
    #---------------------------------------data_query_module sonu------------------------------------------------
    