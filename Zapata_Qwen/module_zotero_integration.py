# module_zotero_integration

# Below is the implementation of the `module_zotero_integration` module. This module focuses on integrating Zotero functionalities, such as fetching bibliographic metadata, managing collections, and handling Zotero-specific operations.

---

### **`module_zotero_integration.py`**

```python
import os
import requests
import logging
from dotenv import load_dotenv

# ----------------------------
# Logging Yapılandırması
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('zotero_integration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Ortam Ayarları ve API Yapılandırması
# ----------------------------

# .env dosyasını yükle
load_dotenv()

# Zotero API anahtarlarını yükle
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
ZOTERO_API_URL = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items"

if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
    logger.error("Zotero API anahtarı veya kullanıcı kimliği tanımlı değil.")
    raise ValueError("Zotero API anahtarı veya kullanıcı kimliği eksik.")

# ----------------------------
# Zotero Entegrasyon Fonksiyonları
# ----------------------------

def fetch_zotero_metadata(item_key):
    """
    Belirtilen Zotero kaydının bibliyografik bilgilerini alır.
    Args:
        item_key (str): Zotero'daki öğe anahtarı.
    Returns:
        dict: Bibliyografik bilgiler (JSON formatında).
    """
    headers = {"Zotero-API-Key": ZOTERO_API_KEY}
    url = f"{ZOTERO_API_URL}/{item_key}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Zotero API isteği başarısız: {e}")
        return None


def fetch_all_zotero_items():
    """
    Kullanıcının tüm Zotero öğelerini çeker.
    Returns:
        list: Zotero öğelerinin listesi (JSON formatında).
    """
    headers = {"Zotero-API-Key": ZOTERO_API_KEY}
    url = f"{ZOTERO_API_URL}?limit=100"  # Örnek olarak 100 öğe limiti
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Zotero API isteği başarısız: {e}")
        return []


def add_item_to_zotero(collection_key, item_data):
    """
    Yeni bir öğeyi belirtilen Zotero koleksiyonuna ekler.
    Args:
        collection_key (str): Zotero koleksiyon anahtarı.
        item_data (dict): Eklenecek öğenin verileri (JSON formatında).
    Returns:
        bool: İşlem başarılıysa True, aksi halde False.
    """
    headers = {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Content-Type": "application/json"
    }
    url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/collections/{collection_key}/items"
    try:
        response = requests.post(url, headers=headers, json=item_data)
        if response.status_code == 201:
            logger.info("Öğe başarıyla Zotero'ya eklendi.")
            return True
        else:
            logger.error(f"Öğe eklenemedi: {response.status_code}, Yanıt: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Zotero API isteği başarısız: {e}")
        return False


def delete_zotero_item(item_key):
    """
    Belirtilen Zotero öğesini siler.
    Args:
        item_key (str): Silinecek öğenin anahtarı.
    Returns:
        bool: İşlem başarılıysa True, aksi halde False.
    """
    headers = {"Zotero-API-Key": ZOTERO_API_KEY}
    url = f"{ZOTERO_API_URL}/{item_key}"
    try:
        response = requests.delete(url, headers=headers)
        if response.status_code == 204:
            logger.info(f"Öğe başarıyla silindi: {item_key}")
            return True
        else:
            logger.error(f"Öğe silinemedi: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Zotero API isteği başarısız: {e}")
        return False


def update_zotero_item(item_key, updated_data):
    """
    Belirtilen Zotero öğesini günceller.
    Args:
        item_key (str): Güncellenecek öğenin anahtarı.
        updated_data (dict): Güncellenmiş veriler (JSON formatında).
    Returns:
        bool: İşlem başarılıysa True, aksi halde False.
    """
    headers = {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Content-Type": "application/json"
    }
    url = f"{ZOTERO_API_URL}/{item_key}"
    try:
        response = requests.patch(url, headers=headers, json=updated_data)
        if response.status_code == 204:
            logger.info(f"Öğe başarıyla güncellendi: {item_key}")
            return True
        else:
            logger.error(f"Öğe güncellenemedi: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Zotero API isteği başarısız: {e}")
        return False


def create_zotero_collection(collection_name):
    """
    Yeni bir Zotero koleksiyonu oluşturur.
    Args:
        collection_name (str): Oluşturulacak koleksiyonun adı.
    Returns:
        str: Yeni koleksiyonun anahtarı (başarılıysa), aksi halde None.
    """
    headers = {
        "Zotero-API-Key": ZOTERO_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "name": collection_name
    }
    url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/collections"
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            collection_key = response.headers.get("Location").split("/")[-1]
            logger.info(f"Koleksiyon başarıyla oluşturuldu: {collection_name} (Anahtar: {collection_key})")
            return collection_key
        else:
            logger.error(f"Koleksiyon oluşturulamadı: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Zotero API isteği başarısız: {e}")
        return None


def delete_zotero_collection(collection_key):
    """
    Belirtilen Zotero koleksiyonunu siler.
    Args:
        collection_key (str): Silinecek koleksiyonun anahtarı.
    Returns:
        bool: İşlem başarılıysa True, aksi halde False.
    """
    headers = {"Zotero-API-Key": ZOTERO_API_KEY}
    url = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/collections/{collection_key}"
    try:
        response = requests.delete(url, headers=headers)
        if response.status_code == 204:
            logger.info(f"Koleksiyon başarıyla silindi: {collection_key}")
            return True
        else:
            logger.error(f"Koleksiyon silinemedi: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Zotero API isteği başarısız: {e}")
        return False


# ----------------------------
# Yardımcı Fonksiyonlar
# ----------------------------

def get_zotero_item_key_from_url(item_url):
    """
    Zotero öğesi URL'sinden anahtar çıkarır.
    Args:
        item_url (str): Zotero öğesi URL'si.
    Returns:
        str: Öğe anahtarı.
    """
    return item_url.split("/")[-1]


def save_zotero_metadata_to_file(metadata, file_path):
    """
    Zotero bibliyografik bilgilerini JSON dosyasına kaydeder.
    Args:
        metadata (dict): Kaydedilecek bibliyografik bilgiler.
        file_path (str): Kayıt edilecek dosya yolu.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        logger.info(f"Bibliyografik bilgiler kaydedildi: {file_path}")
    except Exception as e:
        logger.error(f"Bibliyografik bilgiler kaydedilirken hata oluştu: {e}")


# ----------------------------
# Örnek Kullanım
# ----------------------------

if __name__ == "__main__":
    # Örnek: Tüm öğeleri çekme
    items = fetch_all_zotero_items()
    for item in items:
        print(f"Öğe Başlığı: {item.get('data', {}).get('title')}")

    # Örnek: Yeni bir koleksiyon oluşturma
    new_collection_key = create_zotero_collection("Yeni Koleksiyon")
    if new_collection_key:
        print(f"Yeni koleksiyon anahtarı: {new_collection_key}")

    # Örnek: Bir öğeyi silme
    item_key = "örnek_anahtar"
    delete_zotero_item(item_key)
# ```
# 
# ---
# 
# ### **Modül Açıklaması**
# 
# 1. **`fetch_zotero_metadata`**:
#    - Belirli bir Zotero öğesinin bibliyografik bilgilerini çeker.
#    - Öğe anahtarı ile çağrılır.
# 
# 2. **`fetch_all_zotero_items`**:
#    - Kullanıcının tüm Zotero öğelerini çeker.
#    - Öğeler JSON formatında döner.
# 
# 3. **`add_item_to_zotero`**:
#    - Yeni bir öğeyi belirtilen Zotero koleksiyonuna ekler.
#    - Öğe verisi JSON formatında sağlanır.
# 
# 4. **`delete_zotero_item`**:
#    - Belirli bir Zotero öğesini siler.
#    - Öğe anahtarı ile çağrılır.
# 
# 5. **`update_zotero_item`**:
#    - Mevcut bir Zotero öğesini günceller.
#    - Güncellenmiş veri JSON formatında sağlanır.
# 
# 6. **`create_zotero_collection`**:
#    - Yeni bir Zotero koleksiyonu oluşturur.
#    - Koleksiyon adı ile çağrılır.
# 
# 7. **`delete_zotero_collection`**:
#    - Belirli bir Zotero koleksiyonunu siler.
#    - Koleksiyon anahtarı ile çağrılır.
# 
# 8. **Yardımcı Fonksiyonlar**:
#    - `get_zotero_item_key_from_url`: URL'den öğe anahtarı çıkarır.
#    - `save_zotero_metadata_to_file`: Bibliyografik bilgileri JSON dosyasına kaydeder.
# 
# ---
# 
# ### **Özellikler**
# 
# - **API Entegrasyonu**: Zotero API'si ile tam entegrasyon sağlar.
# - **Hata Yönetimi**: Her işlemde hata kontrolü yapılır ve log kaydı tutulur.
# - **Esneklik**: Farklı işlemler için özelleştirilebilir fonksiyonlar içerir.
# - **Güvenlik**: `.env` dosyası üzerinden API anahtarlarını güvenli bir şekilde yönetir.
# 
# ---

Bu modülü kullanarak programınızın Zotero ile entegre olmasını sağlayabilirsiniz. Eğer herhangi bir değişiklik veya ek özellik isterseniz, lütfen belirtin!