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

Aşağıda, tartışmalarımız ve güncellemeler doğrultusunda oluşturulmuş, 
final versiyonu olan **`data_query_module.py`** modülünü bulabilirsiniz.
Bu modül, "clean text" dosyalarının (TXT formatında) bulunduğu dizinde arama yaparak, 
kullanıcı tarafından girilen sorgu parametrelerine göre benzerlik analizi gerçekleştiriyor.

İş akışı şu şekildedir:

1. **Veri Toplama:**  
   Belirtilen (config üzerinden ayarlanan) "clean texts" dizininde bulunan tüm TXT dosyaları okunur.  
2. **Vektörleştirme:**  
   Tüm dosyalardan elde edilen metinler, TF-IDF yöntemiyle vektörleştirilir.  
3. **Sorgu İşlemi:**  
   Kullanıcı sorgusu da aynı TF-IDF modeli kullanılarak vektörleştirilir ve cosine similarity hesaplanır.  
4. **Sonuçların Sıralanması:**  
   Hesaplanan benzerlik skorlarına göre sonuçlar sıralanır.  
5. **Özetleme:**  
   Her dosyadan, ilk 200 karakterlik bir özet (snippet) alınarak sonuçlara eklenir.  
6. **Raporlama:**  
   Sonuçlar bir sözlük halinde döndürülür ve ilgili işlemler loglanır.

Aşağıda final kodu bulabilirsiniz:

# ### Açıklamalar

# - **Modülün Amacı:**  
#   `query_data` fonksiyonu, temiz metin dosyaları üzerinde gelişmiş veri sorgulama
# işlemleri gerçekleştirmek için tasarlanmıştır. Kullanıcı tarafından girilen sorgu metni,
# dosyalarla karşılaştırılır ve benzerlik skorlarına göre sonuçlar listelenir.

# - **Fonksiyonlar:**
#   - **query_data(query_params):**  
#     - **Girdi:** Kullanıcının sorgu metni.
#     - **İşlev:**  
#       1. Config üzerinden belirlenmiş temiz metin dizinindeki tüm TXT dosyaları okunur.
#       2. Metinler, TfidfVectorizer kullanılarak vektörleştirilir.
#       3. Sorgu metni de aynı şekilde vektörleştirilip cosine similarity hesaplanır.
#       4. Benzerlik skorlarına göre sonuçlar sıralanır ve ilk 200 karakterlik özet (snippet) oluşturulur.
#     - **Çıktı:** Dosya adı, benzerlik skoru ve snippet içeren sonuçların bulunduğu sözlük.

# - **Değişkenler ve Veri Yapıları:**  
#   - `txt_dir`: Clean metin dosyalarının bulunduğu dizin.
#   - `corpus`: Dosya içeriklerinin listesi.
#   - `file_names`: İlgili dosya adlarının listesi.
#   - `vectorizer`: TF-IDF vektörleştirme için kullanılan nesne.
#   - `similarities`: Cosine similarity skorlarının numpy array’i.
#   - `results`: Sorgu sonucunda döndürülecek sonuçların listesi.

# - **Kontrol Yapıları ve Hata Yönetimi:**  
#   - Eğer dizin veya dosyalar bulunamazsa, hata loglanır ve boş sonuç döndürülür.
#   - Try/except blokları kullanılarak oluşan tüm hatalar loglanır.

# Bu final versiyonu, önceki tartışmalarımızdaki gereksinimleri karşılayacak şekilde tasarlanmıştır.
# Eğer ek bir geliştirme veya değişiklik talebiniz varsa, lütfen belirtin.