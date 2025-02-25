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

# Aşağıda, son 45 günlük tartışmalarımız ve tüm güncellemeleri göz önüne alarak,
# final versiyonu olan **clustering_module.py** modülünü bulabilirsiniz. Bu modül, 
# PDF veya metin verileri üzerinden kümeleme analizi yapmayı amaçlıyor. 
# Hem KMeans (varsayılan) hem de isteğe bağlı olarak HDBSCAN (kurulu ise) kullanılabiliyor. 
# Kod, verilerin TF-IDF vektörleştirilmesini sağlıyor, 
# ardından kümeleme algoritması ile analizi gerçekleştiriyor; sonuçlar detaylı loglama ile bildiriliyor.

# ### Açıklamalar

# - **Modülün Amacı:**  
#   Bu modül, verilen metin verileri üzerinden kümeleme analizi yapar. Amaç, metinlerin içerik benzerliklerine göre gruplandırılmasını sağlamaktır.

# - **Fonksiyonlar:**
#   - **perform_clustering(data, n_clusters=5, use_hdbscan=False):**  
#     - **Girdi:** Bir liste halinde metin verileri, isteğe bağlı küme sayısı ve HDBSCAN kullanım bayrağı.
#     - **İşlev:**  
#       1. Metin verilerini TF-IDF yöntemiyle vektörleştirir.
#       2. Kullanıcı HDBSCAN seçerse, HDBSCAN ile kümelemeyi dener; eğer kurulu değilse veya hata alırsa KMeans'e geçer.
#       3. KMeans ile kümeleme yaparsa, küme etiketlerini (labels) ve merkezleri (centers) elde eder.
#       4. Sonuçları, hem etiketleri hem de merkezleri içeren bir sözlük olarak döndürür.
#     - **Hata Yönetimi:**  
#       Tüm işlemler try/except bloklarıyla çevrilmiş, oluşan hatalar config.logger üzerinden loglanmaktadır.

# - **Değişkenler ve Veri Yapıları:**  
#   - **data:** List tipinde metin verileri.
#   - **n_clusters:** KMeans algoritması için küme sayısı.
#   - **use_hdbscan:** HDBSCAN algoritmasının kullanılacağına dair bayrak.
#   - **X:** TF-IDF vektörleştirilmiş veri (scipy sparse matrix).
#   - **result:** Sonuç sözlüğü; "labels" (küme etiketleri), "centers" (merkezler, KMeans için) ve "data" (orijinal veriler).

# - **Kontrol Yapıları:**  
#   - Girdi kontrolü: Geçerli bir veri listesi sağlanmış mı.
#   - İsteğe bağlı HDBSCAN kullanımı: Eğer HDBSCAN yüklü değilse veya hata alınırsa KMeans'e geçilir.
#   - Try/except blokları ile hata yönetimi sağlanır.

# Bu final versiyonu, tüm tartışmalarımız doğrultusunda gelişmiş hata yönetimi,
# loglama ve esnek algoritma seçeneği (KMeans ve HDBSCAN) sunar. Herhangi ek bir iyileştirme veya sorunuz varsa lütfen belirtin.