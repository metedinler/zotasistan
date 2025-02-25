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

# Aşağıda, tartışmalarımız ve yapılan güncellemeler doğrultusunda oluşturulmuş final versiyonunu bulabilirsiniz. 
# Bu ana modül, tüm diğer modülleri (config_module, processing_manager, gui_module, vb.) entegre ederek 
# kullanıcıdan gelen girdiler doğrultusunda dosya işleme, embedding, Zotero entegrasyonu, 
# atıf mapping ve diğer işlemleri yönetiyor. Ayrıca, multiprocessing desteği ve hata yönetimiyle de güvence altına alınmıştır.

# ### Açıklamalar

# - **Multiprocessing Desteği:**  
#   `multiprocessing.freeze_support()` çağrısı, özellikle Windows üzerinde çoklu işlem kullanılırken gerekli.

# - **IslemYoneticisi:**  
#   `IslemYoneticisi` sınıfı, PDF/TXT dosyalarını işleyerek metin çıkarımı, temizleme, bölüm haritalama,
# embedding oluşturma, Zotero entegrasyonu ve diğer adımları yönetiyor. Bu sınıf, sayaçlar ve stack yönetimi ile işlem durumunu izliyor.

# - **GUI Entegrasyonu:**  
#   `AnaArayuz` sınıfı, kullanıcı arayüzünü oluşturur; PDF seçme, işlem başlatma, atıf zinciri görüntüleme 
# ve ek özellikler (embedding arama, kümeleme analizi, fine-tuning, veri sorgulama) için butonlar içerir.

# - **İstatistik Raporlaması:**  
#   İşlem sonunda toplam dosya, başarılı ve hatalı dosya sayıları konsola yazdırılır.

# Bu ana modül, sistemdeki tüm güncellemeleri ve entegrasyonları yönetmek üzere tasarlanmıştır. 
# Herhangi bir ek sorunuz veya değişiklik isteğiniz olursa lütfen belirtin.