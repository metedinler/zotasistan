


import os
import multiprocessing
from config_module import config
from processing_manager import IslemYoneticisi
from gui_module import AnaArayuz

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows için gerekli
    # İşlem yöneticisini oluştur ve STORAGE_DIR içerisindeki toplam dosya sayısını sayaçlara aktar
    islem_yoneticisi = IslemYoneticisi()
    islem_yoneticisi.sayaçlar['toplam'] = len(os.listdir(config.STORAGE_DIR))
    
    try:
        # Ana GUI'yi başlat
        arayuz = AnaArayuz(islem_yoneticisi)
        arayuz.mainloop()
    except Exception as e:
        config.logger.critical(f"Ana uygulama hatası: {e}", exc_info=True)
    finally:
        print("\nİstatistikler:")
        print(f"Toplam Dosya: {islem_yoneticisi.sayaçlar.get('toplam', 0)}")
        print(f"Başarılı: {islem_yoneticisi.sayaçlar.get('başarılı', 0)}")
        print(f"Hatalı: {islem_yoneticisi.sayaçlar.get('hata', 0)}")
