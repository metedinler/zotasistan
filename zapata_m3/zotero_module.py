import os
import re
import requests
import json
from config_module import config

class ZoteroEntegratoru:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTERO_USER_ID}/items"
        self.headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}

    def meta_veri_al(self, item_key):
        """
        Belirtilen item_key için Zotero API'den bibliyometrik veriyi çeker.
        """
        try:
            response = requests.get(f"{self.base_url}/{item_key}", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            else:
                config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API hatası: {str(e)}")
            return None

    def referanslari_analiz_et(self, referans_listesi):
        """
        Verilen referans listesindeki her referans için, yazar bilgisini (ilk kelime) çıkararak analiz eder.
        Eğer referans bulunamazsa 'Bilinmeyen' döndürür.
        """
        try:
            analiz_sonuc = []
            for referans in referans_listesi:
                yazar = re.search(r'^([A-Za-z]+)', referans)
                analiz_sonuc.append({
                    'orijinal': referans,
                    'yazar': yazar.group(1) if yazar else 'Bilinmeyen'
                })
            return analiz_sonuc
        except Exception as e:
            config.logger.error(f"Referans analiz hatası: {str(e)}")
            return []

def dokuman_id_al(dosya_adi):
    """
    Dosya adından, örneğin 'ABCD1234.pdf' şeklinde bir isimden, temel dosya kimliğini (ID) çıkarır.
    """
    m = re.search(r"^([A-Z0-9]+)\..*", dosya_adi)
    return m.group(1) if m else None

def fetch_zotero_metadata(item_key):
    """
    Zotero API'den belirtilen item_key için bibliyometrik veriyi çeker.
    """
    headers = {"Zotero-API-Key": config.ZOTERO_API_KEY}
    try:
        response = requests.get(f"{config.base_url}/{item_key}", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            config.logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
            return None
    except Exception as e:
        config.logger.error(f"Zotero API isteğinde hata: {e}")
        return None

def save_references_for_analysis(references, vosviewer_file, pajek_file):
    """
    Kaynakça verilerini, VOSviewer ve Pajek formatlarında kaydeder.
    
    - VOSviewer dosyası: İlk satır 'label' içermeli, ardından her referans ayrı satırda yer alır.
    - Pajek dosyası: İlk satır "*Vertices <sayı>" şeklinde, sonrasında her referans numaralandırılarak listelenir.
    """
    try:
        with open(vosviewer_file, 'w', encoding='utf-8') as vos_file:
            vos_file.write("label\n")
            for ref in references:
                vos_file.write(f"{ref}\n")
        config.logger.info(f"VOSviewer formatında referanslar kaydedildi: {vosviewer_file}")
        
        with open(pajek_file, 'w', encoding='utf-8') as pajek_f:
            pajek_f.write(f"*Vertices {len(references)}\n")
            for i, ref in enumerate(references, 1):
                pajek_f.write(f'{i} "{ref}"\n')
        config.logger.info(f"Pajek formatında referanslar kaydedildi: {pajek_file}")
    except Exception as e:
        config.logger.error(f"Referanslar analiz formatlarına kaydedilemedi: {e}")
