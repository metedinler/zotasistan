# zoteromodule.py

import re
import requests
from configmodule import Config

config = Config()

class ZoteroEntegrator:
    def __init__(self):
        self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
        self.headers = {
            'Zotero-API-Key': config.ZOTEROAPIKEY,
            'Content-Type': 'application/json'
        }

    def get_metadata(self, item_key):
        """
        Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

        Args:
            item_key (str): Zotero item key.

        Returns:
            dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
        """
        url = f"{self.base_url}/{item_key}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                config.logger.info(f"Zotero API'den veri alındı: {item_key}")
                return response.json()
            else:
                config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
                return None
        except Exception as e:
            config.logger.error(f"Zotero API isteğinde hata: {e}")
            return None

    def analyze_references(self, reference_list):
        """
        Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

        Args:
            reference_list (list): Kaynakça referans metinlerinin listesi.

        Returns:
            list: Her referans için analiz sonucu içeren sözlüklerin listesi.
        """
        analysis_results = []
        for reference in reference_list:
            try:
                # Yazar ismini regex ile çıkar
                match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
                author = match.group() if match else "Bilinmeyen"
                analysis_results.append({"reference": reference, "author": author})
            except Exception as e:
                config.logger.error(f"Referans analizi hatası: {e}")
                analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
        return analysis_results

    def map_citations(self, text, references):
        """
        Metin içindeki atıf ifadelerini tespit eder ve referanslarla eşler.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.

        Returns:
            dict: Atıf ifadeleri ve eşleştirilmiş referanslar.
        """
        # TODO: Atıf ifadelerini regex veya NER ile tespit et
        citations = {}
        # ...
        return citations

# ZoteroEntegrator sınıfını kullanmak için:
zotero_entegrator = ZoteroEntegrator()
item_key = "ITEM-1234"
metadata = zotero_entegrator.get_metadata(item_key)
if metadata:
    print(metadata)
else:
    print("Zotero API'den veri alınamadı.")


""" {
    "key": "ITEM-1234",
    "version": 5,
    "data": {
        "itemType": "journalArticle",
        "title": "Makale Başlığı",
        "creators": [
            {
                "creatorType": "author",
                "firstName": "Ali",
                "lastName": "Veli"
            }
        ],
        "publicationTitle": "Dergi Adı",
        "volume": "10",
        "issue": "2",
        "pages": "100-110",
        "date": "2023",
        "DOI": "10.1234/makale.1234",
        "abstractNote": "Makale özeti...",
        "url": "https://example.com/makale",
        "libraryCatalog": "Zotero",
        "callNumber": "1234",
        "rights": "Telif hakkı...",
        "extra": "Ek bilgiler...",
        "tags": [
            "anahtar kelime 1",
            "anahtar kelime 2"
        ],
        "collections": [
            "koleksiyon 1",
            "koleksiyon 2"
        ],
        "relations": {}
    },
    "meta": {
        "creatorSummary": "Ali Veli",
        "parsedDate": "2023-01-01",
        "numChildren": 0
    }
}
 """

# zoteromodule.py

# import re
# import requests
# from configmodule import Config

# config = Config()

# class ZoteroEntegrator:
#     def __init__(self):
#         self.base_url = f"https://api.zotero.org/users/{config.ZOTEROUSERID}/items"
#         self.headers = {
#             'Zotero-API-Key': config.ZOTEROAPIKEY,
#             'Content-Type': 'application/json'
#         }

#     def get_metadata(self, item_key):
#         """
#         Belirtilen Zotero item key'e göre bibliyometrik verileri alır.

#         Args:
#             item_key (str): Zotero item key.

#         Returns:
#             dict or None: Alınan bibliyometrik veriler. Hata durumunda None.
#         """
#         url = f"{self.base_url}/{item_key}"
#         try:
#             response = requests.get(url, headers=self.headers, timeout=10)
#             if response.status_code == 200:
#                 config.logger.info(f"Zotero API'den veri alındı: {item_key}")
#                 return response.json()
#             else:
#                 config.logger.error(f"Zotero API hatası: {response.status_code} {url}")
#                 return None
#         except Exception as e:
#             config.logger.error(f"Zotero API isteğinde hata: {e}")
#             return None

#     def analyze_references(self, reference_list):
#         """
#         Verilen referans listesini analiz eder ve her referans için yazar ismini çıkarır.

#         Args:
#             reference_list (list): Kaynakça referans metinlerinin listesi.

#         Returns:
#             list: Her referans için analiz sonucu içeren sözlüklerin listesi.
#         """
#         analysis_results = []
#         for reference in reference_list:
#             try:
#                 # Yazar ismini regex ile çıkar
#                 match = re.search(r"[A-Z][a-z]+, [0-9]{4}", reference)
#                 author = match.group() if match else "Bilinmeyen"
#                 analysis_results.append({"reference": reference, "author": author})
#             except Exception as e:
#                 config.logger.error(f"Referans analizi hatası: {e}")
#                 analysis_results.append({"reference": reference, "author": "Bilinmeyen"})
#         return analysis_results
