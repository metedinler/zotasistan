import layoutparser as lp
import fitz  # PyMuPDF kütüphanesi
import json
import os

def haritala_bilimsel_yayin_yapisi(pdf_dosya_yolu, cikti_dosyasi_yolu):
    """
    Bilimsel bir yayının PDF dosyasını analiz ederek sütun ve tablo yapısını haritalar.

    Fonksiyon, PDF dosyasının her sayfasını inceler, sütun yapılarını, tabloları ve
    diğer blok seviyesi öğeleri tespit eder ve bu bilgileri bir JSON dosyasına kaydeder.

    Args:
        pdf_dosya_yolu (str): Analiz edilecek PDF dosyasının yolu.
        cikti_dosyasi_yolu (str): Yapı bilgilerinin kaydedileceği dosyanın yolu (JSON).

    Returns:
        None
        Yapı bilgileri belirtilen JSON dosyasına kaydedilir.
    """

    model = lp.models.PaddleDetectionLayoutModel(config_path="lp://PP-OCRv3/ppyolov2_r50vd_dcn_365e_publaynet_infer",
                                        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure", 5:"Formula", 6:"Caption", 7:"Page-Number", 8:"Section-Header", 9:"Footnote", 10:"Body-Text"})

    doc = fitz.open(pdf_dosya_yolu)
    yayin_yapisi = []

    for sayfa_numarasi in range(doc.page_count):
        sayfa = doc.load_page(sayfa_numarasi)
        sayfa_boyutlari = sayfa.rect
        goruntuler = sayfa.get_images(full=True)

        # Sayfayı görüntüye dönüştürme (layout-parser için)
        sayfa_goruntusu = sayfa.get_pixmap(matrix=fitz.Matrix(300/72, 300/72)) # Yüksek çözünürlük için DPI 300
        sayfa_goruntusu.save(f"sayfa_{sayfa_numarasi}.png") # Geçici görüntü dosyası
        sayfa_goruntu_lp = lp.িও.load_image(f"sayfa_{sayfa_numarasi}.png")


        # Layout Parser ile düzen analizi
        layout = model.detect(sayfa_goruntu_lp)


        sayfa_bilgisi = {
            "sayfa_numarasi": sayfa_numarasi + 1,
            "sayfa_boyutlari": {
                "genislik": sayfa_boyutlari.width,
                "yukseklik": sayfa_boyutlari.height
            },
            "bloklar": []
        }

        for blok in layout:
            blok_bilgisi = {
                "tip": blok.type,
                "sinirlar": blok.coordinates, # [x_min, y_min, x_max, y_max]
                "metin": "" # Metin içeriği başlangıçta boş, aşağıda eklenecek
            }

            # Blok tipine göre metin çıkarma (Şimdilik sadece 'Text' ve 'Table' için)
            if blok.type in ['Text', 'Table']:
                alan = [blok.coordinates[0], blok.coordinates[1], blok.coordinates[2], blok.coordinates[3]]
                metin_bloklari = sayfa.get_text("blocks", clip=alan)
                blok_metni = ""
                for metin_blok in metin_bloklari:
                    blok_metni += metin_blok[4] # Metin içeriği
                blok_bilgisi["metin"] = blok_metni.strip()


            sayfa_bilgisi["bloklar"].append(blok_bilgisi)

        yayin_yapisi.append(sayfa_bilgisi)
        os.remove(f"sayfa_{sayfa_numarasi}.png") # Geçici görüntü dosyasını sil

    # Yapı bilgilerini JSON dosyasına kaydet
    with open(cikti_dosyasi_yolu, 'w', encoding='utf-8') as f:
        json.dump(yayin_yapisi, f, ensure_ascii=False, indent=4)

    print(f"Bilimsel yayın yapısı analiz edildi ve '{cikti_dosyasi_yolu}' dosyasına kaydedildi.")


# Fonksiyonun Kullanım Örneği
pdf_dosya = "örnek_bilimsel_yayin.pdf" # PDF dosyasının yolu
cikti_json_dosyasi = "yayin_yapisi.json" # Çıktı JSON dosyasının yolu

# Örnek bir PDF dosyası oluşturma (test için - gerçek bir PDF dosyası ile değiştirin)
if not os.path.exists(pdf_dosya):
    print(f"'{pdf_dosya}' dosyası bulunamadı. Lütfen geçerli bir PDF dosyası yolu belirtin.")
else:
    haritala_bilimsel_yayin_yapisi(pdf_dosya, cikti_json_dosyasi)