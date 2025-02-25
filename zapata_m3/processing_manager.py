import os
from pathlib import Path
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from config_module import config
from zotero_module import dokuman_id_al, fetch_zotero_metadata
from pdf_processing import extract_text_from_pdf, reflow_columns, map_scientific_sections_extended, detect_columns
from robust_embedding_module import robust_embed_text
from citation_mapping_module import map_citations  # Citation mapping modülündeki ana fonksiyon
from file_save_module import save_citation_file
from helper_module import shorten_title, stack_guncelle

def pdf_txt_isle(dosya_yolu):
    """
    PDF veya TXT dosyalarını işleyerek; 
      - Ham metni çıkarır,
      - reflow_columns ile temizler,
      - Bilimsel bölümleri (map_scientific_sections_extended) ve sütun yapısını (detect_columns) belirler,
      - Tablo ve kaynakça çıkarımını (örneğin, extract_references_enhanced) gerçekleştirir,
      - Robust embedding işlemlerini (robust_embed_text) uygular,
      - Citation Mapping'i (map_citations) çağırır ve elde edilen verileri sonuç sözlüğüne ekler.
    
    İş akışı:
      1. Dosya tipi belirlenir (PDF/TXT).
      2. Metin çıkarılır (PDF için extract_text_from_pdf, TXT için doğrudan okuma).
      3. reflow_columns ile metin temizlenir.
      4. map_scientific_sections_extended ile bilimsel bölümler haritalanır.
      5. detect_columns ile sütun yapısı belirlenir.
      6. (Varsa) tablo ve kaynakça çıkarım işlemleri yapılır.
      7. Temiz metin üzerinden robust embedding işlemi gerçekleştirilir.
      8. Citation Mapping, map_citations fonksiyonu ile çağrılır.
      9. Elde edilen tüm veriler, bir sonuç sözlüğü halinde döndürülür.
      10. İşlem başlangıcında dosya adı stack'e eklenip, işlem sonunda kaldırılır.
    
    Args:
        dosya_yolu (Path): İşlenecek dosyanın yolu.
    
    Returns:
        dict veya None: İşlem başarılı ise dosyaya ait tüm verileri içeren sözlük; hata durumunda None.
    """
    try:
        # Stack'e ekle
        stack_guncelle(dosya_yolu.name, "ekle")
        config.logger.info(f"{dosya_yolu.name} işleme başladı.")
        
        ext = dosya_yolu.suffix.lower()
        if ext == ".pdf":
            ham_metin = extract_text_from_pdf(dosya_yolu, method='pdfminer')
        elif ext == ".txt":
            with open(dosya_yolu, "r", encoding="utf-8") as f:
                ham_metin = f.read()
        else:
            config.logger.error(f"Desteklenmeyen dosya uzantısı: {dosya_yolu}")
            return None
        
        if not ham_metin:
            raise ValueError("Ham metin çıkarılamadı.")
        
        # Reflow ve gelişmiş temizlik (clean_advanced_text fonksiyonu helper_module'da mevcut olabilir)
        temiz_metin = reflow_columns(ham_metin)
        
        # Bilimsel bölümlerin haritalanması
        bolum_haritasi = map_scientific_sections_extended(ham_metin)
        
        # Sütun yapısının belirlenmesi
        sutun_bilgisi = detect_columns(ham_metin)
        
        # (Opsiyonel) Tablo ve kaynakça çıkarımı - bu adım için ilgili fonksiyonlar varsa çağrılabilir.
        # Örneğin: kaynakca = extract_references_enhanced(ham_metin)
        # Burada kaynakça çıkarımı modüler yapıya entegre edilmişse, onun çıktısı da alınabilir.
        kaynakca = []  # Varsayılan boş liste; gerçek çıkarım fonksiyonu eklenebilir.
        tablolar = []   # Benzer şekilde tablolar çıkarımı yapılabilir.
        
        # Robust embedding işlemleri: Temiz metni chunk'lara bölüp her chunk için embedding oluştur.
        # Örneğin, split_text fonksiyonu embedding_module.py içinde bulunuyor.
        from embedding_module import split_text  # İlgili modülden import ediyoruz.
        chunks = split_text(temiz_metin, chunk_size=256)
        embedding_list = []
        for idx, chunk in enumerate(chunks):
            emb = robust_embed_text(chunk, 
                                    pdf_id=shorten_title(dokuman_id_al(dosya_yolu.name) or dosya_yolu.stem, 80),
                                    chunk_index=idx,
                                    total_chunks=len(chunks))
            embedding_list.append(emb)
        
        # Citation Mapping: Temiz metin üzerinden cümle bazında atıf ifadeleri eşleştirmesi yapılır.
        citation_mapping = map_citations(temiz_metin, bibliography=kaynakca, section_info=bolum_haritasi)
        
        # Citation Mapping dosyası kaydedilir
        citation_file = save_citation_file(dosya_yolu.name, citation_mapping)
        
        # Zotero entegrasyonu: Dosya adı üzerinden temel ID alınır, bibliyometrik veriler çekilir.
        file_id = dokuman_id_al(dosya_yolu.name) or dosya_yolu.stem
        file_id = shorten_title(file_id, 80)
        zotero_meta = fetch_zotero_metadata(file_id)
        
        # Sonuç sözlüğü oluşturulur
        result = {
            "dosya": dosya_yolu.name,
            "ham_metin": ham_metin,
            "temiz_metin": temiz_metin,
            "bolum_haritasi": bolum_haritasi,
            "sutun_bilgisi": sutun_bilgisi,
            "tablolar": tablolar,
            "kaynakca": kaynakca,
            "embedding": embedding_list,
            "citation_mapping": citation_mapping,
            "citation_file": str(citation_file) if citation_file else None,
            "zotero_meta": zotero_meta,
            "islem_tarihi": datetime.now().isoformat()
        }
        
        # Stack'ten kaldır
        stack_guncelle(dosya_yolu.name, "sil")
        config.logger.info(f"{dosya_yolu.name} başarıyla işlendi.")
        
        return result
    except Exception as e:
        config.logger.error(f"{dosya_yolu.name} işlenirken hata: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # Örnek test: STORAGE_DIR altındaki dosyaları işlemek
    from config_module import config
    import glob
    dosyalar = glob.glob(str(config.STORAGE_DIR / "*.pdf"))  # PDF dosyalarını örnek olarak işleyelim
    for dosya in dosyalar:
        result = pdf_txt_isle(Path(dosya))
        if result:
            print(f"İşlem tamamlandı: {result['dosya']}")
        else:
            print(f"Hata oluştu: {dosya}")
