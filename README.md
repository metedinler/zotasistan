# ZOTASISTAN

ZPT4'UN  kodunda temel olarak PDF'lerden metin çıkarma, metin temizleme, tabloların tespiti ve referansları tanımlayıp ayrıştırma gibi birçok işlem yer alıyor. 
Ancak kodun henüz tamamlanmadı. 
İşlem akışını düzgün çalıştırabilmek için aşağıdaki geliştirmeleri ve düzeltmeleri yapmanız gerekebilir:

# 1. Eksik Tanımlar ve Hatalar: 
**ZOTERO_API_KEY** ve **ZOTERO_USER_ID** tanımlanmadan kullanılmış. Kodda yukarıda bu değişkenler yüklenmeden kullanıldığı için bir NameError oluşacaktır.

response adında bir değişken kullanılmaya çalışılmış ancak tanımlanmamış. Bu, Zotero API'den yanıt almak için bir istek yapmanız gerektiğini gösteriyor.

Düzeltme: python

    response = requests.get(ZOTERO_API_URL, headers={"Authorization": f"Bearer {ZOTERO_API_KEY}"})

# 2. Fonksiyon Tamamlama:
**extract_references_enhanced** fonksiyonu yarım kalmış. Kaynakçaları metinden daha iyi tespit etmek için tam bir algoritma eklemelisiniz. Örneğin:

python

    def extract_references_enhanced(text):
       references = []
        ref_patterns = [
         r'(?i)(?:KAYNAKÇA|REFERENCES|BIBLIOGRAPHY).*?\n(.*?)(?=\n\s*\n|\Z)',
          r'\[\d+\] .*?(?=\n|$)',  # [1] formatındaki referanslar
       ]
    	   for pattern in ref_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                references.append(match.group(0).strip())
      return references
    
# 3. Zotero API Hatası Kontrolü:

Zotero API çağrısından başarılı yanıt alıp almadığınızı kontrol ederken return yerine bir hata mesajı döndürmelisiniz.

python

    if response.status_code != 200:
        logger.error(f"Zotero'dan veri alınamadı: {response.status_code}")
        return None  # Veya hatayı işlemeye devam edebilirsiniz
    
# 4. ChromaDB Hatası Yönetimi:

ChromaDB bağlantısı başarısız olduğunda **chroma_client = None** yerine daha açıklayıcı bir hata mesajı verebilirsiniz ve programın devam edip edemeyeceğini kontrol edebilirsiniz.

# 5. Bellek ve İşlem Durumu İzleme:

Kodun tamamı yüksek bellek kullanımına veya işlem yüküne sebep olabileceği için **memory_usage** fonksiyonunu belirli adımlarda kullanmak faydalı olacaktır. Ayrıca, tqdm kullanarak işlem ilerleme çubuğunu ekleyebilirsiniz.

# 6. Dizinleri ve Dosya Yollarını Kontrol Etme:

**STORAGE_DIR, SUCCESS_DIR, KAYNAK_DIZIN** gibi ortam değişkenlerinin tanımlanmış olup olmadığını kontrol etmek ve yoksa varsayılan dizinler kullanmak kodun kararlı çalışmasını sağlayacaktır.

# 7. Traceback Hata Yönetimi:

PDF çıkarma sırasında bir hata oluşursa **traceback** ile hata ayıklamayı daha verimli hale getirebilirsiniz:

python 

    except Exception as e:
        logger.error(f"❌ PDF metni çıkarılamadı: {pdf_path}, Hata: {e}")
        traceback.print_exc()
    
# Açıklamalar
**Ham Metin & Akış Düzeltme:**

**extract_text_from_pdf** : PDF’den ham metni çıkarır.
**reflow_columns** : fonksiyonu, özellikle bilimsel yayınlarda özet ve asıl metin farklı sütun düzeninde ise akışı düzeltir.
# Tablo, Kaynakça, Şekil/Resim Çıkarma:

**detect_tables** PDF’den tablo, şekil ve figürleri tespit eder.
**extract_references_enhanced** optimize edilmiş kaynakça çıkarımını yapar.

# Temiz Metin & Parçalama:

**clean_text** fonksiyonu metni temizler, wrap olmadan kaydeder.
**split_text** metni önce paragraflara bölüp, ardından belirli boyutta parçalara ayırır.

# Embedding & ChromaDB:

Her küçük parça (chunk), embed_text fonksiyonu ile embedding’e dönüştürülür.
Her chunk, ayrı dosya olarak **EMBEDDING_DIR** içerisine kaydedilir ve topluca **ChromaDB**’ye eklenir.

# Zotero Entegrasyonu:

**fetch_zotero_metadata** ile Zotero’dan bibliyografik veri çekilir ve dummy embedding ile **bib_collection** içerisine eklenir.

## Veritabanı Sıfırlama:

Kullanıcı [B] seçerse, collection.delete(where={}) ve bib_collection.delete(where={}) ile tüm kayıtlar temizlenir.

## Hata Yönetimi & Takip:

Her adımda hata yönetimi, loglama ve sayaçlar ile işlem durumu konsola yansıtılır.
Bu program, önceki tüm konuşmalarımızdaki tüm işlevsellikleri birleştirip, detaylı hata yönetimi ve izleme özellikleriyle tam olarak çalışacak şekilde tasarlanmıştır.

## ZPT 5 programinda degisiklikler  ##
 ** Açıklamalar (REMARK): **

1.  **Ortam Değişkenleri ve Zotero Tanımlamaları:**
    
    -   `ZOTERO_API_KEY` ve `ZOTERO_USER_ID` yüklenip, tanımlanmamışsa hata loglanıyor.
    -   Global düzeyde `response` kontrolü kaldırıldı çünkü geçerli bir istek yapılmıyordu.
2.  **Embedding İşlemleri:**
    
    -   `get_embedding` fonksiyonu kaldırılarak tüm embedding işlemi `embed_text` içinde yapıldı.
    -   `embed_text` fonksiyonunda hata durumunda `embedding_failed_count` artıyor ve hata loglanıyor.
3.  **Referans Çıkarma:**
    
    -   `extract_references_enhanced` fonksiyonu, iki farklı pattern kullanarak referansları çekiyor ve temizlenmiş referansları döndürüyor.
4.  **Genel Hata Yönetimi:**
    
    -   PDF metin çıkarma, Zotero API isteği ve tablo dönüştürme işlemlerinde try/except blokları ile detaylı hata loglama eklendi.
    -   Traceback bilgileri `traceback.print_exc()` ile çıktı veriliyor.
5.  **Dizin Oluşturma:**
    
    -   `init_dirs` fonksiyonunda `PDF_DIR` ve `EMBEDDING_PARCA_DIR` gibi dizinler string olarak oluşturulması sağlandı.

Bu versiyonda ana mantığa dokunulmadan hata yönetimi ve embedding bölümleri düzenlendi.
