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

ZPT5 CALISTIRILDIGINDA

>  Baştan başlamak için [B], kaldığınız yerden devam için [C],
> güncelleme için [G]: b 2025-02-16 09:02:23 - WARNING - ⚠️ Veritabanı
> sıfırlanıyor... 2025-02-16 09:02:23 - ERROR - ❌ Veritabanı sıfırlama
> hatası: local variable 'collection' referenced before assignment 

hatasi verdi

** BU SORUNU COZDUM.**
kodda, "b" seçeneği sırasında collection ve bib_collection değişkenlerine global olarak erişip yeniden atama yapabilmek için global bildirimi ekledim. Böylece "local variable referenced before assignment" hatası ortadan kalkdi.

**Global Değişkenler:**
collection ve bib_collection global alanda tanımlandığı için, main fonksiyonu içerisinde yeniden atama yapılabilmesi adına global collection, bib_collection ifadesi eklendi.
Bu güncelleme ile veritabanı sıfırlama kısmında yaşanan "local variable referenced before assignment" hatası giderildi.

***kodu incelerken embeding yapilmadigini ana programda embedinge atif yapilmadan bos gecildigini fark ettim.***

**embed_text** fonksiyonu, şu anki kod akışı içerisinde herhangi bir fonksiyon tarafından çağrılmıyor. Yani, tanımlı ancak kullanılmıyor. Eğer embedding işlemlerini dosya işleme sürecine entegre etmek gerekli, örneğin **process_file** fonksiyonu içerisinde metin temizlendikten sonra ya da belirli bir aşamada çağırabilirsiniz.

Önceki versiyonda, **embed_text** fonksiyonunun metin çıkarma işleminin hemen ardından, yani process_file fonksiyonu içerisinde temizlenen metin üzerinde embedding oluşturulması amacıyla çağrılması planlanmıştı. Ancak kodun o kısmında embedding işlemi entegre edilmemişti; yani, embed_text fonksiyonu tanımlanmış fakat process_file veya başka bir fonksiyon içerisinde kullanılmıyordu.

Kısacası, asıl niyet:

Metin çıkarıldıktan ve temizlendikten sonra,
Embedding işlemi yapmak için embed_text çağrılacaktı.
Fakat mevcut kodda bu çağrı yer almıyor. Eğer embedding işlemini eklemek isterseniz, örneğin process_file fonksiyonu içerisinde temiz metin elde edildikten sonra şu şekilde çağırabilirsiniz:

python

    embedding = embed_text(temiz_metin)

Bu, metin üzerinde OpenAI embedding API'sini kullanarak vektör oluşturmanıza olanak tanır.

Aşağıda, process_file fonksiyonu içerisine, temizlenen metin üzerinde embedding işleminin gerçekleştirilmesini ve ChromaDB'ye eklenmesini sağlayan güncellemeyi yaptım. İşte ilgili kısmın güncellenmiş hali:

python

    def process_file(item):
        """Bir dosyayı işleyen ana fonksiyon"""
        try:
            if not isinstance(item, dict):
                raise ValueError("Geçersiz öğe formatı")
                
            title = item.get("title")
            if not title:
                raise ValueError("Başlık bulunamadı")
                
            kaynak_path = os.path.join(STORAGE_DIR, title)
            if not os.path.exists(kaynak_path):
                raise FileNotFoundError(f"Dosya bulunamadı: {kaynak_path}")
                
            hedef_path = os.path.join(HEDEF_DIZIN, f"{os.path.splitext(title)[0]}.temizmetin.txt")
            
            is_pdf = title.lower().endswith('.pdf')
            
            if is_pdf:
                text = extract_text_from_pdf(kaynak_path)
                source_type = "pdf"
            else:
                with open(kaynak_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                source_type = "txt"
            
            if not text:
                raise Exception("Metin çıkarılamadı")
            
            # Temizleme işlemleri
            temiz_metin = clean_text(text)
            
            # Embedding işlemi - REMARK: embed_text fonksiyonu çağrılarak embedding oluşturuluyor.
            embedding = embed_text(temiz_metin)
            if embedding is not None:
                try:
                    # REMARK: Oluşturulan embedding, ChromaDB koleksiyonuna ekleniyor.
                    collection.add(
                        ids=[os.path.splitext(title)[0]],
                        embeddings=[embedding],
                        metadatas=[{'title': title, 'source': source_type, 'timestamp': datetime.now().isoformat()}]
                    )
                    logger.info(f"✅ {title} için embedding başarıyla eklendi.")
                except Exception as e:
                    logger.error(f"Embedding eklenirken hata oluştu: {e}")
            else:
                logger.warning("Embedding oluşturulamadı, sonraki işlemlere geçiliyor.")
            
            tablolar = detect_tables(text)
            
            references = extract_references_enhanced(text)
            if references:
                save_references(references, kaynak_path, source_type)
            
            # Temiz metni dosyaya kaydet
            with open(hedef_path, 'w', encoding='utf-8') as f:
                f.write(temiz_metin)
            
            if tablolar:
                base_name = os.path.splitext(title)[0]
                with open(os.path.join(TEMIZ_TABLO_DIZIN, 'json', f"{base_name}.json"), 'w', encoding='utf-8') as f:
                    json.dump(tablolar, f, indent=2, ensure_ascii=False)
                
                for i, table in enumerate(tablolar, 1):
                    if table['veriler'] and len(table['veriler']) > 0:
                        try:
                            df = pd.DataFrame(table['veriler'][1:], columns=table['veriler'][0])
                            
                            csv_path = os.path.join(TEMIZ_TABLO_DIZIN, 'csv', f"{base_name}_tablo{i}.csv")
                            df.to_csv(csv_path, index=False, encoding='utf-8')
                            
                            excel_path = os.path.join(TEMIZ_TABLO_DIZIN, 'excel', f"{base_name}_tablo{i}.xlsx")
                            df.to_excel(excel_path, index=False)
                        except Exception as e:
                            logger.warning(f"Tablo dönüştürme hatası: {e}")
                            continue
            
            log_entry = {
                'dosya': title,
                'tarih': datetime.now(pytz.timezone('Turkey')).isoformat(),
                'tablo_sayisi': len(tablolar),
                'referans_sayisi': len(references),
                'dosya_tipi': source_type,
                'bellek_kullanim': memory_usage()
            }
            
            return (True, log_entry)
        
        except Exception as e:
            error_log = {
                'dosya': title if 'title' in locals() else 'unknown',
                'hata': str(e),
                'traceback': traceback.format_exc(),
                'zaman': datetime.now().isoformat()
            }
            return (False, error_log)

**REMARK:**
**Embedding İşleminin Eklenmesi:**
**Temiz metin**(temiz_metin) elde edildikten sonra, embed_text(temiz_metin) çağrılarak embedding oluşturuluyor.
Eğer embedding başarılıysa, collection.add() metodu ile ChromaDB koleksiyonuna ekleniyor. Bu ekleme, title (dosya adı) baz alınarak benzersiz bir kimlik (id) oluşturulması şeklinde yapılıyor.
Hata oluşursa veya embedding oluşturulamıyorsa, uygun loglama mesajı veriliyor.
Bu güncelleme ile embedding işlemi, dosya işleme sürecinin uygun bir noktasında çağrılmış ve sonucu veritabanına eklenmiştir.

embeding isleminden once iceriginde tablo, kaynakca bulunmayan temiz text kucuk parcalara ayrilip embedinge gonderiliyordu. su anki durum bunun gibi mi degilse yapalim

Aşağıdaki güncellenmiş kod parçasında, temiz metin önce küçük parçalara bölünüyor, her parça ayrı ayrı embedding'e gönderiliyor ve ChromaDB'ye ekleniyor. Bu sayede, dosyanın tüm içeriği küçük parçalardan oluşan embedding'lere sahip oluyor.

python

    def process_file(item):
        """Bir dosyayı işleyen ana fonksiyon"""
        try:
            if not isinstance(item, dict):
                raise ValueError("Geçersiz öğe formatı")
                
            title = item.get("title")
            if not title:
                raise ValueError("Başlık bulunamadı")
                
            kaynak_path = os.path.join(STORAGE_DIR, title)
            if not os.path.exists(kaynak_path):
                raise FileNotFoundError(f"Dosya bulunamadı: {kaynak_path}")
                
            hedef_path = os.path.join(HEDEF_DIZIN, f"{os.path.splitext(title)[0]}.temizmetin.txt")
            
            is_pdf = title.lower().endswith('.pdf')
            
            if is_pdf:
                text = extract_text_from_pdf(kaynak_path)
                source_type = "pdf"
            else:
                with open(kaynak_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                source_type = "txt"
            
            if not text:
                raise Exception("Metin çıkarılamadı")
            
            # Temizleme işlemleri
            temiz_metin = clean_text(text)
            
            # Parçalara bölme işlemi
            chunks = split_text(temiz_metin, chunk_size=256)  # REMARK: Burada chunk boyutu isteğe bağlı ayarlanabilir.
            chunk_ids = [f"{os.path.splitext(title)[0]}_{i}" for i in range(len(chunks))]
            
            # Her chunk için embedding işlemi
            embeddings = []
            for idx, chunk in enumerate(chunks):
                emb = embed_text(chunk)
                if emb is None:
                    logger.warning(f"Chunk {idx} için embedding oluşturulamadı.")
                    # Boş embedding yerine None ekliyoruz, ya da isteğe bağlı olarak işlemi atlayabilirsiniz.
                    embeddings.append([0.0]*768)  # REMARK: Modelin embedding boyutuna göre ayarlayın.
                else:
                    embeddings.append(emb)
            
            # ChromaDB'ye embedding ekleme
            try:
                collection.add(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    metadatas=[{'title': title, 'source': source_type, 'chunk_index': i, 'timestamp': datetime.now().isoformat()} 
                               for i in range(len(chunks))]
                )
                logger.info(f"✅ {title} için tüm chunk embedding'leri başarıyla eklendi.")
            except Exception as e:
                logger.error(f"Embedding eklenirken hata oluştu: {e}")
            
            # Tablo ve kaynakça işlemleri
            tablolar = detect_tables(text)
            
            references = extract_references_enhanced(text)
            if references:
                save_references(references, kaynak_path, source_type)
            
            # Temiz metni dosyaya kaydet
            with open(hedef_path, 'w', encoding='utf-8') as f:
                f.write(temiz_metin)
            
            if tablolar:
                base_name = os.path.splitext(title)[0]
                with open(os.path.join(TEMIZ_TABLO_DIZIN, 'json', f"{base_name}.json"), 'w', encoding='utf-8') as f:
                    json.dump(tablolar, f, indent=2, ensure_ascii=False)
                
                for i, table in enumerate(tablolar, 1):
                    if table['veriler'] and len(table['veriler']) > 0:
                        try:
                            df = pd.DataFrame(table['veriler'][1:], columns=table['veriler'][0])
                            
                            csv_path = os.path.join(TEMIZ_TABLO_DIZIN, 'csv', f"{base_name}_tablo{i}.csv")
                            df.to_csv(csv_path, index=False, encoding='utf-8')
                            
                            excel_path = os.path.join(TEMIZ_TABLO_DIZIN, 'excel', f"{base_name}_tablo{i}.xlsx")
                            df.to_excel(excel_path, index=False)
                        except Exception as e:
                            logger.warning(f"Tablo dönüştürme hatası: {e}")
                            continue
            
            log_entry = {
                'dosya': title,
                'tarih': datetime.now(pytz.timezone('Turkey')).isoformat(),
                'tablo_sayisi': len(tablolar),
                'referans_sayisi': len(references),
                'dosya_tipi': source_type,
                'bellek_kullanim': memory_usage()
            }
            
            return (True, log_entry)
        
        except Exception as e:
            error_log = {
                'dosya': title if 'title' in locals() else 'unknown',
                'hata': str(e),
                'traceback': traceback.format_exc(),
                'zaman': datetime.now().isoformat()
            }
            return (False, error_log)

**REMARK:**
Metin Parçalama:
split_text fonksiyonu kullanılarak temiz metin küçük parçalara (chunk) bölünüyor. Bu işlem, embedding işlemine girdi olarak verilecek metin uzunluğunu optimize eder.
Embedding İşlemi:

Her bir parça için embed_text çağrılarak embedding oluşturuluyor.
Oluşturulan embedding'ler embeddings listesine ekleniyor.
Eğer bir parça için embedding oluşturulamazsa, varsayılan bir embedding (örneğin sıfırlardan oluşan vektör) ekleniyor. (Burada modelin embedding boyutuna göre ayarlamanız gerekebilir.)
ChromaDB'ye Ekleme:

Her parça için benzersiz bir ID (chunk_ids) oluşturuluyor.
Her parça için ilgili metaveriler (örn. chunk indeksi, dosya başlığı, kaynak türü, timestamp) ekleniyor.
Tüm embedding'ler ve metaveriler, tek seferde ChromaDB koleksiyonuna ekleniyor.
Bu güncelleme ile dosya içeriği önce parçalara ayrılıyor, her parça embedding'e gönderiliyor ve koleksiyona ekleniyor; böylece önceki yöntemle uyumlu hale getiriliyor.

Ama kucuk parcalara bolme fonksiyonum var di. daha once yazmistim.

Aşağıdaki kodda, önceden tanımlı split_text fonksiyonunuzu kullanarak temiz metni küçük parçalara bölüyoruz. Her parça için embedding oluşturulup ChromaDB'ye ekleniyor. Böylece, embedding işlemi önceki yönteminize uygun hale getirilmiş oluyor.

python

    def process_file(item):
        """Bir dosyayı işleyen ana fonksiyon"""
        try:
            if not isinstance(item, dict):
                raise ValueError("Geçersiz öğe formatı")
                
            title = item.get("title")
            if not title:
                raise ValueError("Başlık bulunamadı")
                
            kaynak_path = os.path.join(STORAGE_DIR, title)
            if not os.path.exists(kaynak_path):
                raise FileNotFoundError(f"Dosya bulunamadı: {kaynak_path}")
                
            hedef_path = os.path.join(HEDEF_DIZIN, f"{os.path.splitext(title)[0]}.temizmetin.txt")
            
            is_pdf = title.lower().endswith('.pdf')
            
            if is_pdf:
                text = extract_text_from_pdf(kaynak_path)
                source_type = "pdf"
            else:
                with open(kaynak_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                source_type = "txt"
            
            if not text:
                raise Exception("Metin çıkarılamadı")
            
            # Temizleme işlemleri
            temiz_metin = clean_text(text)
            
            # REMARK: Önceden tanımlı split_text fonksiyonunuz kullanılarak metin küçük parçalara bölünüyor.
            chunks = split_text(temiz_metin, chunk_size=256)
            chunk_ids = [f"{os.path.splitext(title)[0]}_{i}" for i in range(len(chunks))]
            
            # Her chunk için embedding işlemi
            embeddings = []
            for idx, chunk in enumerate(chunks):
                emb = embed_text(chunk)
                if emb is None:
                    logger.warning(f"Chunk {idx} için embedding oluşturulamadı.")
                    # REMARK: Modelin embedding boyutuna göre sıfır vektör ekleniyor.
                    embeddings.append([0.0]*768)
                else:
                    embeddings.append(emb)
            
            # REMARK: Tüm chunk'ler için embedding'ler ChromaDB'ye ekleniyor.
            try:
                collection.add(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    metadatas=[{'title': title, 'source': source_type, 'chunk_index': i, 'timestamp': datetime.now().isoformat()} 
                               for i in range(len(chunks))]
                )
                logger.info(f"✅ {title} için tüm chunk embedding'leri başarıyla eklendi.")
            except Exception as e:
                logger.error(f"Embedding eklenirken hata oluştu: {e}")
            
            # Tablo ve kaynakça işlemleri
            tablolar = detect_tables(text)
            references = extract_references_enhanced(text)
            if references:
                save_references(references, kaynak_path, source_type)
            
            # Temiz metni dosyaya kaydet
            with open(hedef_path, 'w', encoding='utf-8') as f:
                f.write(temiz_metin)
            
            if tablolar:
                base_name = os.path.splitext(title)[0]
                with open(os.path.join(TEMIZ_TABLO_DIZIN, 'json', f"{base_name}.json"), 'w', encoding='utf-8') as f:
                    json.dump(tablolar, f, indent=2, ensure_ascii=False)
                for i, table in enumerate(tablolar, 1):
                    if table['veriler'] and len(table['veriler']) > 0:
                        try:
                            df = pd.DataFrame(table['veriler'][1:], columns=table['veriler'][0])
                            csv_path = os.path.join(TEMIZ_TABLO_DIZIN, 'csv', f"{base_name}_tablo{i}.csv")
                            df.to_csv(csv_path, index=False, encoding='utf-8')
                            excel_path = os.path.join(TEMIZ_TABLO_DIZIN, 'excel', f"{base_name}_tablo{i}.xlsx")
                            df.to_excel(excel_path, index=False)
                        except Exception as e:
                            logger.warning(f"Tablo dönüştürme hatası: {e}")
                            continue
            
            log_entry = {
                'dosya': title,
                'tarih': datetime.now(pytz.timezone('Turkey')).isoformat(),
                'tablo_sayisi': len(tablolar),
                'referans_sayisi': len(references),
                'dosya_tipi': source_type,
                'bellek_kullanim': memory_usage()
            }
            
            return (True, log_entry)
        
        except Exception as e:
            error_log = {
                'dosya': title if 'title' in locals() else 'unknown',
                'hata': str(e),
                'traceback': traceback.format_exc(),
                'zaman': datetime.now().isoformat()
            }
            return (False, error_log)

**REMARK:**
Metin Parçalama:

**split_text** fonksiyonunuz, temiz_metin üzerinde çağrılarak metni küçük parçalara ayırıyor.
Her parçaya benzersiz bir ID oluşturuluyor (chunk_ids).
**Embedding İşlemi:**

Her bir küçük metin parçası için embed_text fonksiyonu kullanılarak embedding oluşturuluyor.
Eğer bir parça için embedding oluşturulamazsa, modelin embedding boyutuna uygun (örneğin 768) sıfır vektör ekleniyor.
**ChromaDB'ye Ekleme:**

Tüm chunk embedding'leri ve ilgili metaveriler, tek seferde ChromaDB koleksiyonuna ekleniyor.
Bu güncelleme, önceki yönteminizdeki küçük parçalara bölme işlemiyle embedding oluşturma mantığını korumaktadır.

Temizlenen metin icerisinden kaynakca tablolar sekiller resimler cikarilmis sekilde mi embedinge gidiyor. onceden bu sekilde bir sistem vardi

Şu anki yapı, temiz metni doğrudan alıp, önceden belirlediğimiz küçük parçalara bölüyor; yani referanslar, tablolar, şekiller ve resimler gibi bölümler zaten ayrı olarak çıkartılabiliyor (örneğin, detect_tables ve extract_references_enhanced ile) fakat embedinge gönderilen metin, clean_text fonksiyonuyla temel temizlik yapıldıktan sonra tüm içerik olarak kullanılıyor.

Önceki sistemde, bu öğelerin embedinge gitmeden önce metinden çıkarılması (örneğin, sadece düz metin üzerinden embedding oluşturulması) sağlanıyordu. Eğer embedinge gönderilecek metnin bu bölümlerden arındırılmış olmasını istiyorsanız, clean_text fonksiyonunu veya yeni bir filtreleme aşamasını ekleyerek, referansları, tabloları, şekilleri ve resimleri metinden ayıklamanız gerekir.

## Özetle:##

Mevcut sistemde: Tüm temizlenmiş metin, küçük parçalara bölünüp embedding'e gönderiliyor.
Önceki sistemde: Bu öğeler embedinge gitmeden önce metinden çıkarılıyordu.
Gerekirse, embedinge gitmeden önce bu bölümleri ayıklayacak ek bir adım ekleyebiliriz.

> sonucta chtgpt den yardim aldim ve kafasina gore programin anasini
> aglati.  daha efendice yazmayacagim. sorin cozmesi guzel ama prompta
> soylesende soylemesende kafasina gore fonksiyon atip fonksiyon adi
> degistirip her boka musallat olabiliyor.
=======
>>>>>>> 9934da9f16f370b2da4e303cda39d19f3e371f8f


## ZPT8 ##
Aşağıda önceki sürümde vardıktan sonra güncelleme sırasında çıkarılan, değiştirilmiş ve eklenen fonksiyonları özetliyorum:

### Var Olan ve Kısmen Güncellenen Fonksiyonlar

-   **init_dirs:**
    
    -   Eski sürümde dizin oluşturma işlemleri vardı. Aynı işlevi görüyor ancak dizin isimleri ve oluşturulacak ek klasörler (ör. JSON, CSV, Excel alt dizinleri) bu sürümde daha net tanımlandı.
-   **memory_usage:**
    
    -   Bellek kullanımını hesaplama fonksiyonu, önceki sürümde de vardı.
-   **clean_text:**
    
    -   Temizlik işlemi yapıyor; önceki sürümde vardı ancak bu sürümde satır sonlarının ve boşlukların düzenlenmesinde küçük iyileştirmeler yapıldı.
-   **split_text:**
    
    -   Metni parçalara bölme işlemi, önceki sürümde vardı. Bu sürümde paragraflara bölünme mantığı eklenerek (önce paragraflara ayrılıyor, ardından kelime sayısına göre bölünüyor) iyileştirildi.
-   **embed_text:**
    
    -   OpenAI API’si kullanılarak embedding oluşturma fonksiyonu, daha önce de vardı; hata yönetimi ve loglama eklemeleriyle güncellendi.
-   **fetch_zotero_metadata:**
    
    -   Zotero’dan bibliyografik bilgi çekme işlemi, önceki sürümde de mevcuttu.
-   **save_last_processed_index** ve **get_last_processed_index:**
    
    -   İşlenmiş dosya indekslerinin kaydedilip okunması, önceki sürümde de vardı.
-   **process_file:**
    
    -   Temel dosya işleme fonksiyonu; önceki sürümde PDF’den ham metin çıkarma, temizleme, tablo ve kaynakça işlemleri vardı. Bu sürümde, sütun tespiti (reflow_columns çağrısı), ham metin içerisinden ekstraların (tablolar, resimler, referanslar) ayrılması, temiz metnin wrap olmadan kaydedilmesi, metnin küçük parçalara bölünüp embedding işlemi ve bu embedding’lerin ChromaDB’ye eklenmesi gibi işlemler eklenerek önemli ölçüde güncellendi.
-   **main:**
    
    -   Ana akış fonksiyonu; kullanıcı seçimlerine göre işleyişi başlatan kısım, önceki sürümde vardı ancak bu sürümde yeni işlevler (ör. reflow_columns, save_text_file gibi ek fonksiyonlar) ve hata yönetimiyle genişletildi.

### Yeni Eklenen Fonksiyonlar

-   **save_text_file:**
    
    -   Belirtilen dizine, verilen dosya adını kullanarak içerik kaydetmeyi standartlaştırmak için eklendi.
-   **reflow_columns:**
    
    -   Çok sütunlu PDF metinlerinde, özellikle bilimsel yayınlarda özet ve asıl metin farklı sütun düzeninde olduğunda, akışı düzeltmek amacıyla eklendi. Bu fonksiyon, metindeki sütun yapısını tespit edip, tek sütunlu akışa dönüştürme amaçlı basit bir algoritma içeriyor.
-   **detect_tables:**
    
    -   PDF’den tablo çıkarımını doğrudan yapmaya yönelik fonksiyon, önceki sürümde metin içerisinden tablo tespiti yapılıyordu. Bu sürümde PDF üzerinden doğrudan tablo çıkarımı için basit yöntemler eklenerek geliştirildi.
-   **extract_references:**
    
    -   Önceki sürümde "extract_references_enhanced" şeklinde tanımlanmıştı. Bu sürümde, kaynakça çıkarımına yönelik mantık daha kısa ve net bir şekilde "extract_references" fonksiyonu altında toplandı.

----------

Bu özet, önceki sürüme göre hangi fonksiyonların var olduğunu, hangilerinin güncellendiğini ve yeni eklenen fonksiyonları göstermektedir.