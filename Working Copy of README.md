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

| Fonksiyon Adı | ZPT9.py | ZPT4.py | ZPT5.py | Çağrıldığı Fonksiyonlar | Gelişim Durumu |
|---------------|---------|----------|----------|------------------------|----------------|
| init_dirs() | ✓ | ✓ | ✓ | main() | Sabit/Temel |
| memory_usage() | ✓ | ✓ | ✓ | process_file(), log kayıtları | Sabit/Temel |
| clean_text() | ✓+ | ✓ | ✓ | process_file() | ZPT9'da gelişmiş |
| extract_text_from_pdf() | ✓+ | ✓ | ✓ | process_file() | ZPT9'da gelişmiş |
| detect_tables() | ✓+ | ✓ | ✓ | process_file() | ZPT9'da gelişmiş |
| extract_references_enhanced() | ✓+ | ✓ | ✓ | process_file() | ZPT9'da çok gelişmiş |
| embed_text() | ✓+ | ✕ | ✓ | process_file() | ZPT9'da gelişmiş |
| fetch_zotero_metadata() | ✓+ | ✓ | ✓ | process_file() | ZPT9'da hata yönetimi eklenmiş |
| process_file() | ✓+ | ✓ | ✓ | main() | ZPT9'da en kapsamlı |
| save_references() | ✓ | ✓ | ✓ | process_file() | Benzer yapıda |
| print_status() | ✓ | ✓ | ✓ | main() | Benzer yapıda |

  

4.  **Embedding İşlemleri**

-   ZPT4.py'da embedding işlemi yok
-   ZPT5.py'da temel embedding desteği
-   ZPT9.py'da chunk'lara bölme ve paralel işleme desteği

### İşlevsellik Analizi

**En İşlevsel Fonksiyonlar:**

1.  `extract_references_enhanced()`  (ZPT9.py)
    
    -   Çoklu format desteği
    -   Akıllı bölüm tespiti
    -   Gelişmiş temizleme
2.  `process_file()`  (ZPT9.py)
    
    -   Kapsamlı hata yönetimi
    -   Paralel işleme desteği
    -   Detaylı loglama

**Geliştirilmesi Gereken Fonksiyonlar:**

1.  `detect_tables()`  (Tüm versiyonlar)
    
    -   Karmaşık tablo yapıları için yetersiz
    -   OCR desteği eksik
2.  `embed_text()`  (ZPT5.py, ZPT9.py)
    
    -   Rate limiting yönetimi eksik
    -   Alternatif model desteği yok

### Önemli İyileştirmeler

1.  **Hata Yönetimi**

-   ZPT9.py'da try-except blokları daha granüler
-   Detaylı hata loglaması eklenmiş

2.  **Performans**

-   ZPT9.py'da multiprocessing desteği
-   Bellek yönetimi iyileştirilmiş

3.  **Modülerlik**

-   ZPT9.py'da fonksiyonlar daha modüler
-   Tekrar kullanılabilirlik artırılmış

### Öneriler

1.  **Kod İyileştirmeleri:**

-   Daha fazla type hinting eklenebilir
-   Async/await yapısına geçilebilir
-   Unit test coverage artırılabilir

2.  **Fonksiyon Geliştirmeleri:**

-   Table detection için ML modeli entegre edilebilir
-   Embedding için farklı model seçenekleri eklenebilir
-   Referans çıkarma için akademik format desteği genişletilebilir

3.  **Performans İyileştirmeleri:**

-   Caching mekanizması eklenebilir
-   Streaming processing için destek eklenebilir
-   Bellek optimizasyonu yapılabilir
```markdown
| Fonksiyon | Amaç | Girdiler | Çıktılar | Çağrıldığı Fonks. | Versiyon ||-----------|------|----------|-----------|-------------------|-----------|| init_dirs() | Dizin yapısını oluşturur | - | - | main() | Tüm vers. || extract_text_from_pdf() | PDF'den metin çıkarır | pdf_path | text | process_file() | PDF vers. || clean_text() | Metni temizler | text | cleaned_text | process_file() | Tüm vers. || detect_tables() | Tablo tespiti yapar | text | tables[] | process_file() | Tüm vers. || split_text() | Metni chunklara böler | text, size | chunks[] | process_file() | PDF vers. || get_embedding() | Embedding oluşturur | text | vector[] | process_file() | PDF vers. |
```

Select a file

Apply

Bu analiz oldukça kapsamlı ve devam ediyor. İsterseniz:

Haklısınız, daha detaylı bir analiz yapayım. İlk olarak tüm programların temel fonksiyonlarını analiz eden kapsamlı bir tablo oluşturacağım:

### 1. Temel Fonksiyonlar Analiz Tablosu19  lines▼
```markdown

```
| Fonksiyon | Amaç | Girdiler | Çıktılar | Çağrıldığı Fonks. | Versiyon |
|-----------|------|----------|-----------|-------------------|-----------|
| init_dirs() | Dizin yapısını oluşturur | - | - | main() | Tüm vers. |
| extract_text_from_pdf() | PDF'den metin çıkarır | pdf_path | text | process_file() | PDF vers. |
| clean_text() | Metni temizler | text | cleaned_text | process_file() | Tüm vers. |
| detect_tables() | Tablo tespiti yapar | text | tables[] | process_file() | Tüm vers. |
| split_text() | Metni chunklara böler | text, size | chunks[] | process_file() | PDF vers. |
| get_embedding() | Embedding oluşturur | text | vector[] | process_file() | PDF vers. |

| Fonksiyon | Amaç | Girdiler | Çıktılar | Global Değişkenler | Çağrıldığı Fonks. | Program |
|-----------|------|----------|-----------|-------------------|-------------------|----------|
| init_dirs() | Dizin yapısını oluşturur | - | bool | HEDEF_DIZIN, TEMIZ_TABLO_DIZIN, TEMIZ_KAYNAKCA_DIZIN | main() | zottxt30, zpt4, zpt5 |
| memory_usage() | Bellek kullanımını ölçer | - | str (MB) | - | process_file(), main() | Tüm versiyonlar |
| clean_text() | Metni temizler | text:str | str | - | process_file() | Tüm versiyonlar |
| detect_tables() | Tablo tespiti yapar | text:str | list[dict] | - | process_file() | zottxt30, zpt4, zpt5 |
| extract_references() | Kaynakça çıkarır | text:str | str/None | - | process_file() | Tüm versiyonlar |
| extract_references_enhanced() | Gelişmiş kaynakça çıkarır | text:str | list[str] | - | process_file() | zpt9 |
| process_file() | Dosya işleme | item/dosya_adi:str | tuple(bool,dict) | success_count, embedding_failed_count | main() | Tüm versiyonlar |
| split_text() | Metni parçalara böler | text:str, chunk_size:int | list[str] | - | process_file() | zpt9, zpt4, zpt5 |
| embed_text()/get_embedding() | Embedding oluşturur | text:str | list[float]/None | embedding_failed_count | process_file() | zpt9, zpt4, zpt5 |
| save_references() | Referansları kaydeder | references:list, path:str | bool | - | process_file() | Tüm versiyonlar |
| fetch_zotero_metadata() | Zotero verisi çeker | item_key:str | dict/None | ZOTERO_API_KEY | process_file() | zpt9, zpt4, zpt5 |
| save_last_processed_index() | İlerlemeyi kaydeder | index:int | - | - | main() | zpt9, zpt4, zpt5 |
| get_last_processed_index() | İlerlemeyi okur | - | int | - | main() | zpt9, zpt4, zpt5 |
| print_status() | Durum gösterir | current:int, total:int, vs | - | - | main() | zpt9, zpt4, zpt5 |
| save_log() | Log kaydeder | entry:dict | - | LOG_DOSYASI | process_file() | zottxt30 |
| save_error() | Hata loglar | entry:dict | - | HATA_LOG_DOSYASI | process_file() | zottxt30 |
| main() | Ana program akışı | - | - | Tüm globals | - | Tüm versiyonlar |

| Fonksiyon | Input | Output | Global Değişkenler | Özellikler | Çağrıldığı Yer | Program |
|-----------|-------|--------|-------------------|-------------|----------------|----------|
| wrapped_text_fix() | text:str | str | - | Satır sonunda kesilen kelimeleri birleştirir | process_file() | zottxt24, zottxt25 |
| sutunlu_yapi_tespit() | text:str | bool | - | Metinde sütun yapısı olup olmadığını kontrol eder | process_file() | zottxt24, zottxt25 |
| metin_tablo_cikar() | text:str, doc_id:str | list[dict] | - | Metinden tablo yapılarını çıkarır ve yapılandırır | process_file() | zottxt24, zottxt25 |
| dosya_listesini_guncelle() | - | list[str] | LOG_DOSYASI | İşlenecek dosyaları tespit eder ve günceller | main() | zottxt24, zottxt25 |
| kalan_dosyalari_oku() | - | list[str] | LOG_DOSYASI | İşlenmemiş dosyaları belirler | main() | zottxt24, zottxt25 |
| stack_yukle() | - | set[str] | STACK_DOSYASI | İşlem yığınını yükler | main() | zottxt24, zottxt25 |
| stack_guncelle() | dosya_adi:str, islem_tipi:str | - | STACK_DOSYASI | İşlem yığınını günceller | process_file() | zottxt24, zottxt25 |
| pdf_isle() | pdf_path:str | tuple(str,list) | - | PDF'den metin ve tablo çıkarır | dosya_isle() | zottxt29 |
| txt_isle() | dosya_adi:str | tuple(str,list) | - | TXT dosyasından metin okur | dosya_isle() | zottxt29 |
| dokuman_id_al() | dosya_adi:str | str | - | Dosya adından ID çıkarır | process_file() | zottxt24, zottxt25 |
| log_guncelle() | dosya_adi:str, durum:dict | - | LOG_DOSYASI | İşlem loglarını günceller | process_file() | zottxt24, zottxt25 |
| baslik_yaz() | - | - | - | Program başlığını yazdırır | main() | zottxt29 |
| dizinleri_hazirla() | - | - | TEMIZ_TABLO_DIZIN | Alt dizinleri oluşturur | init_dirs() | zottxt24, zottxt25 |
| shorten_title() | title:str, max_length:int | str | - | Uzun başlıkları kısaltır | process_file() | zpt4, zpt5, zpt9 |
| embedding_exists() | embedding_id:str | bool | collection | Embedding'in var olup olmadığını kontrol eder | process_file() | zpt4, zpt5, zpt9 |
| bibliography_exists() | bib_id:str | bool | bib_collection | Bibliyografik bilginin varlığını kontrol eder | process_file() | zpt4, zpt5, zpt9 |
| save_last_processed_chunk_index() | pdf_id:str, chunk_idx:int | - | - | Son işlenen chunk'ı kaydeder | process_file() | zpt4, zpt5, zpt9 |
| get_last_processed_chunk_index() | pdf_id:str | int | - | Son işlenen chunk'ı okur | process_file() | zpt4, zpt5, zpt9 |
Bu tabloya ek olarak, programlar arasındaki bazı önemli farkları da belirteyim:

1.  **Fonksiyon Geliştirmeleri:**

-   `extract_references()`  ->  `extract_references_enhanced()`: Daha kapsamlı referans çıkarma
-   `get_embedding()`  ->  `embed_text()`: Daha iyi hata yönetimi
-   Bazı programlarda chunk takibi eklendi

2.  **Özel Fonksiyonlar:**

-   zottxt30: Özel log yönetimi (`save_log`,  `save_error`)
-   zpt9: Gelişmiş referans çıkarma
-   zpt4,zpt5: ChromaDB entegrasyonu

Devam etmemi isterseniz:

1.  Her programın özel fonksiyonlarının detaylı analizini yapabilirim
2.  Program akış şemalarını oluşturabilirim
3.  Fonksiyonların gelişim sürecini detaylandırabilirim

Hangi konuda devam etmemi istersiniz?
  

Bu tabloya ek olarak, programlar arasındaki bazı önemli farkları da belirteyim:

1.  **Fonksiyon Geliştirmeleri:**

-   `extract_references()`  ->  `extract_references_enhanced()`: Daha kapsamlı referans çıkarma
-   `get_embedding()`  ->  `embed_text()`: Daha iyi hata yönetimi
-   Bazı programlarda chunk takibi eklendi

2.  **Özel Fonksiyonlar:**

-   zottxt30: Özel log yönetimi (`save_log`,  `save_error`)
-   zpt9: Gelişmiş referans çıkarma
-   zpt4,zpt5: ChromaDB entegrasyonu

Devam etmemi isterseniz:

1.  Her programın özel fonksiyonlarının detaylı analizini yapabilirim
2.  Program akış şemalarını oluşturabilirim
3.  Fonksiyonların gelişim sürecini detaylandırabilirim

Hangi konuda devam etmemi istersiniz?
 ### Özel Fonksiyon Grupları

1.  **Metin İşleme Fonksiyonları:**

| Grup | Fonksiyonlar | Programlar |
|------|--------------|------------|
| Temel Temizleme | clean_text(), wrapped_text_fix() | Tüm versiyonlar |
| Yapı Analizi | sutunlu_yapi_tespit(), detect_tables() | zottxt24, zottxt25, zpt9 |
| Referans Çıkarma | extract_references(), extract_references_enhanced() | Tüm versiyonlar |

2. **Durum Takip Fonksiyonları:**

| Grup | Fonksiyonlar | Programlar |
|------|--------------|------------|
| İşlem Takibi | save_last_processed_index(), get_last_processed_index() | zpt4, zpt5, zpt9 |
| Chunk Takibi | save_last_processed_chunk_index(), get_last_processed_chunk_index() | zpt4, zpt5, zpt9 |
| Log Yönetimi | save_log(), save_error(), log_guncelle() | zottxt24, zottxt25, zottxt30 |

3.  **Veri Dönüşüm Fonksiyonları:**5  lines▼

| Grup | Fonksiyonlar | Programlar |
|------|--------------|------------|
| PDF İşleme | extract_text_from_pdf(), pdf_isle() | zpt4, zpt5, zpt9, zottxt29 |
| Embedding | embed_text(), get_embedding() | zpt4, zpt5, zpt9 |
| Tablo İşleme | detect_tables(), metin_tablo_cikar() | Tüm versiyonlar |
Devam etmemi isterseniz:

1.  Her fonksiyon grubunun detaylı akış analizini yapabilirim
2.  Fonksiyonlar arası bağımlılık grafiğini çıkarabilirim
3.  Program versiyonları arasındaki fonksiyon geliştirmelerini detaylandırabilirim

Hangi konuda devam etmemi istersiniz?
Her programın akış şemasını hazırlıyorum. Markdown formatında mermaid diagram kullanarak göstereceğim:

### 1. zottxt30.py (TXT Odaklı İşleme)18  lines▼
```mermaid
sequenceDiagram
graph TD
    A[main] --> B[init_dirs]
    A --> C[Kullanıcı Seçimi B/C]
    C -->|B| D[Dizinleri Temizle]
    C -->|C| E[Devam Et]
    A --> F[Dosya Listesi Oluştur]
    F --> G[Paralel İşleme Başlat]
    G --> H[process_file]
    H --> I[Metin Oku]
    I --> J[Temizle ve İşle]
    J --> K[Tablo Tespit]
    J --> L[Kaynakça Tespit]
    K --> M[JSON/CSV/Excel Kaydet]
    L --> N[Kaynakça Kaydet]
    H --> O[Log Kayıt]
    O --> P{Başarılı?}
    P -->|Evet| Q[save_log]
    P -->|Hayır| R[save_error]
    
### 2. zpt9.py (PDF ve Embedding Odaklı)17  lines▼
```mermaid
sequenceDiagram
graph TD
    A[main] --> B[init_dirs]
    A --> C[JSON Dosyası Seç]
    C --> D[Kullanıcı Seçimi B/C/G]
    D -->|B| E[ChromaDB Sıfırla]
    D -->|C| F[Kaldığı Yerden Devam]
    D -->|G| G[Yeni Başla]
    A --> H[Paralel İşleme]
    H --> I[process_file]
    I --> J[PDF'den Text Çıkar]
    J --> K[Metin Temizle]
    K --> L[Paragraf Bölme]
    L --> M[Chunk Oluştur]
    M --> N[Embedding Al]
    N --> O[ChromaDB Kaydet]
    I --> P[Bibliyografik Bilgi]
    P --> Q[ChromaDB Bib Koleksiyon]

### 3. zpt4.py ve zpt5.py (Hibrit İşleme)17  lines▼
```mermaid
sequenceDiagram
graph TD
    A[main] --> B[init_dirs]
    A --> C[JSON/PDF Seçimi]
    C --> D[İşlem Modu Seç]
    D --> E[Dosya İşleme Döngüsü]
    E --> F[process_file]
    F --> G[Dosya Türü Kontrol]
    G -->|PDF| H[extract_text_from_pdf]
    G -->|TXT| I[Doğrudan Oku]
    H --> J[Metin İşleme]
    I --> J
    J --> K[Tablo Tespit]
    J --> L[Referans Çıkar]
    J --> M[Embedding Oluştur]
    M --> N[ChromaDB Kaydet]
    F --> O[İlerleme Takibi]
    O --> P[save_last_processed_index]
### 4. zottxt24.py ve zottxt25.py (TXT Yapısal Analiz)14  lines▼
```mermaid
    A[main] --> B[init_dirs]
    A --> C[dosya_listesini_guncelle]
    C --> D[kalan_dosyalari_oku]
    D --> E[Dosya İşleme Döngüsü]
    E --> F[process_file]
    F --> G[wrapped_text_fix]
    F --> H[sutunlu_yapi_tespit]
    F --> I[metin_tablo_cikar]
    F --> J[kaynakca_cikar]
    I --> K[Tablo Kaydet]
    J --> L[Kaynakça Kaydet]
    F --> M[Log Güncelle]
    M --> N[stack_guncelle]
Her program için önemli noktalar:

1.  zottxt30: Paralel işleme ve format dönüşümlerine odaklanıyor
2.  zpt9: Embedding ve ChromaDB entegrasyonuna ağırlık veriyor
3.  zpt4/5: Hibrit yapı ile hem PDF hem TXT işleyebiliyor
4.  zottxt24/25: Yapısal analiz ve metin temizlemeye yoğunlaşıyor

Devam etmemi isterseniz:

1.  Her akış şemasının daha detaylı alt süreçlerini gösterebilirim
2.  Programlar arası geçiş ve geliştirme sürecini görselleştirebilirim
3.  Kritik karar noktalarını ve hata yönetimini detaylandırabilirim

Başlıyorum. Analizin kapsamlı olması için sırayla ilerleyeceğim:

### 1. Detaylı Alt Süreçler29  lines▼
```mermaid
```graph TD
   ``` subgraph "Metin İşleme Alt Süreci"
        A[PDF/TXT Okuma] --> B[Ham Metin]
        B --> C[Temizleme]
        C --> D[Yapısal Analiz]
        D --> E1[Tablo Tespiti]
        D --> E2[Referans Tespiti]
        D --> E3[Sütun Analizi]
        E1 --> F1[Tablo Formatları]
        F1 --> G1[JSON]
        F1 --> G2[CSV]
        F1 --> G3[Excel]
    ```end

    ```subgraph "Embedding Alt Süreci"
        H[Temiz Metin] --> I[Chunk Bölme]
        I --> J[Token Kontrolü]
        J --> K[OpenAI API]
        K --> L[Embedding Vektörü]
        L --> M[ChromaDB Kayıt]
    ```end

    ```subgraph "Hata Yönetimi Alt Süreci"
        N[İşlem Başlangıcı] --> O{Hata Kontrolü}
        O -->|Başarılı| P[Log Kayıt]
        O -->|Başarısız| Q[Hata Log]
        Q --> R[Retry Mekanizması]
        R --> S[Max Deneme Kontrolü]
    ```end
### 2. Programlar Arası Geçiş ve Gelişim12  lines▼
```mermaid
graph LR
    A[zottxt24/25] -->|Temel Metin İşleme| B[zottxt30]
    B -->|Paralel İşleme Eklendi| C[zpt4]
    C -->|Embedding Eklendi| D[zpt5]
    D -->|ChromaDB Geliştirmesi| E[zpt9]

    subgraph "Geliştirme Aşamaları"
        F[Temel İşlevler] --> G[Paralel İşleme]
        G --> H[Embedding]
        H --> I[Veritabanı]
        I --> J[Optimizasyon]
    end
### 3. Kritik Karar Noktaları ve Hata Yönetimi22  lines▼
```mermaid
graph TD
    A[Program Başlangıcı] --> B{Dizin Kontrolü}
    B -->|Başarısız| C[Dizin Oluştur]
    B -->|Başarılı| D{Dosya Formatı}

    D -->|PDF| E[PDF İşleme Akışı]
    D -->|TXT| F[TXT İşleme Akışı]

    E --> G{Metin Çıkarma}
    G -->|Başarısız| H[Hata Logu]
    G -->|Başarılı| I[İşleme Devam]

    subgraph "Hata Yönetimi Detayları"
        J[Hata Tespiti] --> K{Hata Tipi}
        K -->|IO Error| L[Dosya Erişim]
        K -->|API Error| M[OpenAI]
        K -->|DB Error| N[ChromaDB]
        
        L --> O[Retry Logic]
        M --> P[Rate Limit]
        N --> Q[Connection]
    end
### Kritik Noktalar ve Öneriler:

1.  **Metin İşleme İyileştirmeleri:**10  lines▼

    def improved_text_processing(text):
        # Paragraf yapısını koru
        paragraphs = text.split('\n\n')
        # Her paragrafı ayrı işle
        processed = []
        for p in paragraphs:
            p = clean_text(p)
            if is_valid_paragraph(p):
                processed.append(p)
        return '\n\n'.join(processed)

2.  **Hata Yönetimi Geliştirmesi:**24  lines▼

    class ProcessingError(Exception):
        def __init__(self, error_type, message, retry_count=0):
            self.error_type = error_type
            self.message = message
            self.retry_count = retry_count
            super().__init__(self.message)
    
    def error_handler(func):
        def wrapper(*args, **kwargs):
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise ProcessingError(
                            type(e).__name__, 
                            str(e), 
                            retry_count
                        )
                    time.sleep(retry_count * 2)
        return wrapper

3.  **Veri Tutarlılığı Kontrolleri:**13  lines▼

    def validate_process_state(file_id, chunk_index):
        # Dosya durumu kontrolü
        if not os.path.exists(f"state_{file_id}.json"):
            return False
        
        # Chunk kontrolü
        with open(f"state_{file_id}.json", 'r') as f:
            state = json.load(f)
        
        return (
            state['last_chunk'] == chunk_index - 1 and
            state['embeddings_count'] == chunk_index - 1
        )

Devam etmek isterseniz:

1.  Her bir alt sürecin performance optimizasyonlarını gösterebilirim
2.  Veri tutarlılığı ve güvenliği için ek kontroller önerebilirim
3.  Paralel işleme için daha detaylı bir mimari sunabilirim

Hangi konuda devam etmemi istersiniz?

Tüm programların karşılaştırmalı analizini yapıyorum:

### Dosya Karşılaştırma Analizi12  lines▼\
|Özellik/Yetenek |zottxt24/25|zottxt29/30|zpt4/5  |zpt9   |
|----------------|-----------|-----------|--------|-------|
|Dosya Desteği   |TXT        |TXT,PDF    |TXT,PDF |TXT,PDF|
|Paralel İşleme  |❌		 |✅		|✅       |✅      |
|Embedding       |❌          |❌          |✅       |✅+     |
|ChromaDB        |❌          |❌          |✅       |✅+     |
|Tablo Tespiti   |✅          |✅+         |✅+      |✅+     |
|Kaynakça Çıkarma|✅          |✅          |✅+      |✅++    |
|Hata Yönetimi   |Basit      |Gelişmiş   |Gelişmiş|İleri  |
|İlerleme Takibi |✅          |✅          |✅+      |✅++    |
|Bellek Yönetimi |❌          |✅          |✅       |✅+     |
|Log Sistemi     |Basit      |Gelişmiş   |Gelişmiş|İleri  |

### Detaylı Karşılaştırma

1.  **Metin İşleme Yetenekleri:**6  lines▼
|Program    |Temizleme        |Yapısal Analiz|Format Dönüşümü|
|-----------|-----------------|--------------|---------------|
|zottxt24/25|Basit regex      |Sütun tespiti |JSON           |
|zottxt29/30|Gelişmiş regex   |Çoklu format  |JSON,CSV,Excel |
|zpt4/5     |İleri regex      |PDF+TXT       |Çoklu format   |
|zpt9       |İleri regex + NLP|Akıllı tespit |Tam entegrasyon|

2.  **Performans Özellikleri:**6  lines▼
|Program    |Paralellik          |Chunk Yönetimi|Bellek Optimizasyonu|
|-----------|--------------------|--------------|--------------------|
|zottxt24/25|Tek thread          |❌             |❌                   |
|zottxt29/30|MultiProcess        |❌             |✅                   |
|zpt4/5     |MultiProcess        |✅             |✅                   |
|zpt9       |MultiProcess + Queue|✅+            |✅+                  |

3.  **Veri Tabanı ve API Entegrasyonu:**  
|Program    |ChromaDB|OpenAI|Zotero API|
|-----------|--------|------|----------|
|zottxt24/25|❌       |❌     |❌         |
|zottxt29/30|❌       |❌     |❌         |
|zpt4/5     |Temel   |✅     |✅         |
|zpt9       |Gelişmiş|✅+    |✅+        |
4.  **Dosya İşleme Farklılıkları:**6  lines▼
|Özellik          |zottxt24/25|zottxt29/30|zpt4/5      |zpt9         |
|-----------------|-----------|-----------|------------|-------------|
|PDF Metin Çıkarma|❌         |PyMuPDF    |PyMuPDF     |PyMuPDF+     |
|Tablo Tespiti    |Regex      |Regex+     |Regex+      |AI Destekli  |
|Referans Analizi |Basit      |Orta       |Gelişmiş    |İleri        |
|Format Desteği   |TXT        |TXT,PDF    |TXT,PDF,JSON|Tüm Formatlar|

5.  **Hata Yönetimi ve Loglama:**6  lines▼
|Program    |Log Seviyesi|Hata Kurtarma|İlerleme Takibi|
|-----------|------------|-------------|---------------|
|zottxt24/25|Basit       |❌            |Dosya bazlı    |
|zottxt29/30|JSON        |✅            |Chunk bazlı    |
|zpt4/5     |JSON+Renkli |✅+           |Tam izleme     |
|zpt9       |Çoklu Kanal |✅++          |Gerçek zamanlı |

##  Program Özellikleri Gelişim Süreci: ##

1.  **zottxt24/25 -> zottxt29/30**

-   Paralel işleme eklendi
-   Gelişmiş log sistemi
-   Bellek yönetimi eklendi

2.  **zottxt29/30 -> zpt4/5**

-   Embedding desteği
-   ChromaDB entegrasyonu
-   PDF işleme yetenekleri

3.  **zpt4/5 -> zpt9**

-   Gelişmiş embedding yönetimi
-   İleri düzey hata kontrolü
-   Akıllı metin analizi

Devam etmemi isterseniz:

1.  Her programın benzersiz güçlü yönlerini detaylandırabilirim
2.  Performans karşılaştırma metrikleri sunabilirim
3.  Geliştirme önerileri ve optimizasyon tavsiyeleri verebilirim

Hangi konuda devam etmemi istersiniz?

Her programın özel fonksiyonlarını ve bu fonksiyonların detaylı analizini yapıyorum:

##  zottxt24/25 Özel Fonksiyonları24  lines▼

1. wrapped_text_fix()
   - Amaç: Satır sonlarında kesilmiş kelimeleri birleştirir
   - Girdiler: text:str
   - Çıktılar: fixed_text:str
   - Özellikler:
     * Paragraf yapısını korur
     * Tire ile bölünmüş kelimeleri birleştirir
     * Boş satırları yönetir

2. sutunlu_yapi_tespit()
   - Amaç: Metinde sütunlu yapı olup olmadığını tespit eder
   - Girdiler: text:str
   - Çıktılar: bool
   - Özellikler:
     * Tab karakterlerini kontrol eder
     * Boşluk örüntülerini analiz eder
     * Sütun oranını hesaplar

3. stack_yukle() ve stack_guncelle()
   - Amaç: İşlem durumunu takip eder
   - Özellikler:
     * Dosya bazlı ilerleme takibi
     * Hata durumunda kurtarma
     * Durum belleği

## zottxt29/30 Özel Fonksiyonları18  lines▼
1. process_file()
   - Gelişmiş özellikler:
     * Paralel işleme desteği
     * Bellek yönetimi
     * Çoklu format çıktı

2. save_log() ve save_error()
   - Amaç: Gelişmiş loglama sistemi
   - Özellikler:
     * JSON formatında log
     * Hata izleme
     * Bellek kullanım takibi

3. detect_tables()
   - Geliştirilmiş özellikleri:
     * Çoklu tablo formatı desteği
     * Gelişmiş regex pattern'leri
     * DataFrame dönüşümü
## zpt4/5 Özel Fonksiyonları20  lines▼
1. embed_text()
   - Amaç: OpenAI API ile embedding oluşturma
   - Özellikler:
     * Rate limiting kontrolü
     * Hata yönetimi
     * Retry mekanizması

2. fetch_zotero_metadata()
   - Amaç: Zotero verisi çekme
   - Özellikler:
     * API entegrasyonu
     * Veri doğrulama
     * Hata yönetimi

3. save_last_processed_chunk_index()
   - Amaç: Chunk seviyesinde ilerleme takibi
   - Özellikler:
     * Dosya bazlı takip
     * Kurtarma noktaları
     * Durum yönetimi
## zpt9 Özel Fonksiyonları18  lines▼
1. extract_references_enhanced()
   - Gelişmiş özellikler:
     * Çoklu referans formatı desteği
     * Akıllı bölüm tespiti
     * Gelişmiş temizleme

2. process_file()
   - En kapsamlı versiyonu:
     * Tam entegrasyon
     * Gelişmiş hata yönetimi
     * İleri düzey loglama
     * Bellek optimizasyonu

3. ChromaDB Entegrasyonu
   - Özel fonksiyonlar:
     * embedding_exists()
     * bibliography_exists()
     * Toplu veri ekleme
## Her Programın Benzersiz Yetenekleri

1.  **zottxt24/25**11  lines▼

    # Satır birleştirme yeteneği
    def wrapped_text_fix(text):
        lines = text.split("\n")
        fixed_lines = []
        for i, line in enumerate(lines):
            if line.endswith("-") and i + 1 < len(lines):
                fixed_lines.append(line[:-1] + lines[i + 1])
                lines[i + 1] = ""
            else:
                fixed_lines.append(line)
        return "\n".join(fixed_lines)

2.  **zottxt29/30**8  lines▼

    # Gelişmiş log sistemi
    def save_log(entry, log_type="info"):
        log_file = LOG_DOSYASI if log_type == "info" else HATA_LOG_DOSYASI
        with open(log_file, 'a', encoding='utf-8') as f:
            entry['timestamp'] = datetime.now().isoformat()
            entry['memory_usage'] = memory_usage()
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

3.  **zpt4/5**8  lines▼

    # Chunk yönetimi
    def process_chunks(text, file_id):
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_{i}"
            if not embedding_exists(chunk_id):
                embedding = embed_text(chunk)
                save_chunk_embedding(chunk_id, embedding)

4.  **zpt9**12  lines▼
# Gelişmiş referans çıkarma

    def extract_references_enhanced(text):
        patterns = [
            r'(?i)(?:KAYNAKÇA|REFERENCES).*?\n(.*?)(?=\n\s*\n|\Z)',
            r'\[\d+\]\s.*?(?=\n|$)',
            r'(?:[A-Za-z]+,\s+\d{4})'
        ]
        references = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            references.extend([m.group(0) for m in matches])
        return clean_references(references)

Her program kendi özel kullanım alanına göre optimize edilmiş fonksiyonlara sahiptir. İsterseniz herhangi bir programın fonksiyonları hakkında daha detaylı bilgi verebilirim.

Programların benzersiz güçlü yönlerini detaylı olarak analiz ediyorum:

### zpt4 (Ana Özellikler)14  lines▼
1. Ortam Değişkeni Kontrolleri
- Kapsamlı ortam değişkeni kontrolü
- Başlangıçta tüm API anahtarlarının kontrolü
- Detaylı hata loglama sistemi

2. Gelişmiş Loglama
- Renkli konsol çıktısı
- Çoklu log hedefi (dosya + konsol)
- Detaylı hata izleme

3. Process Pool İşleme
- Çoklu işlemci desteği
- Paralel PDF işleme
- İş yükü dengeleme
### zpt6 (Özel Yetenekler)14  lines▼
1. PDF İşleme Geliştirmeleri
- Sütun yapısı tespiti
- Gelişmiş metin çıkarma
- Sayfa bazlı işleme

2. Hata Yönetimi
- Her işlem adımında hata kontrolü
- Kurtarma mekanizmaları
- Detaylı hata raporlama

3. Bellek Optimizasyonu
- Chunk bazlı işleme
- Verimli bellek kullanımı
- Kaynak temizleme
### zpt7 (Yenilikçi Özellikler)14  lines▼
1. Metin İşleme
- Paragraf yapısını koruyan temizleme
- Akıllı chunk bölümleme
- İçerik analizi

2. Dosya Yönetimi
- Otomatik dizin oluşturma
- Format dönüşümleri
- Yapılandırılmış dosya organizasyonu

3. API Entegrasyonları
- ChromaDB optimizasyonu
- OpenAI API hata yönetimi
- Zotero metadata entegrasyonu
### zpt8 (Gelişmiş Özellikler)14  lines▼
1. Referans Yönetimi
- Gelişmiş referans çıkarma
- Çoklu format desteği
- Metadata ilişkilendirme

2. Veri Tabanı İşlemleri
- Verimli ChromaDB kullanımı
- Toplu veri ekleme
- İndeksleme optimizasyonları

3. Süreç Takibi
- İlerleme çubuğu
- Durum raporlama
- Performans metrikleri
### Özel Fonksiyon Örnekleri

1.  **zpt4'ün Benzersiz Fonksiyonu:**10  lines▼

    def process_with_error_handling(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"İşlem hatası: {str(e)}")
                traceback.print_exc()
                return None
        return wrapper

2.  **zpt6'nın Özel PDF İşleme Yeteneği:**13  lines▼

    def extract_text_with_layout(pdf_path):
        """PDF'den metin çıkarma (sayfa düzeni korunarak)"""
        try:
            with fitz.open(pdf_path) as doc:
                text_blocks = []
                for page in doc:
                    blocks = page.get_text("blocks")
                    sorted_blocks = sorted(blocks, key=lambda b: (-b[1], b[0]))
                    text_blocks.extend([b[4] for b in sorted_blocks])
                return "\n".join(text_blocks)
        except Exception as e:
            logger.error(f"PDF işleme hatası: {e}")
            return None

3.  **zpt7'nin Chunk Yönetimi:**21  lines▼

    def adaptive_chunk_split(text, target_size=256):
        """Akıllı metin bölümleme"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            if current_size + para_size <= target_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        return chunks

4.  **zpt8'in Veri Tabanı Optimizasyonu:**16  lines▼

    def batch_chromadb_insert(embeddings, metadata, batch_size=100):
        """Toplu ChromaDB veri ekleme"""
        try:
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadata = metadata[i:i+batch_size]
                batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_embeddings))]
                
                collection.add(
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                logger.info(f"Batch {i//batch_size + 1} eklendi")
        except Exception as e:
            logger.error(f"Batch ekleme hatası: {e}")

Bu benzersiz özellikler ve fonksiyonlar, her programın belirli kullanım senaryolarında öne çıkmasını sağlar. İsterseniz herhangi bir programın özel yeteneklerini daha detaylı açıklayabilirim.

Her programın zayıf yönlerini detaylı olarak analiz ediyorum:

### zpt4 Zayıf Yönleri14  lines▼
1. Bellek Yönetimi
- Büyük PDF'lerde bellek sızıntısı riski
- Chunk işlemlerinde bellek optimizasyonu eksik
- Geçici dosyaların temizlenmesi yetersiz

2. Hata Yönetimi
- Rate limiting kontrolü yetersiz
- API hataları için retry mekanizması yok
- Kısmi işlem durumunda veri kaybı riski

3. Performans
- Paralel işlemede iş yükü dengesiz
- Büyük dosyalarda yavaşlama
- Disk I/O optimizasyonu eksik
### zpt6/7 Zayıf Yönleri14  lines▼
1. API Entegrasyonları
- OpenAI API kesintilerinde yetersiz hata yönetimi
- Zotero API timeout kontrolü eksik
- API istek limitleri yönetilmiyor

2. Veri Doğrulama
- Eksik metadata kontrolü
- Format doğrulama yetersiz
- Dosya bütünlüğü kontrolü zayıf

3. ChromaDB İşlemleri
- Veritabanı bağlantı hataları yönetilmiyor
- Toplu işlem optimizasyonu eksik
- İndeks yönetimi yetersiz
### zpt8 Zayıf Yönleri14  lines▼
1. Dosya İşleme
- PDF yapı analizi sınırlı
- Karmaşık tablo yapıları işlenemiyor
- Görsel içerik yönetimi eksik

2. Süreç Yönetimi
- Kesinti durumunda kurtarma mekanizması zayıf
- İlerleme takibi yetersiz
- Durum raporlama sınırlı

3. Kaynak Yönetimi
- Bellek kullanımı optimize edilmemiş
- Disk kullanımı verimsiz
- Çoklu işlem yönetimi iyileştirilmeli
### Detaylı Zayıf Yön Analizleri

1.  **API ve Bağlantı Sorunları:**21  lines▼

    # Mevcut Yetersiz Hata Yönetimi
    try:
        response = client.embeddings.create(...)
    except Exception as e:
        logger.error(f"Embedding hatası: {e}")
        return None
    
    # Olması Gereken
    def create_embedding_with_retry(text, max_retries=3, delay=1):
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(...)
                return response.data[0].embedding
            except RateLimitError:
                time.sleep(delay * (attempt + 1))
            except APIError as e:
                if e.status_code >= 500:
                    time.sleep(delay)
                else:
                    raise
        return None

2.  **Bellek Yönetimi Sorunları:**16  lines▼
# Mevcut Verimsiz Kod

    def process_large_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()  # Bellek birikimine yol açar
        return text
    
    # Olması Gereken
    def process_large_pdf(pdf_path):
        text_chunks = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                chunk = page.get_text()
                process_chunk(chunk)  # Her chunk'ı hemen işle
                del chunk  # Belleği temizle

3.  **Dosya İşleme Sorunları:**18  lines▼

    # Mevcut Yetersiz Kontrol
    
        def save_text_file(path, content):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Olması Gereken
        def save_text_file(path, content):
            temp_path = path + '.tmp'
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                if os.path.exists(path):
                    os.backup(path, path + '.bak')
                os.rename(temp_path, path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

4.  **Veri Doğrulama Sorunları:**22  lines▼

    # Mevcut Yetersiz Doğrulama def process_metadata(data):
        return {
            'title': data.get('title', ''),
            'author': data.get('author', '')
        }
    
    # Olması Gereken def process_metadata(data):
        required_fields = ['title', 'author', 'date']
        for field in required_fields:
            if not data.get(field):
                raise ValueError(f"Zorunlu alan eksik: {field}")
        
        if len(data['title']) > 256:
            raise ValueError("Başlık çok uzun")
            
        return {
            'title': clean_and_validate_title(data['title']),
            'author': clean_and_validate_author(data['author']),
            'date': parse_and_validate_date(data['date'])
        }

Bu zayıf yönlerin giderilmesi için öneriler:

1.  **API Yönetimi İyileştirmeleri:**

-   Rate limiting için token bucket implementasyonu
-   Retry mekanizması için exponential backoff
-   API durum kontrolü için health check sistemi

2.  **Bellek Optimizasyonu:**
-   Streaming işleme desteği
-   Garbage collection optimizasyonu
-   Bellek kullanım limitleri

3.  **Hata Yönetimi:**

-   Detaylı hata sınıfları
-   Transaction-based işlem yönetimi
-   Otomatik kurtarma mekanizmaları

4.  **Performans İyileştirmeleri:**

-   Asenkron işleme desteği
-   Önbellekleme mekanizmaları
-   İş parçacığı havuzu optimizasyonu