# Zotero Entegre PDF İşleyici

Bu proje, Zotero tarafından kullanılan PDF ve TXT dosyalarından ham metin çıkarımı, 
metin temizleme, bilimsel bölüm haritalaması, tablo ve kaynakça çıkarımı, embedding
oluşturma, atıf mapping (citation mapping) ve verilerin kümeleme analizi gibi işlemleri
gerçekleştiren kapsamlı bir sistemdir. Proje, çok modüllü bir yapıda organize 
edilmiştir ve her modül, belirli işlevleri yerine getirir.

---

## İçerik

- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Dosya Yapısı](#dosya-yapısı)
- [Modüller](#modüller)
  - [config_module.py](#config_modulepy)
  - [zotero_module.py](#zotero_modulepy)
  - [pdf_processing.py](#pdf_processingpy)
  - [embedding_module.py](#embedding_modulepy)
  - [alternative_embedding_module.py](#alternative_embedding_modulepy)
  - [robust_embedding_module.py](#robust_embedding_modulepy)
  - [helper_module.py](#helper_modulepy)
  - [processing_manager.py](#processing_managerpy)
  - [file_save_module.py](#file_save_modulepy)
  - [citation_mapping_module.py](#citation_mapping_modulepy)
  - [gui_module.py](#gui_modulepy)
  - [main.py](#mainpy)
- [Çalıştırma](#çalıştırma)
- [Geliştirme Önerileri](#geliştirme-önerileri)
- [Ek Notlar](#ek-notlar)

---

## Gereksinimler

Proje, aşağıdaki Python paketlerine bağımlıdır. Bu paketlerin doğru sürümleri `requirements.txt` dosyasında belirtilmiştir:

- openai
- chromadb
- python-dotenv
- pdfplumber
- pdfminer.six
- customtkinter
- tqdm
- pandas
- scikit-learn
- psutil
- rapidfuzz
- sentence-transformers
- transformers
- xlsxwriter
- (Opsiyonel) hdbscan

Bu paketleri kurmak için, terminalde aşağıdaki komutu çalıştırabilirsiniz:

```bash
pip install -r requirements.txt

Kurulum
.env Dosyasını Oluşturun:
Proje kök dizinine bir .env dosyası ekleyin. Yukarıdaki .env içeriğini kullanarak, dizin yolları, API anahtarları ve diğer ayarları belirleyin.

Gerekli Paketleri Kurun:
requirements.txt dosyasını kullanarak tüm bağımlılıkları yükleyin.

Projeyi Çalıştırın:
Terminal veya VS Code içerisinden main.py dosyasını çalıştırarak uygulamayı başlatın.


Dosya Yapısı
Projenin dosya yapısı aşağıdaki gibidir:

project_root/
│
├── config_module.py                # Ortam değişkenleri, dizin yapılandırması ve loglama ayarlarını içerir.
├── zotero_module.py                # Zotero API entegrasyonu ve bibliyometrik veri çekme işlevlerini sağlar.
├── pdf_processing.py               # PDF dosyalarından metin çıkarma, reflow, bilimsel bölüm haritalaması, sütun tespiti ve tablo çıkarımı.
├── embedding_module.py             # OpenAI API kullanarak metin embedding oluşturma ve metin parçalama işlemleri.
├── alternative_embedding_module.py # SentenceTransformer tabanlı alternatif embedding modelleri ile çalışır.
├── robust_embedding_module.py      # Hata toleransı, retry ve alternatif model geçiş mekanizması ile robust embedding oluşturma.
├── helper_module.py                # Genel yardımcı fonksiyonlar: metin temizleme, fuzzy matching, bellek ölçümü, stack yönetimi.
├── processing_manager.py           # Dosya işleme akışını yöneten ana sınıf; PDF/TXT dosyalarının tüm iş adımlarını koordine eder.
├── file_save_module.py             # İşlenmiş verilerin (temiz metin, kaynakça, tablolar, embedding) dosya sistemine kaydedilmesini sağlar.
├── citation_mapping_module.py      # Atıf mapping (citation mapping) işlemlerini gerçekleştiren modül; metni cümlelere bölme, atıf eşleştirme ve JSON olarak kaydetme.
├── gui_module.py                   # Kullanıcı arayüzünü (GUI) sağlayan modül; dosya seçimi, işlem başlatma, atıf zinciri görüntüleme, ek özellikler.
├── main.py                         # Tüm modülleri entegre eden ana giriş noktası.
├── .env                            # Ortam ayarlarını içeren dosya.
├── requirements.txt                # Gerekli Python paketlerinin listesi.
└── README.md                       # Proje hakkında detaylı bilgi.

Modüller

config_module.py

Amaç: Proje genelinde kullanılacak ortam değişkenlerini yükler, gerekli dizinleri oluşturur ve merkezi loglama yapılandırmasını ayarlar.

Özellikler:

Dizin yolları (STORAGE_DIR, SUCCESS_DIR, LOG_DIR, vb.)
API anahtarları (OPENAI_API_KEY, ZOTERO_API_KEY, ZOTERO_USER_ID)
PDF metin çıkarma yöntemi, embedding modeli, paralel işlem ayarları, vb.
Kullanım: Diğer modüller from config_module import config şeklinde erişir.

zotero_module.py

Amaç: Zotero API'si üzerinden bibliyometrik verileri çekmek ve referans analizlerini yapmak.

Özellikler:

dokuman_id_al: Dosya adından Zotero ID'si çıkarma.
fetch_zotero_metadata: Zotero API kullanarak bibliyometrik verileri çekme.
referanslari_analiz_et: Kaynakça referanslarını analiz etme.
Kullanım: PDF işleme ve citation mapping süreçlerinde kullanılır.

pdf_processing.py

Amaç: PDF dosyalarından ham metin çıkarma, metni reflow ederek tek akışa dönüştürme, bilimsel bölümleri haritalama, sütun tespiti ve gelişmiş tablo çıkarımı.

Özellikler:

extract_text_from_pdf: Pdfplumber öncelikli, başarısız olursa pdfminer kullanarak metin çıkarır.
detect_columns: Metindeki sütun yapısını tespit eder.
map_scientific_sections_extended: Bilimsel bölümleri (Abstract, Introduction, Methods, Results, Discussion, Conclusion vb.) tespit eder.
map_pdf_before_extraction: PDF'den metin çıkarılmadan önce yapısal analiz yapar.
reflow_columns: HTML/Markdown etiketlerini, sayfa bilgilerini ve ekstra boşlukları temizleyerek metni tek akışa dönüştürür.

Kullanım: Dosya işleme sürecinin ilk adımları olarak kullanılır.

embedding_module.py

Amaç: OpenAI API kullanarak metin embedding oluşturma işlemlerini gerçekleştirir. 

Ayrıca metinleri belirli boyutlarda parçalara ayırma ve büyük dosyaların işlenmesi için ek fonksiyonlar içerir.

Özellikler:

split_text: Metni kelime veya paragraf bazında parçalara ayırır.
embed_text: OpenAI API ile metin embedding oluşturur.
process_large_text: Büyük metinleri parçalara ayırıp robust embedding oluşturma işlemini gerçekleştirir.
Kullanım: İşleme sürecinde embedding işlemleri için kullanılır.

alternative_embedding_module.py

Amaç: OpenAI haricinde alternatif embedding modelleri (SentenceTransformer tabanlı) kullanarak metin embedding oluşturmayı sağlar.

Özellikler:

MODEL_LIST: Alternatif modellerin anahtar ve isimlerini içeren sözlük.
get_sentence_transformer: Belirtilen model anahtarına göre modeli yükler.
embed_text_with_model: Yüklenen modeli kullanarak embedding oluşturur.
get_available_models: Kullanılabilir alternatif modellerin listesini döndürür.
Kullanım: Robust embedding modülünde ve kullanıcı seçimine göre alternatif model geçişlerinde kullanılır.

robust_embedding_module.py

Amaç: Embedding oluşturma sırasında hata toleransı, yeniden deneme (retry) ve alternatif model geçiş mekanizması sağlar.

Özellikler:

robust_embed_text: Belirtilen metin için, model priority listesi üzerinden sırasıyla deneme yapar; her model için exponential backoff ve retry mekanizması uygular.
Kullanım: Embedding işlemlerinde, hata durumlarında güvenli geçiş sağlamak için kullanılır.

helper_module.py

Amaç: Genel yardımcı fonksiyonları sağlar; metin temizleme, başlık kısaltma, bellek ölçümü, fuzzy matching ve stack yönetimi.

Özellikler:

memory_usage: Bellek kullanımını MB cinsinden ölçer.
shorten_title: Uzun başlıkları belirtilen karakter sayısına indirger.
clean_advanced_text: HTML/Markdown etiketlerini, sayfa baş/sonu ifadelerini ve ekstra boşlukları temizler.
fuzzy_match: RapidFuzz kullanarak iki metin arasındaki benzerlik oranını hesaplar.
stack_yukle & stack_guncelle: İşlem sırasını takip etmek için stack yönetimi sağlar.
Kullanım: Diğer modüller tarafından destekleyici işlevler için kullanılır.

processing_manager.py

Amaç: PDF/TXT dosyalarını işleme sürecinin tüm adımlarını (metin çıkarımı, reflow, 
bölüm haritalaması, sütun tespiti, kaynakça çıkarımı, Zotero entegrasyonu, embedding oluşturma) yöneten ana sınıfı içerir.

Özellikler:

pdf_txt_isle: Dosya türüne göre uygun metin çıkarımı yapar, temiz metin, bölüm haritalaması,
 sütun tespiti, kaynakça çıkarımı, embedding oluşturma ve Zotero entegrasyonunu gerçekleştirir.
stack_guncelle: Dosya işleme sırasını günceller (ekle/sil).
Kullanım: İşlemlerin merkezi kontrol noktasıdır.

file_save_module.py

Amaç: İşlenmiş verilerin dosya sistemine kaydedilmesini sağlar;
temiz metin, kaynakça, tablolar, embedding ve chunk'lar belirlenen formatlarda saklanır.

Özellikler:

save_text_file: Metin verilerini TXT formatında kaydeder.
save_json_file: JSON formatında veri kaydeder.
save_clean_text_files: Temiz metni TXT ve JSON formatında kaydeder.
save_references_files: Kaynakça verilerini TXT, JSON, VOSviewer, Pajek ve CSV formatlarında kaydeder.
save_table_files: Tabloları JSON, CSV ve Excel formatlarında kaydeder.
save_embedding_file: Embedding verilerini dosya sistemine kaydeder.
save_chunked_text_files: Büyük metinleri parçalara bölüp kaydeder.
Kullanım: İşlenen verilerin arşivlenmesi ve sonraki işlemler için saklanması.

citation_mapping_module.py

Amaç: Metin içerisindeki atıf ifadelerinin (citation mapping) 
tespit edilip, kaynakça verileriyle eşleştirilmesini sağlar.

Özellikler:

split_into_sentences: Metni cümlelere bölüp her cümleye sıra numarası ekler.
extract_citations_from_sentence: Cümle içindeki atıf ifadelerini regex, fuzzy matching ve NER (opsiyonel) kullanarak tespit eder.
match_citation_with_references: Tespit edilen atıf ifadesini kaynakça referanslarıyla eşleştirir.
get_section_for_sentence: Cümlenin hangi bilimsel bölümde olduğunu belirler.
map_citations: Tüm cümleler üzerinden atıf mapping verilerini oluşturur.
save_citation_mapping ve load_citation_mapping: Citation mapping verilerini JSON formatında kaydeder ve yükler.
Kullanım: Atıf zinciri oluşturma ve ilgili analizlerde kullanılır.

gui_module.py

Amaç: Kullanıcı arayüzünü (GUI) yönetir; kullanıcı dosya seçimi, işlem başlatma, 
atıf zinciri görüntüleme ve ek özellikler (embedding arama, kümeleme analizi, 
fine-tuning, veri sorgulama) aracılığıyla etkileşimde bulunur.

Özellikler:

PDF seçme, işleme, atıf zinciri görüntüleme için butonlar ve metin kutusu.
İlave özellikler menüsü: Embedding arama, kümeleme analizi, fine-tuning ve gelişmiş veri sorgulama.
Kullanıcı girdilerini almak için input diyalog kutuları.
Durum çubuğu ve sonuç ekranı ile işlem ilerleyişi hakkında bilgi verir.
Kullanım: Projeyi görsel olarak kontrol etmek ve işlemleri başlatmak için kullanılır.

main.py

Amaç: Tüm modüllerin entegre edildiği ana giriş noktasıdır.

Özellikler:

Ortam değişkenlerini yükler, IslemYoneticisi oluşturur, STORAGE_DIR’deki dosya sayısını sayaçlara aktarır.
Ana GUI arayüzünü başlatır.
İşlem sonuçlarını terminalde raporlar.
Kullanım: Uygulamayı başlatmak için python main.py komutu kullanılır.

Çalıştırma
.env Dosyasını Oluşturun:
Proje kök dizininde yukarıdaki .env içeriğini içeren bir dosya oluşturun ve ilgili yollar, API anahtarları gibi bilgileri girin.

Bağımlılıkları Yükleyin:
Terminalde pip install -r requirements.txt komutunu çalıştırarak tüm gereksinimleri yükleyin.

Ana Programı Çalıştırın:
Terminalden python main.py komutunu çalıştırarak uygulamayı başlatın.

Geliştirme Önerileri
Error Handling ve Retry Mekanizmaları:
Embedding oluşturma, PDF metin çıkarma gibi kritik işlemlerde hata durumunda otomatik 
yeniden deneme (exponential backoff) ve alternatif model geçiş mekanizması uygulanmıştır.

Paralel İşlem:
İşlemler, ProcessPoolExecutor gibi yapılar kullanılarak çok çekirdekli sistemlerde paralel şekilde yürütülmektedir.

Yığın (Cache) Sistemi:
Embedding işlemlerinde aynı metin parçalarının tekrar işlenmesini önlemek için stack (işlem listesi) yönetimi uygulanmıştır.

Modüler Yapı:
Her modül, projenin belirli bir işlevini yerine getirir. Kod tekrarı önlenmiş, fonksiyonlar merkezi yapılandırma ve loglama üzerinden yönetilmektedir.

Alternatif Embedding Modelleri:
OpenAI haricinde, SentenceTransformer tabanlı alternatif modeller kullanılabilmekte; kullanıcı tercihine göre model geçişleri desteklenmektedir.

Citation Mapping Gelişmişlikleri:
Atıf ifadelerinin tespiti, fuzzy matching ve NER ile geliştirilmiş; cümle bazlı bölüm eşleştirmeleri yapılmaktadır.

Ek Notlar
Proje, Zotero ve PDF/TXT dosyaları üzerinden bibliyometrik analiz ve metin işleme süreçlerini entegre eden kapsamlı bir sistemdir.
Loglama sistemi sayesinde tüm kritik işlemler hem dosyaya hem de konsola yazdırılarak izlenebilir.
Her modül, gelecekteki genişletme ve gerçek model eğitimleri için temel sağlam bir altyapı sunmaktadır.
Bu README, projenin tam yapısını, modüllerin işlevlerini ve kullanım yönergelerini detaylı olarak açıklamaktadır.
 Herhangi bir soru veya ek geliştirme talebiniz olursa lütfen bildirin.
