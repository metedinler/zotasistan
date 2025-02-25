import re
import json
from datetime import datetime
from config_module import config
from rapidfuzz import fuzz
import spacy
from helper_module import shorten_title

# SpaCy modelini yükle (en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    config.logger.error("SpaCy modeli yüklenemedi. Lütfen 'python -m spacy download en_core_web_sm' komutunu çalıştırın.")
    raise

def split_into_sentences(text):
    """
    Metni cümlelere böler ve her cümleye sıra numarası ekler.
    
    Args:
        text (str): İşlenecek metin.
        
    Returns:
        list: Her biri (cümle_no, cümle) tuple'su içeren liste.
    """
    # Basit cümle bölme: nokta, ünlem veya soru işareti sonrası boşluk
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [(i, s.strip()) for i, s in enumerate(sentences) if s.strip()]

def extract_citations_from_sentence(sentence):
    """
    Verilen cümledeki atıf ifadelerini regex, fuzzy matching ve spaCy NER kullanarak tespit eder.
    
    Args:
        sentence (str): İşlenecek cümle.
        
    Returns:
        list: Tespit edilen atıf ifadelerinin listesi (örn. ["(Smith et al., 2020)", "[2]"]).
    """
    citations = []
    # Regex: Parantez içi atıflar örn. (Smith et al., 2020)
    paren_pattern = r"\(([A-Za-z\s\.,&\-]+?\d{4}[a-zA-Z]?)\)"
    paren_candidates = re.findall(paren_pattern, sentence)
    
    for candidate in paren_candidates:
        candidate = candidate.strip()
        doc = nlp(candidate)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        # Fuzzy matching: örneğin "et al.," ifadesine yakınlık kontrolü
        if persons and dates:
            similarity = fuzz.partial_ratio(candidate.lower(), "et al.,")
            if similarity >= 50:
                citations.append(f"({candidate})")
            else:
                citations.append(f"({candidate})")
    
    # Regex: Köşeli parantez içi atıflar örn. [2] veya [1, 2, 3]
    bracket_pattern = r"\[([\d,\s]+)\]"
    bracket_candidates = re.findall(bracket_pattern, sentence)
    for candidate in bracket_candidates:
        candidate = candidate.strip()
        if re.fullmatch(r"[\d,\s]+", candidate):
            citations.append(f"[{candidate}]")
    return citations

def match_citation_with_references(citation_marker, references):
    """
    Tespit edilen atıf ifadesini, PDF'nin kaynakça listesindeki referanslarla eşleştirmeye çalışır.
    İlk aşamada tam eşleşme, bulunamazsa fuzzy matching ile benzerlik hesaplaması yapılır.
    
    Args:
        citation_marker (str): Örneğin, "(Smith et al., 2020)" veya "[2]".
        references (list): Kaynakça referanslarının listesi (her biri bir string).
        
    Returns:
        dict veya None: Eşleşen referansın detayları (örneğin, {'ref_no': 1, 'text': ...}) veya None.
    """
    # İlk aşama: tam eşleşme
    normalized_marker = citation_marker.strip("()[]").lower()
    for idx, ref in enumerate(references, start=1):
        if normalized_marker in ref.lower():
            return {"ref_no": idx, "text": ref}
    # İkinci aşama: fuzzy matching
    best_score = 0
    best_match = None
    for idx, ref in enumerate(references, start=1):
        score = fuzz.ratio(normalized_marker, ref.lower())
        if score > best_score:
            best_score = score
            best_match = {"ref_no": idx, "text": ref}
    if best_score >= 60:
        return best_match
    return None

def map_citations(clean_text, bibliography, section_info):
    """
    Temiz metin içindeki tüm cümleleri işleyerek, cümle numarası, cümle metni,
    tespit edilen atıf ifadesi, eşleşen referans ve bilimsel bölüm bilgisini içeren bir yapı oluşturur.
    
    Args:
        clean_text (str): Temiz metin.
        bibliography (list): PDF'nin kaynakça referansları.
        section_info (dict): Bölümlerin haritalanması bilgisi (örn. {"Introduction": (start, end), ...}).
    
    Returns:
        dict: Citation mapping verisini içeren sözlük.
    """
    mapping = {"citations": []}
    sentences = split_into_sentences(clean_text)
    for sent_no, sentence in sentences:
        citations = extract_citations_from_sentence(sentence)
        for citation in citations:
            matched = match_citation_with_references(citation, bibliography)
            # Bölüm bilgisini belirlemek için: Eğer cümle numarası, haritalandığı bölüme denk geliyorsa ekle
            section = None
            for sec, span in section_info.items():
                if span and span[0] <= sent_no <= span[1]:
                    section = sec
                    break
            mapping["citations"].append({
                "sentence_no": sent_no,
                "sentence_text": sentence,
                "citation_marker": citation,
                "matched_reference": matched,
                "section": section
            })
    return mapping

def save_citation_mapping(pdf_id, citation_mapping):
    """
    Oluşturulan Citation Mapping verilerini, bibliyometrik bilgilerle birlikte "pdf_id.citation.json"
    dosyası olarak, CITATIONS_DIR altındaki klasörde saklar.
    
    Args:
        pdf_id (str): PDF'nin temel kimliği.
        citation_mapping (dict): Elde edilen citation mapping verisi.
    
    Returns:
        Path veya None: Oluşturulan dosyanın yolu.
    """
    file_id = shorten_title(pdf_id, max_length=80)
    target_dir = config.CITATIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{file_id}.citation.json"
    target_path = target_dir / filename
    citation_mapping["olusturma_tarihi"] = datetime.now().isoformat()
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(citation_mapping, f, ensure_ascii=False, indent=2)
        config.logger.info(f"Citation mapping dosyası kaydedildi: {target_path}")
        return target_path
    except Exception as e:
        config.logger.error(f"Citation mapping dosyası kaydedilirken hata: {e}")
        return None

# Test bölümü (modülü direkt çalıştırmak için)
if __name__ == "__main__":
    test_text = (
        "This study utilized (Smith et al., 2020) method which has been cited. "
        "Other studies, such as [2], support these findings. Additional details in (Doe et al., 2019) are provided."
    )
    test_bibliography = [
        "Smith, J., et al. (2020). Title of the paper. Journal Name. DOI:10.xxxx/xxxxxx",
        "Doe, J., et al. (2019). Another paper title. Another Journal. DOI:10.xxxx/yyyyyy"
    ]
    test_section_info = {
        "Introduction": (0, 2),
        "Methods": (3, 5),
        "Results": (6, 10)
    }
    mapping = map_citations(test_text, test_bibliography, test_section_info)
    print(json.dumps(mapping, indent=2, ensure_ascii=False))
