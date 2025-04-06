# citationmappingmodule.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
from configmodule import Config

config = Config()

class CitationMapper:
    def __init__(self):
        self.logger = config.logger
        self.citation_patterns = config.CITATIONPATTERNS

    def split_into_sentences(self, text):
        """
        Metni cümlelere ayırır.

        Args:
            text (str): Ayırılacak metin.

        Returns:
            list: Cümlelerin listesi.
        """
        return sent_tokenize(text)

    def extract_citations_from_sentence(self, sentence):
        """
        Cümle içindeki atıf ifadelerini regex veya fuzzy matching ile tespit eder.

        Args:
            sentence (str): Cümle.

        Returns:
            list: Tespit edilen atıf ifadelerinin listesi.
        """
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, sentence)
            citations.extend(matches)
        return citations

    def match_citation_with_references(self, citation, references):
        """
        Tespit edilen atıf ifadesini kaynakça referanslarıyla eşler.

        Args:
            citation (str): Atıf ifadesi.
            references (list): Kaynakça referansları.

        Returns:
            dict: Eşleştirilmiş atıf ifadesi ve referans bilgisi.
        """
        best_match = None
        best_score = 0
        for reference in references:
            score = fuzz.partial_ratio(citation.lower(), reference.lower())
            if score > best_score:
                best_score = score
                best_match = reference
        return {"citation": citation, "reference": best_match}

    def get_section_for_sentence(self, sentence, sections):
        """
        Cümlenin hangi bilimsel bölümde olduğunu belirler.

        Args:
            sentence (str): Cümle.
            sections (dict): Bilimsel bölümler.

        Returns:
            str: Cümlenin ait olduğu bölüm adı.
        """
        for section_name, section_text in sections.items():
            if sentence in section_text:
                return section_name
        return "Unknown"

    def map_citations(self, text, references, sections):
        """
        Metin içindeki tüm cümleler için atıf mapping verilerini oluşturur.

        Args:
            text (str): Metin.
            references (list): Kaynakça referansları.
            sections (dict): Bilimsel bölümler.

        Returns:
            list: Atıf mapping verilerinin listesi.
        """
        sentences = self.split_into_sentences(text)
        citation_mappings = []
        for sentence in sentences:
            citations = self.extract_citations_from_sentence(sentence)
            for citation in citations:
                matched_citation = self.match_citation_with_references(citation, references)
                section_name = self.get_section_for_sentence(sentence, sections)
                citation_mappings.append({
                    "sentence": sentence,
                    "citation": citation,
                    "reference": matched_citation["reference"],
                    "section": section_name
                })
        return citation_mappings

    def save_citation_mapping(self, citation_mappings, output_path):
        """
        Atıf mapping verilerini JSON formatında kaydeder.

        Args:
            citation_mappings (list): Kaydedilecek atıf mapping verilerinin listesi.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(citation_mappings, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Atıf mapping verileri kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf mapping verileri kaydedilemedi: {e}")

# CitationMapper sınıfını kullanmak için:
citation_mapper = CitationMapper()
text = "Örnek metin..."
references = ["Referans 1", "Referans 2"]
sections = {"Giriş": "Giriş metni...", "Sonuç": "Sonuç metni..."}
citation_mappings = citation_mapper.map_citations(text, references, sections)
if citation_mappings:
    output_path = "citation_mapping.json"
    citation_mapper.save_citation_mapping(citation_mappings, output_path)
else:
    print("Atıf mapping verileri oluşturulamadı.")
