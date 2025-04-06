# main.py

import os
import logging
from configmodule import Config
from guimodule import GUIManager
from processingmanager import ProcessingManager
from citationmappingmodule import CitationMapper
from veri_gorsellestirme import VisualizationManager

config = Config()
logger = logging.getLogger(__name__)

def process_pdf_workflow(pdf_path):
    """
    PDF işleme, embedding oluşturma, atıf zinciri analizi ve görselleştirme işlemlerini koordine eder.

    Args:
        pdf_path (str): İşlenecek PDF dosyasının yolu.
    """
    try:
        logger.info(f"İşlem başlatıldı: {pdf_path}")

        # 1. PDF İşleme
        processing_manager = ProcessingManager()
        results = processing_manager.process_pdf(pdf_path)
        if not results:
            logger.error("PDF işleme başarısız.")
            return

        pdf_id = os.path.basename(pdf_path).split(".")[0]

        # 2. Verileri Kaydetme
        processing_manager.save_data_to_chromadb(results, pdf_id)

        # 3. Atıf Zinciri Analizi
        citation_mapper = CitationMapper()
        citation_mappings = citation_mapper.map_citations(
            results["text"], results["references"], results["sections"]
        )
        if citation_mappings:
            citation_graph = citation_mapper.create_citation_graph(citation_mappings)
            citation_mapper.save_citation_mapping(citation_mappings, f"{pdf_id}_citation_mapping.json")

            # 4. Görselleştirme
            viz_manager = VisualizationManager()
            viz_manager.visualize_citation_graph(citation_graph, f"{pdf_id}_citation_graph.png")
            viz_manager.save_graph_as_vosviewer(citation_graph, f"{pdf_id}_citation_graph.vos")

            logger.info("Atıf zinciri analizi ve görselleştirme tamamlandı.")
        else:
            logger.warning("Atıf zinciri analizi yapılamadı.")

    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {e}")

def main():
    """
    Programın ana giriş noktasıdır. Kullanıcı arayüzünü başlatır.
    """
    try:
        logger.info("Program başlatılıyor...")
        
        # Kullanıcı arayüzünü başlat
        app = GUIManager()
        app.mainloop()

    except Exception as e:
        logger.error(f"Program çalıştırılırken hata oluştu: {e}")

if __name__ == "__main__":
    main()
