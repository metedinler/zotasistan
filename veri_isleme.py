# veri_isleme.py

import json
import networkx as nx
from configmodule import Config

config = Config()

class CitationChainProcessor:
    def __init__(self):
        self.logger = config.logger

    def create_citation_graph(self, citation_mappings):
        """
        Atıf zinciri için bir yönlendirilmiş grafik oluşturur.

        Args:
            citation_mappings (list): Atıf mapping verilerinin listesi.

        Returns:
            nx.DiGraph: Atıf zinciri grafiği.
        """
        try:
            graph = nx.DiGraph()
            for mapping in citation_mappings:
                citation = mapping.get("citation")
                reference = mapping.get("reference")
                if citation and reference:
                    graph.add_edge(citation, reference)
            self.logger.info("✅ Atıf zinciri grafiği başarıyla oluşturuldu.")
            return graph
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği oluşturulamadı: {e}")
            return None

    def save_graph_to_file(self, graph, output_path):
        """
        Atıf zinciri grafiğini bir dosyaya kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            nx.write_gml(graph, output_path)
            self.logger.info(f"✅ Atıf zinciri grafiği başarıyla kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği kaydedilemedi: {e}")

    def analyze_citation_graph(self, graph):
        """
        Atıf zinciri grafiğini analiz eder.

        Args:
            graph (nx.DiGraph): Analiz edilecek atıf zinciri grafiği.

        Returns:
            dict: Analiz sonuçları (örneğin, düğüm sayısı, kenar sayısı, en uzun yol).
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            longest_path = nx.dag_longest_path(graph) if nx.is_directed_acyclic_graph(graph) else []
            
            analysis_results = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "longest_path": longest_path,
                "longest_path_length": len(longest_path)
            }
            
            self.logger.info("✅ Atıf zinciri grafiği başarıyla analiz edildi.")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği analiz edilemedi: {e}")
            return None

    def save_analysis_to_json(self, analysis_results, output_path):
        """
        Analiz sonuçlarını JSON formatında kaydeder.

        Args:
            analysis_results (dict): Kaydedilecek analiz sonuçları.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"✅ Analiz sonuçları JSON formatında kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"Analiz sonuçları JSON formatında kaydedilemedi: {e}")

# CitationChainProcessor sınıfını kullanmak için:
processor = CitationChainProcessor()

# Örnek atıf mapping verileri
citation_mappings = [
    {"citation": "Smith et al., 2020", "reference": "Johnson et al., 2018"},
    {"citation": "Johnson et al., 2018", "reference": "Brown et al., 2015"}
]

# Atıf zinciri grafiğini oluştur
graph = processor.create_citation_graph(citation_mappings)

# Grafiği dosyaya kaydet
if graph:
    processor.save_graph_to_file(graph, "citation_graph.gml")

# Grafiği analiz et
analysis_results = processor.analyze_citation_graph(graph)

# Analiz sonuçlarını JSON dosyasına kaydet
if analysis_results:
    processor.save_analysis_to_json(analysis_results, "citation_analysis.json")
