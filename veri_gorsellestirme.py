# veri_gorsellestirme.py

import matplotlib.pyplot as plt
import networkx as nx
from configmodule import Config

config = Config()

class VisualizationManager:
    def __init__(self):
        self.logger = config.logger

    def visualize_citation_graph(self, graph, output_path=None):
        """
        Atıf zinciri grafiğini görselleştirir.

        Args:
            graph (nx.DiGraph): Görselleştirilecek atıf zinciri grafiği.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)  # Düğüm yerleşimi için spring layout kullanılır
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_size=3000,
                node_color="lightblue",
                font_size=10,
                font_weight="bold",
                edge_color="gray"
            )
            plt.title("Atıf Zinciri Grafiği", fontsize=14)
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Atıf zinciri grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Atıf zinciri grafiği görselleştirilemedi: {e}")

    def visualize_clustered_data(self, embeddings, cluster_labels, output_path=None):
        """
        Kümeleme sonuçlarını görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Küme etiketlerinin listesi.
            output_path (str, optional): Grafiğin kaydedileceği dosya yolu. Eğer None ise grafik ekranda gösterilir.
        """
        try:
            from sklearn.decomposition import PCA

            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)

            # Kümeleme sonuçlarının görselleştirilmesi
            plt.figure(figsize=(12, 8))
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                cluster_points = reduced_embeddings[cluster_labels == label]
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    label=f"Küme {label}",
                    alpha=0.7
                )
            
            plt.title("Kümeleme Sonuçları", fontsize=14)
            plt.legend()
            
            if output_path:
                plt.savefig(output_path, format="png", dpi=300)
                self.logger.info(f"✅ Kümeleme grafiği kaydedildi: {output_path}")
            else:
                plt.show()
        except Exception as e:
            self.logger.error(f"Kümeleme grafiği görselleştirilemedi: {e}")

    def save_graph_as_vosviewer(self, graph, output_path):
        """
        Atıf zinciri grafiğini VOSviewer formatında kaydeder.

        Args:
            graph (nx.DiGraph): Kaydedilecek atıf zinciri grafiği.
            output_path (str): Kaydedilecek dosyanın yolu.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("*Vertices\n")
                for i, node in enumerate(graph.nodes(), start=1):
                    f.write(f"{i} \"{node}\"\n")
                
                f.write("*Edges\n")
                for source, target in graph.edges():
                    f.write(f"{source} {target}\n")
            
            self.logger.info(f"✅ VOSviewer formatında grafik kaydedildi: {output_path}")
        except Exception as e:
            self.logger.error(f"VOSviewer formatında grafik kaydedilemedi: {e}")

# VisualizationManager sınıfını kullanmak için:
viz_manager = VisualizationManager()

# Örnek kullanım
graph = nx.DiGraph()
graph.add_edge("Makale A", "Makale B")
graph.add_edge("Makale B", "Makale C")

# Grafiği görselleştir
viz_manager.visualize_citation_graph(graph)

# Grafiği VOSviewer formatında kaydet
viz_manager.save_graph_as_vosviewer(graph, "citation_graph.vos")
