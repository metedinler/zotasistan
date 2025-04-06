# clustering_module.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from embeddingmodule import EmbeddingManager

class ClusteringManager:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()

    def cluster_embeddings(self, embeddings, n_clusters=5):
        """
        Verilen embedding vektörlerini kümelendirir.

        Args:
            embeddings (list): Kümelendirilecek embedding vektörlerinin listesi.
            n_clusters (int): Küme sayısı. Varsayılan 5.

        Returns:
            list: Her kümenin indekslerinin listesi.
        """
        try:
            # Standartlaştırma
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings)

            # PCA ile boyut indirgeme (opsiyonel)
            pca = PCA(n_components=0.95)  # %95 varyans açıklanana kadar boyut indirgeme
            pca_embeddings = pca.fit_transform(scaled_embeddings)

            # K-Means kümeleme
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(pca_embeddings)

            # Silhouette skoru ile kümeleme kalitesi değerlendirme
            silhouette = silhouette_score(pca_embeddings, cluster_labels)
            print(f"Silhouette Score: {silhouette}")

            return cluster_labels
        except Exception as e:
            print(f"Kümeleme hatası: {e}")
            return None

    def visualize_clusters(self, embeddings, cluster_labels):
        """
        Kümelendirilmiş verileri görselleştirir.

        Args:
            embeddings (list): Embedding vektörlerinin listesi.
            cluster_labels (list): Her kümenin indekslerinin listesi.
        """
        try:
            # PCA ile 2 boyutlu indirgeme
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            # Görselleştirme
            plt.figure(figsize=(10, 8))
            for label in np.unique(cluster_labels):
                cluster_points = pca_embeddings[cluster_labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
            plt.title("Kümeleme Sonuçları")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ClusteringManager sınıfını kullanmak için:
clustering_manager = ClusteringManager()
embeddings = []  # Örnek embedding verileri
cluster_labels = clustering_manager.cluster_embeddings(embeddings)
if cluster_labels is not None:
    clustering_manager.visualize_clusters(embeddings, cluster_labels)
else:
    print("Kümeleme işlemi başarısız.")
