# module_clustering_and_analysis

# Below is the implementation of the `module_clustering_and_analysis` module. This module focuses on clustering and analyzing data, particularly for text-based embeddings, using techniques like K-means clustering, TF-IDF vectorization, and cosine similarity. It also includes functionality for exporting fine-tuning datasets, preparing data for fine-tuning, and visualizing citation chains.
# 
# ---

### **`module_clustering_and_analysis.py`**

```python
import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaTokenizer
import logging
from datetime import datetime
import traceback

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('clustering_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Global Variables
# ----------------------------
cluster_results_list = []

# ----------------------------
# Clustering Analysis Function
# ----------------------------

def cluster_analysis_from_chromadb(embedding_results, n_clusters=5, output_dir="cluster_results"):
    """
    Performs clustering analysis on embedding results using KMeans.
    Args:
        embedding_results (list): List of dictionaries containing 'content' and 'bibliography'.
        n_clusters (int): Number of clusters for KMeans.
        output_dir (str): Directory to save clustering results.
    """
    try:
        texts = [result["content"] for result in embedding_results]
        bibliographies = [result["bibliography"] for result in embedding_results]
        
        # Vectorize text data
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Save clustering results
        os.makedirs(output_dir, exist_ok=True)
        cluster_data = pd.DataFrame({
            "content": texts,
            "cluster": clusters,
            "bibliography": bibliographies
        })
        cluster_file = os.path.join(output_dir, "cluster_results.csv")
        cluster_data.to_csv(cluster_file, index=False, encoding='utf-8')
        logger.info(f"✅ Clustering results saved to '{cluster_file}'")
    except Exception as e:
        logger.error(f"Error during clustering analysis: {e}")
        traceback.print_exc()

# ----------------------------
# Fine-Tuning Preparation Function
# ----------------------------

def fine_tuning_preparation():
    """
    Prepares data for fine-tuning by tokenizing text using LLaMA tokenizer.
    Returns:
        tuple: Summary string and tokenized data if successful, otherwise None.
    """
    try:
        # Load data from CSV or global list
        if os.path.exists("fine_tuning_dataset.csv"):
            df = pd.read_csv("fine_tuning_dataset.csv")
            texts = df["content"].tolist()
        else:
            texts = [record["content"] for record in cluster_results_list if "content" in record]
        
        if not texts:
            logger.warning("Insufficient data for fine-tuning.")
            return None
        
        # Tokenize data
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        tokenized_data = [tokenizer(text, truncation=True, max_length=512) for text in texts]
        token_counts = [len(item["input_ids"]) for item in tokenized_data]
        
        # Generate summary
        summary = (
            f"Total records: {len(tokenized_data)}\n"
            f"Average token count: {sum(token_counts)/len(token_counts):.2f}\n"
            f"Max token count: {max(token_counts)}\n"
            f"Min token count: {min(token_counts)}\n"
        )
        return summary, tokenized_data
    except Exception as e:
        logger.error(f"Error during fine-tuning preparation: {e}")
        traceback.print_exc()
        return None

# ----------------------------
# Export Fine-Tuning Dataset
# ----------------------------

def export_fine_tuning_dataset():
    """
    Exports the current cluster results list to a CSV file for fine-tuning.
    """
    try:
        if not cluster_results_list:
            logger.warning("No data available to create a fine-tuning dataset.")
            return
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(cluster_results_list)
        output_file = "fine_tuning_dataset.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✅ Fine-tuning dataset saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Error exporting fine-tuning dataset: {e}")
        traceback.print_exc()

# ----------------------------
# Citation Chain Visualization
# ----------------------------

def citation_chain_view(index):
    """
    Displays the citation chain for a given index in the cluster results list.
    Args:
        index (int): Index of the record in cluster_results_list.
    """
    try:
        if index < 0 or index >= len(cluster_results_list):
            logger.warning(f"Invalid index: {index}")
            return
        
        record = cluster_results_list[index]
        bib_data = record.get("bibliography", "No bibliographic data available.")
        
        # Attempt to parse bibliographic data as JSON
        try:
            bib_json = json.loads(bib_data)
            bib_str = json.dumps(bib_json, indent=2, ensure_ascii=False)
        except Exception:
            bib_str = str(bib_data)
        
        # Display citation chain report
        chain_report = f"Citation Chain Report (Record {index}):\n{bib_str}\n"
        logger.info(chain_report)
    except Exception as e:
        logger.error(f"Error viewing citation chain: {e}")
        traceback.print_exc()

# ----------------------------
# Data Search Function
# ----------------------------

def data_search(query):
    """
    Searches the cluster results list for content matching the query.
    Args:
        query (str): Search query.
    """
    try:
        if not query:
            logger.warning("Search query cannot be empty.")
            return
        
        texts = [result["content"] for result in cluster_results_list]
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(X, query_vec).flatten()
        top_indices = sims.argsort()[-10:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "Index": idx,
                "Similarity": round(sims[idx], 4),
                "Summary": texts[idx][:100] + "..."
            })
        
        # Display search results
        df_results = pd.DataFrame(results)
        logger.info(f"Search Results:\n{df_results.to_string(index=False)}")
    except Exception as e:
        logger.error(f"Error during data search: {e}")
        traceback.print_exc()

# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    # Example: Perform clustering analysis
    sample_data = [
        {"content": "This is a sample text for clustering.", "bibliography": "Sample bibliography"},
        {"content": "Another example of text data.", "bibliography": "Another bibliography"}
    ]
    cluster_analysis_from_chromadb(sample_data, n_clusters=2)
    
    # Example: Prepare data for fine-tuning
    summary, tokenized_data = fine_tuning_preparation()
    if summary:
        print(summary)
    
    # Example: Export fine-tuning dataset
    export_fine_tuning_dataset()
    
    # Example: View citation chain
    citation_chain_view(0)
    
    # Example: Perform data search
    data_search("sample text")
# ```
# 
# ---
# 
# ### **Module Description**
# 
# 1. **`cluster_analysis_from_chromadb`**:
#    - Performs KMeans clustering on text embeddings.
#    - Saves clustering results to a CSV file.
# 
# 2. **`fine_tuning_preparation`**:
#    - Prepares data for fine-tuning by tokenizing text using the LLaMA tokenizer.
#    - Generates a summary of token counts.
# 
# 3. **`export_fine_tuning_dataset`**:
#    - Exports the current cluster results list to a CSV file for fine-tuning.
# 
# 4. **`citation_chain_view`**:
#    - Displays the citation chain for a given index in the cluster results list.
# 
# 5. **`data_search`**:
#    - Searches the cluster results list for content matching a query using TF-IDF vectorization and cosine similarity.
# 
# ---
# 
# ### **Features**
# 
# - **Clustering**: Uses KMeans clustering to group similar text embeddings.
# - **Fine-Tuning Preparation**: Tokenizes text data for use in fine-tuning models.
# - **Citation Chain Visualization**: Displays bibliographic data for a given record.
# - **Data Search**: Allows searching for similar content using cosine similarity.
# 
# ---

This module provides robust tools for clustering, analyzing, and preparing text data for further processing. If you need any additional features or modifications, feel free to ask!