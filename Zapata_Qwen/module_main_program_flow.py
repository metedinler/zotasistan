# Below is the implementation of the `module_main_program_flow` module. This module defines the main program flow for processing files, extracting metadata, generating embeddings, and integrating with Zotero and ChromaDB. It also includes error handling, logging, and progress tracking.
# 
# ---

### **`module_main_program_flow.py`**

```python
import os
import json
import fitz  # PyMuPDF
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import logging
import sys
import requests
import traceback
from datetime import datetime
import pytz
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('main_program_flow.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Environment Variables
# ----------------------------
load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")  # Zotero storage directory
SUCCESS_DIR = os.getenv("SUCCESS_DIR")  # Main success directory
PDF_DIR = Path(SUCCESS_DIR) / "pdfler" if SUCCESS_DIR else Path("pdfler")
EMBEDDING_PARCA_DIR = Path(SUCCESS_DIR) / "embedingparca" if SUCCESS_DIR else Path("embedingparca")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
ZOTERO_API_URL = f"https://api.zotero.org/users/{ZOTERO_USER_ID}/items"

# ----------------------------
# ChromaDB Configuration
# ----------------------------
try:
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
    bib_collection = chroma_client.get_or_create_collection(name="pdf_bibliography")
except Exception as e:
    logger.error(f"ChromaDB connection failed: {e}")
    chroma_client = None

# ----------------------------
# Counters and Tracking Variables
# ----------------------------
total_files = 0
success_count = 0
embedding_failed_count = 0
text_extraction_failed_count = 0

# ----------------------------
# Helper Functions
# ----------------------------

def init_dirs():
    """Creates necessary directories."""
    for directory in [PDF_DIR, EMBEDDING_PARCA_DIR]:
        os.makedirs(directory, exist_ok=True)

def save_last_processed_index(index):
    """Saves the last processed file index."""
    with open("last_processed_index.txt", "w", encoding="utf-8") as f:
        f.write(str(index))

def get_last_processed_index():
    """Reads the last processed file index."""
    if os.path.exists("last_processed_index.txt"):
        with open("last_processed_index.txt", "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        raw_text = "\n\n".join(page.get_text("text", sort=True) for page in doc)
        doc.close()
        return raw_text
    except Exception as e:
        logger.error(f"❌ Failed to extract text from PDF: {pdf_path}, Error: {e}")
        return None

def split_text(text, chunk_size=256):
    """Splits text into chunks of specified size."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    """Generates an embedding using OpenAI API."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Failed to generate embedding: {e}")
        return None

def fetch_zotero_metadata(item_key):
    """Fetches bibliographic metadata from Zotero."""
    headers = {"Zotero-API-Key": ZOTERO_API_KEY}
    try:
        response = requests.get(f"{ZOTERO_API_URL}/{item_key}", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch Zotero metadata: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Zotero API request failed: {e}")
        return None

# ----------------------------
# File Processing Function
# ----------------------------

def process_file(item):
    """
    Processes a single file, extracts text, generates embeddings, and stores metadata.
    Args:
        item (dict): Dictionary containing file metadata.
    Returns:
        tuple: Success status and log entry.
    """
    global success_count, embedding_failed_count, text_extraction_failed_count
    try:
        title = item.get("title")
        if not title:
            raise ValueError("Title not found")
        
        pdf_source_path = Path(STORAGE_DIR) / title
        if not pdf_source_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_source_path}")
        
        # Extract text
        raw_text = extract_text_from_pdf(pdf_source_path)
        if not raw_text:
            text_extraction_failed_count += 1
            raise Exception("Text extraction failed")
        
        # Split text into chunks
        chunks = split_text(raw_text)
        embeddings = []
        for chunk in chunks:
            embedding = get_embedding(chunk)
            if embedding:
                embeddings.append(embedding)
            else:
                embedding_failed_count += 1
        
        # Save embeddings to ChromaDB
        if chroma_client:
            chunk_ids = [f"{title}_{i}" for i in range(len(chunks))]
            collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=[{"title": title, "chunk_index": i} for i in range(len(chunks))]
            )
        
        # Fetch and store Zotero metadata
        item_key = item.get("key")
        if item_key:
            bib_data = fetch_zotero_metadata(item_key)
            if bib_data:
                bib_collection.add(
                    ids=[title],
                    embeddings=[[0] * 768],  # Dummy embedding
                    metadatas=[bib_data]
                )
        
        success_count += 1
        log_entry = {
            "file": title,
            "status": "success",
            "timestamp": datetime.now(pytz.timezone("Turkey")).isoformat(),
            "chunks_processed": len(chunks),
            "embeddings_generated": len(embeddings)
        }
        return True, log_entry
    except Exception as e:
        error_log = {
            "file": title if 'title' in locals() else 'unknown',
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        return False, error_log

# ----------------------------
# Main Program Flow
# ----------------------------

def main():
    global total_files
    try:
        init_dirs()
        print("\n" + "="*80)
        print("### PDF/TXT PROCESSING, CLEAN TEXT, EMBEDDING, ZOTERO, CLUSTER ANALYSIS AND MAPPING SYSTEM ###")
        print("="*80)
        json_file_name = input("Enter the JSON file name to process (e.g., Books.json): ")
        json_file_path = os.path.join(SUCCESS_DIR, json_file_name)

        if not os.path.exists(json_file_path):
            logger.error(f"❌ File not found: {json_file_name}")
            return

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_items = [item for item in data if isinstance(item, dict) and item.get("title")]
        total_files = len(valid_items)
        if total_files == 0:
            logger.error("❌ No valid records found!")
            return

        user_input = input("Start from scratch [B], continue from where you left off [C], or update [G]: ").lower()
        if user_input == "b":
            logger.warning("⚠️ Resetting database...")
            try:
                collection.delete(where={"id": {"$exists": True}})
                bib_collection.delete(where={"id": {"$exists": True}})
                collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
                bib_collection = chroma_client.get_or_create_collection(name="pdf_bibliography")
            except Exception as e:
                logger.error(f"❌ Database reset failed: {e}")
                return
            last_index = 0
        elif user_input == "c":
            last_index = get_last_processed_index()
        else:
            last_index = 0

        print(f"\nProcessing starting... ({last_index + 1}/{total_files})")
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = {executor.submit(process_file, item): item for item in valid_items[last_index:]}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
                item = futures[future]
                try:
                    success, result = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"✅ {item.get('title', 'Unknown file')} processed")
                    else:
                        logger.error(f"❌ {item.get('title', 'Unknown file')} error: {result['error']}")
                except Exception as e:
                    logger.error(f"❌ Processing error: {item.get('title', 'Unknown file')} - {str(e)}")
                save_last_processed_index(valid_items.index(item))

    except Exception as e:
        logger.error(f"Error in main program: {str(e)}")
        error_log = {
            "file": "main",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        logger.error(error_log)
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("Processing completed!")
        print(f"Total files: {total_files}")
        print(f"Successful: {success_count}")
        print(f"Embedding errors: {embedding_failed_count}")
        print(f"Text extraction errors: {text_extraction_failed_count}")
        print("="*60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
# ```
# 
# ---
# 
# ### **Module Description**
# 
# This module orchestrates the main program flow for processing files, extracting metadata, generating embeddings, and integrating with Zotero and ChromaDB. Key features include:
# 
# 1. **Directory Initialization**:
#    - Ensures all required directories exist.
# 
# 2. **Environment Configuration**:
#    - Loads environment variables for storage paths, API keys, and other settings.
# 
# 3. **Text Extraction**:
#    - Extracts raw text from PDF files using PyMuPDF.
# 
# 4. **Text Chunking**:
#    - Splits extracted text into manageable chunks for embedding generation.
# 
# 5. **Embedding Generation**:
#    - Uses OpenAI's `text-embedding-ada-002` model to generate embeddings for text chunks.
# 
# 6. **Metadata Handling**:
#    - Fetches bibliographic metadata from Zotero using its API.
# 
# 7. **ChromaDB Integration**:
#    - Stores embeddings and metadata in ChromaDB collections.
# 
# 8. **Error Handling and Logging**:
#    - Logs detailed error messages and stack traces for debugging.
# 
# 9. **Progress Tracking**:
#    - Tracks the last processed file index to allow resuming interrupted processes.
# 
# 10. **Parallel Processing**:
#     - Processes files in parallel using `ProcessPoolExecutor` for improved performance.
# 
# ---
# 
# ### **Usage**
# 
# 1. **Input**:
#    - A JSON file containing metadata about the files to process (e.g., file paths, titles, Zotero keys).
# 
# 2. **Output**:
#    - Processed files with embeddings stored in ChromaDB.
#    - Logs detailing the processing status of each file.
# 
# 3. **Execution**:
#    - Run the script and provide the JSON file name when prompted.
# 
# ---

This module is designed to handle large-scale file processing efficiently while maintaining robust error handling and progress tracking. If you need any additional features or modifications, feel free to ask!