# module_memory_and_performance_monitoring

# Below is the implementation of the `module_memory_and_performance_monitoring` module. This module focuses on monitoring memory usage, optimizing performance, and providing real-time feedback during file processing. It integrates with the existing system to ensure efficient resource utilization and error handling.
# 
# ---

### **`module_memory_and_performance_monitoring.py`**

```python
import psutil
import logging
import sys
from datetime import datetime
import multiprocessing
from tqdm import tqdm
import traceback

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler('memory_performance.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Global Variables
# ----------------------------
total_files = 0
success_count = 0
embedding_failed_count = 0
text_extraction_failed_count = 0

# ----------------------------
# Memory Monitoring Functions
# ----------------------------

def memory_usage():
    """Returns current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"{mem_info.rss / 1024 ** 2:.2f} MB"

def log_memory_usage():
    """Logs current memory usage."""
    usage = memory_usage()
    logger.info(f"📊 Current memory usage: {usage}")

def optimize_memory_usage():
    """Attempts to free up unused memory."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        if mem_info.rss > 500 * 1024 ** 2:  # If memory usage exceeds 500MB
            logger.warning("MemoryWarning: High memory usage detected. Attempting to optimize...")
            gc.collect()  # Trigger garbage collection
            logger.info("MemoryWarning: Garbage collection completed.")
    except Exception as e:
        logger.error(f"MemoryWarning: Failed to optimize memory usage: {e}")

# ----------------------------
# Performance Monitoring Functions
# ----------------------------

def print_status(current_file, total, success, embed_fail, text_fail):
    """
    Updates the console with real-time status information.
    Uses ANSI escape sequences to save and restore cursor position.
    Colors: Embed Error (red), Text Error (yellow). Success and other text are default (black).
    Total embeddings = success + embed_fail.
    """
    total_embed = success + embed_fail
    status_msg = (
        f"📊 Processing file: {current_file}/{total} | "
        f"✅ Success: {success}/{total_embed} | "
        f"\033[31m❌ Embed Errors: {embed_fail}\033[0m | "
        f"\033[33m⚠️ Text Errors: {text_fail}\033[0m"
    )
    sys.stdout.write("\033[s")  # Save cursor position
    sys.stdout.write("\033[1;1H" + status_msg.ljust(100))  # Move to top row and write message
    sys.stdout.write("\033[u")  # Restore cursor position
    sys.stdout.flush()

def monitor_performance(func):
    """
    Decorator to monitor performance and memory usage of a function.
    Logs execution time and memory usage before and after the function call.
    """
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        initial_memory = memory_usage()
        logger.info(f"PerformanceMonitor: Starting {func.__name__}. Initial memory: {initial_memory}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            final_memory = memory_usage()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(
                f"PerformanceMonitor: Completed {func.__name__}. "
                f"Duration: {duration:.2f}s, Final memory: {final_memory}"
            )
            return result
        except Exception as e:
            logger.error(f"PerformanceMonitor: Error in {func.__name__}: {e}")
            traceback.print_exc()
            raise
    return wrapper

# ----------------------------
# Parallel Processing with Monitoring
# ----------------------------

def process_files_in_parallel(file_list, process_function):
    """
    Processes files in parallel using ProcessPoolExecutor.
    Monitors memory and performance during execution.
    """
    global total_files, success_count, embedding_failed_count, text_extraction_failed_count
    
    total_files = len(file_list)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_function, file_list),
            total=total_files,
            desc="Processing Files"
        ))
    
    for success, result in results:
        if success:
            success_count += 1
        elif "embedding" in result.get("error", ""):
            embedding_failed_count += 1
        elif "text_extraction" in result.get("error", ""):
            text_extraction_failed_count += 1
    
    print_status(total_files, total_files, success_count, embedding_failed_count, text_extraction_failed_count)

# ----------------------------
# Main Function
# ----------------------------

@monitor_performance
def main():
    try:
        log_memory_usage()
        json_file_name = input("Enter the JSON file name to process (e.g., Books.json): ")
        json_file_path = Path(SUCCESS_DIR) / json_file_name

        if not json_file_path.exists():
            logger.error(f"❌ File not found: {json_file_name}")
            return

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        process_files_in_parallel(data, process_file)

        logger.info("🎉 Processing completed!")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Embedding errors: {embedding_failed_count}")
        logger.info(f"Text extraction errors: {text_extraction_failed_count}")

    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
# ```
# 
# ---
# 
# ### **Module Description**
# 
# This module provides functionality for monitoring memory usage, optimizing performance, and tracking progress during file processing. Key features include:
# 
# 1. **Memory Monitoring**:
#    - Tracks current memory usage using `psutil`.
#    - Attempts to optimize memory by triggering garbage collection when usage exceeds a threshold.
# 
# 2. **Performance Monitoring**:
#    - Measures execution time and memory usage for critical functions.
#    - Logs performance metrics before and after function execution.
# 
# 3. **Real-Time Status Updates**:
#    - Displays real-time progress in the console using `tqdm` and ANSI escape sequences.
#    - Highlights errors in red and yellow for better visibility.
# 
# 4. **Parallel Processing**:
#    - Processes files in parallel using `multiprocessing.Pool`.
#    - Ensures efficient resource utilization and faster processing.
# 
# 5. **Error Handling**:
#    - Logs detailed error messages and stack traces for debugging.
# 
# ---
# 
# ### **Usage**
# 
# 1. **Input**:
#    - A JSON file containing metadata about the files to process (e.g., file paths, titles).
# 
# 2. **Output**:
#    - Logs detailing memory usage, performance metrics, and processing status.
# 
# 3. **Execution**:
#    - Run the script and provide the JSON file name when prompted.
# 
# ---

This module is designed to enhance the efficiency and reliability of the file processing pipeline by continuously monitoring resource usage and providing actionable insights. If you need any additional features or modifications, feel free to ask!