# module_logging_and_status

import logging
import colorlog
from datetime import datetime
import os
from pathlib import Path

# ----------------------------
# Loglama Yapılandırması
# ----------------------------

def setup_logger(log_file="app.log", log_level=logging.DEBUG):
    """
    Logger yapılandırmasını ayarlar.
    Args:
        log_file (str): Log dosyasının adı.
        log_level (int): Log seviyesi (örneğin: logging.DEBUG, logging.INFO).
    Returns:
        logger: Yapılandırılmış logger nesnesi.
    """
    # Renkli log formatı
    log_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Logger oluştur
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Konsol handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Dosya handler
    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)  # Logs dizinini oluştur
        file_handler = logging.FileHandler(log_dir / log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    return logger


def print_status(message, status="INFO"):
    """
    Konsola renkli durum mesajı yazdırır.
    Args:
        message (str): Yazdırılacak mesaj.
        status (str): Durum türü ("INFO", "WARNING", "ERROR").
    """
    colors = {
        "INFO": "\033[92m",  # Yeşil
        "WARNING": "\033[93m",  # Sarı
        "ERROR": "\033[91m",  # Kırmızı
    }
    reset_color = "\033[0m"
    color_code = colors.get(status.upper(), "")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color_code}[{timestamp}] {status.upper()}: {message}{reset_color}")


def log_and_print(logger, message, level="info"):
    """
    Hem log dosyasına yazar hem de konsola yazdırır.
    Args:
        logger: Logger nesnesi.
        message (str): Yazdırılacak mesaj.
        level (str): Log seviyesi ("info", "warning", "error").
    """
    if level.lower() == "info":
        logger.info(message)
        print_status(message, "INFO")
    elif level.lower() == "warning":
        logger.warning(message)
        print_status(message, "WARNING")
    elif level.lower() == "error":
        logger.error(message)
        print_status(message, "ERROR")


# ----------------------------
# Örnek Kullanım
# ----------------------------

if __name__ == "__main__":
    # Logger'ı ayarla
    logger = setup_logger(log_file="app.log")

    # Durum mesajlarını test et
    log_and_print(logger, "Bu bir bilgi mesajıdır.", level="info")
    log_and_print(logger, "Bu bir uyarı mesajıdır.", level="warning")
    log_and_print(logger, "Bu bir hata mesajıdır.", level="error")
    
#     Modül Açıklaması
# setup_logger :
# Log dosyasını ve konsol çıktısını yapılandırır.
# Renkli loglama için colorlog kullanır.
# Log dosyalarını logs dizinine kaydeder.
# print_status :
# Konsola renkli durum mesajları yazdırır.
# Durum türlerine göre farklı renkler kullanır (INFO, WARNING, ERROR).
# log_and_print :
# Hem log dosyasına yazar hem de konsola renkli bir şekilde yazdırır.
# Log seviyelerini (info, warning, error) destekler.
# Özellikler
# Renkli Çıktı : Konsol çıktısı için ANSI renk kodları kullanılır.
# Log Dosyası : Tüm loglar logs/app.log dosyasına kaydedilir.
# Esneklik : Farklı log seviyeleri ve çıktı biçimleri desteklenir.
# Otomatik Dizin Oluşturma : logs dizini otomatik olarak oluşturulur.
