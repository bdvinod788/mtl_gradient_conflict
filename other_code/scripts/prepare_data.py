import sys
sys.path.append("src")

from config_old import Config
from data import download_and_process_all

if __name__ == "__main__":
    cfg = Config()
    download_and_process_all(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        processed_dir=cfg.processed_dir
    )