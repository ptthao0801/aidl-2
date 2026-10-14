"""
download_datasets.py
=====================

TrashNet:
  source: https://github.com/garythung/trashnet 
  License: MIT 

RealWaste:
  source: https://archive.ics.uci.edu/dataset/908/realwaste
  License: CC BY-NC-SA 4.0 — non-commercial use.

Chạy: python -m src.data_prep.download_datasets
"""

import os
import ssl
import zipfile
import urllib.request
from pathlib import Path

TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
REALWASTE_ZIP_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MAX_RETRIES = 3


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[SKIP] Already existed: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {url} -> {dest}")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024 

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            pct = downloaded / total_size * 100
                            print(f"\r    {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB ({pct:.1f}%)",
                                  end="", flush=True)
                print() 

            print(f"[OK] Downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return

        except Exception as e:
            last_error = e
            print(f"\n[RETRY {attempt}/{MAX_RETRIES}] Error: {e}")
            if dest.exists():
                dest.unlink() 
            if attempt < MAX_RETRIES:
                import time
                time.sleep(3)

    raise RuntimeError(
        f"Failed to download after {MAX_RETRIES} tries: {url}\n"
        f"Last error: {last_error}\n"
    )


def _unzip(zip_path: Path, extract_to: Path) -> None:
    print(f"[UNZIP] {zip_path} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[OK] Unzip and extract to: {extract_to}")


def download_trashnet() -> Path:
    """
    Download TrashNet 'dataset-resized' 384x512. resize again in build_splits.py to 224x224 for input ImageNet-pretrained.
    """
    dest_zip = RAW_DIR / "trashnet" / "dataset-resized.zip"
    _download(TRASHNET_URL, dest_zip)
    _unzip(dest_zip, RAW_DIR / "trashnet")
    extracted_path = RAW_DIR / "trashnet" / "dataset-resized"
    print(f"[INFO] TrashNet ready at: {extracted_path}")
    return extracted_path


def download_realwaste() -> Path:
    dest_zip = RAW_DIR / "realwaste" / "realwaste.zip"
    _download(REALWASTE_ZIP_URL, dest_zip)
    _unzip(dest_zip, RAW_DIR / "realwaste")
    print(f"[INFO] RealWaste ready at: {RAW_DIR / 'realwaste'}")
    return RAW_DIR / "realwaste"


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOAD DATASET — TrashNet + RealWaste")
    print("=" * 60)

    trashnet_path = download_trashnet()
    realwaste_path = download_realwaste()
