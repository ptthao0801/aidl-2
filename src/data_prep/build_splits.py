"""
build_splits.py
=================
1. class mapping after integrity audit
2. Resize to 224x224 
3. spliting: 70% train / 15% val / 15% test (stratified)

Chạy: python -m src.data_prep.build_splits
"""

import csv
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image

from src.data_prep.class_mapping import (
    UNIFIED_CLASSES,
    TRASHNET_MAPPING,
    REALWASTE_MAPPING,
    count_images_per_class,
    print_audit_report,
)

RANDOM_SEED = 42  
TARGET_SIZE = (224, 224)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DELETE_RAW_REALWASTE_AFTER_PROCESSING = False

BASE = Path(__file__).resolve().parents[2]
RAW_DIR = BASE / "data" / "raw"
PROCESSED_DIR = BASE / "data" / "processed"
SPLITS_DIR = BASE / "data" / "splits"


def load_exclusion_list(log_path: Path) -> set:
    """
    Read the list of files to be removed. Returns an empty set if the log file does not exist.
    """
    if not log_path.exists():
        print(f"[WARN] Cannot find exclusion list: {log_path}")
        print("[WARN] Need to run 'python -m src.data_prep.integrity_audit' first.")
        return set()
    with open(log_path, "r", encoding="utf-8") as f:
        excluded = {line.strip() for line in f if line.strip()}
    print(f"[INFO] Loaded {len(excluded)} files that need removing from: {log_path}")
    return excluded


def collect_filepaths_by_unified_class(root_dir: Path, mapping: dict,
                                        excluded_paths: set = None) -> dict:
    """
    Filtering: Return the dictionary of files after mapping and exclusion.
    """
    excluded_paths = excluded_paths or set()
    result = defaultdict(list)
    valid_exts = {".jpg", ".jpeg", ".png"}
    n_skipped = 0

    for original_label, unified_label in mapping.items():
        if unified_label is None:
            continue  
        class_dir = root_dir / original_label
        if not class_dir.exists():
            print(f"[WARN] Cannot find: {class_dir}")
            continue
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                if str(f) in excluded_paths:
                    n_skipped += 1
                    continue
                result[unified_label].append(f)

    if n_skipped > 0:
        print(f"[INFO] Excluded {n_skipped} file (corrupt/undersized/duplicate) "
              f"from {root_dir.parent.name if root_dir.name != root_dir.parent.name else root_dir}")

    return result


def stratified_split(filepaths_by_class: dict, train_ratio: float,
                      val_ratio: float, test_ratio: float, seed: int):
    """
    Perform stratified splitting per class.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)

    splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    for label, filepaths in filepaths_by_class.items():
        shuffled = filepaths.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"][label] = shuffled[:n_train]
        splits["val"][label] = shuffled[n_train:n_train + n_val]
        splits["test"][label] = shuffled[n_train + n_val:]

    return splits


def resize_and_save(src_path: Path, dest_path: Path, size=TARGET_SIZE) -> bool:
    """
    Resize the image and save it with JPEG compression.
    """
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"[ERROR] Resize thất bại {src_path}: {e}")
        return False


def write_split_csv(split_data: dict, csv_path: Path, processed_root: Path,
                     split_name_for_path: str) -> None:
    """
    Write the CSV for a single split and handle copying and resizing the images.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for label, filepaths in split_data.items():
        for i, src_path in enumerate(filepaths):
            dest_filename = f"{label}_{i:04d}{src_path.suffix.lower()}"
            dest_path = processed_root / split_name_for_path / label / dest_filename

            ok = resize_and_save(src_path, dest_path)
            if ok:
                rows.append({
                    "filepath": str(dest_path.relative_to(BASE)),
                    "label": label,
                    "original_source": str(src_path)
                })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "original_source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] {csv_path} ({len(rows)} dòng)")


def main():
    print("=" * 60)
    print("STEP 0.4 — BUILD SPLITS")
    print("=" * 60)

    # ---- 1. Class mapping audit ----
    trashnet_root = RAW_DIR / "trashnet" / "dataset-resized"
   
    realwaste_root = RAW_DIR / "realwaste" / "RealWaste"

    trashnet_audit = count_images_per_class(str(trashnet_root), TRASHNET_MAPPING)
    print_audit_report(trashnet_audit)

    realwaste_audit = count_images_per_class(str(realwaste_root), REALWASTE_MAPPING)
    print_audit_report(realwaste_audit)

    # ---- 2. TrashNet: collect + stratified split 70/15/15 ----
    print("[INFO] Collecting TrashNet filepaths ...")
    trashnet_excluded = load_exclusion_list(
        BASE / "outputs" / "logs" / "E0_exclusion_list_trashnet.txt"
    )
    trashnet_files = collect_filepaths_by_unified_class(
        trashnet_root, TRASHNET_MAPPING, excluded_paths=trashnet_excluded
    )
    for label in UNIFIED_CLASSES:
        print(f"    {label}: {len(trashnet_files.get(label, []))} ảnh")

    trashnet_splits = stratified_split(
        trashnet_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed=RANDOM_SEED
    )

    for split_name in ["train", "val", "test"]:
        write_split_csv(
            trashnet_splits[split_name],
            SPLITS_DIR / f"trashnet_{split_name}.csv",
            PROCESSED_DIR / "trashnet_filtered",
            split_name,
        )

    # ---- 3. RealWaste - test ----
    print("\n[INFO] Collecting RealWaste filepaths...")
    realwaste_excluded = load_exclusion_list(
        BASE / "outputs" / "logs" / "E0_exclusion_list_realwaste.txt"
    )
    realwaste_files = collect_filepaths_by_unified_class(
        realwaste_root, REALWASTE_MAPPING, excluded_paths=realwaste_excluded
    )
    for label in UNIFIED_CLASSES:
        print(f"    {label}: {len(realwaste_files.get(label, []))} ảnh")

    write_split_csv(
        realwaste_files,
        SPLITS_DIR / "realwaste_test.csv",
        PROCESSED_DIR / "realwaste_filtered",
        "test",
    )

    if DELETE_RAW_REALWASTE_AFTER_PROCESSING:
        import shutil
        print(f"[CLEANUP] Delete raw RealWaste to free disk storage {realwaste_root}")
        shutil.rmtree(realwaste_root, ignore_errors=True)
        print("[CLEANUP] Completed. Resized images are in data/processed/")

    # ---- 4. SPLITING RESULTS ----
    print("\n" + "=" * 60)
    print("SPLITING RESULTS")
    print("=" * 60)
    print(f"{'Class':12s} {'TN-train':>10s} {'TN-val':>10s} {'TN-test':>10s} {'RW-test':>10s}")
    for label in UNIFIED_CLASSES:
        n_train = len(trashnet_splits["train"].get(label, []))
        n_val = len(trashnet_splits["val"].get(label, []))
        n_test = len(trashnet_splits["test"].get(label, []))
        n_rw = len(realwaste_files.get(label, []))
        print(f"{label:12s} {n_train:>10d} {n_val:>10d} {n_test:>10d} {n_rw:>10d}")

    total_train = sum(len(v) for v in trashnet_splits["train"].values())
    total_val = sum(len(v) for v in trashnet_splits["val"].values())
    total_test = sum(len(v) for v in trashnet_splits["test"].values())
    total_rw = sum(len(v) for v in realwaste_files.values())
    print(f"{'TOTAL':12s} {total_train:>10d} {total_val:>10d} {total_test:>10d} {total_rw:>10d}")
    print("=" * 60)


if __name__ == "__main__":
    main()