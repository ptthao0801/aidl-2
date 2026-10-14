"""
integrity_audit.py
====================
Rules:
- No corrupt / undersized / duplicate.

Chạy: python -m src.data_prep.integrity_audit
"""

import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image


MIN_IMAGE_DIM = 20  # minimum pixels per dimension
VALID_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class IntegrityAuditResult:
    dataset_name: str
    total_files_scanned: int = 0
    corrupt_files: list = field(default_factory=list)
    undersized_files: list = field(default_factory=list)
    duplicate_files: list = field(default_factory=list)  # list of (original, duplicate)
    valid_files: int = 0


def _file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute the SHA-256 hash of the file contents to detect duplicates."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_directory(root_dir: str, dataset_name: str) -> IntegrityAuditResult:
    """Scan all images in root_dir, checking for corrupt, undersized, or duplicate images."""
    root = Path(root_dir)
    result = IntegrityAuditResult(dataset_name=dataset_name)
    seen_hashes = {} 

    all_images = [
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ]
    result.total_files_scanned = len(all_images)

    for filepath in all_images:
        # 1. Check corrupt
        try:
            with Image.open(filepath) as img:
                img.verify()
            with Image.open(filepath) as img:
                width, height = img.size
        except Exception as e:
            result.corrupt_files.append((str(filepath), str(e)))
            continue

        # 2. Check undersize
        if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
            result.undersized_files.append((str(filepath), f"{width}x{height}"))
            continue

        # 3. Check dujplicate (hash)
        file_hash = _file_hash(filepath)
        if file_hash in seen_hashes:
            result.duplicate_files.append((seen_hashes[file_hash], str(filepath)))
        else:
            seen_hashes[file_hash] = str(filepath)

    result.valid_files = (
        result.total_files_scanned
        - len(result.corrupt_files)
        - len(result.undersized_files)
        - len(result.duplicate_files)
    )
    return result


def print_audit_report(result: IntegrityAuditResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"INTEGRITY AUDIT — {result.dataset_name}")
    print(f"{'=' * 60}")
    print(f"Total scanned files: {result.total_files_scanned}")
    print(f"File corrupt: {len(result.corrupt_files)}")
    for fp, err in result.corrupt_files[:5]:
        print(f"    - {fp}: {err}")
    if len(result.corrupt_files) > 5:
        print(f"    ... and {len(result.corrupt_files) - 5} other files")

    print(f"File undersize ({MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}px): "
          f"{len(result.undersized_files)}")
    for fp, dim in result.undersized_files[:5]:
        print(f"    - {fp}: {dim}")

    print(f"File duplicate: {len(result.duplicate_files)}")
    for orig, dup in result.duplicate_files[:5]:
        print(f"    - {dup} duplicate with {orig}")

    print(f"\nFile ACCEPTED after audit: {result.valid_files}")
    print(f"Acceptance rate: {result.valid_files / result.total_files_scanned * 100:.2f}%")
    print(f"{'=' * 60}\n")


def save_audit_log(result: IntegrityAuditResult, output_path: str) -> None:
    """Save log details to file text (for outputs/logs/)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"INTEGRITY AUDIT LOG — {result.dataset_name}\n")
        f.write(f"Total scanned: {result.total_files_scanned}\n")
        f.write(f"Valid: {result.valid_files}\n")
        f.write(f"Corrupt: {len(result.corrupt_files)}\n")
        f.write(f"Undersized: {len(result.undersized_files)}\n")
        f.write(f"Duplicates: {len(result.duplicate_files)}\n\n")

        f.write("--- Corrupt files ---\n")
        for fp, err in result.corrupt_files:
            f.write(f"{fp}\t{err}\n")

        f.write("\n--- Undersized files ---\n")
        for fp, dim in result.undersized_files:
            f.write(f"{fp}\t{dim}\n")

        f.write("\n--- Duplicate files ---\n")
        for orig, dup in result.duplicate_files:
            f.write(f"{dup}\tduplicate_of\t{orig}\n")

    print(f"[SAVED] Audit log: {output_path}")


def save_exclusion_list(result: IntegrityAuditResult, output_path: str) -> None:
    """
    Save the list of files to remove (corrupt, undersized, and both files
    of each duplicate pair) into a separate text file. This allows build_splits.py
    to read it and automatically skip them during image collection.

    """
    excluded = set()

    for fp, _ in result.corrupt_files:
        excluded.add(fp)
    for fp, _ in result.undersized_files:
        excluded.add(fp)
    for orig, dup in result.duplicate_files:
        excluded.add(orig)  # remove original
        excluded.add(dup)   # remove duplicate

    with open(output_path, "w", encoding="utf-8") as f:
        for fp in sorted(excluded):
            f.write(f"{fp}\n")

    print(f"[SAVED] Exclusion list ({len(excluded)} file that needs removing): {output_path}")
    return excluded


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]

    trashnet_result = audit_directory(
        str(BASE / "data" / "raw" / "trashnet" / "dataset-resized"),
        "TrashNet"
    )
    print_audit_report(trashnet_result)
    save_audit_log(trashnet_result, str(BASE / "outputs" / "logs" / "E0_integrity_audit_trashnet.txt"))
    save_exclusion_list(trashnet_result, str(BASE / "outputs" / "logs" / "E0_exclusion_list_trashnet.txt"))

    realwaste_result = audit_directory(
        str(BASE / "data" / "raw" / "realwaste" / "RealWaste"),
        "RealWaste"
    )
    print_audit_report(realwaste_result)
    save_audit_log(realwaste_result, str(BASE / "outputs" / "logs" / "E0_integrity_audit_realwaste.txt"))
    save_exclusion_list(realwaste_result, str(BASE / "outputs" / "logs" / "E0_exclusion_list_realwaste.txt"))

    print("\n[NEXT STEP] Run build_splits.py — it will read")
    print("E0_exclusion_list_*.txt and remove files before spliting.")