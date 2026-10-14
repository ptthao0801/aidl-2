"""
class_mapping.py
=================
"""

from pathlib import Path
from dataclasses import dataclass, field


# 5 mutual classes
UNIFIED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]

# Mapping TrashNet
TRASHNET_MAPPING = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": None, 
}

# Mapping RealWaste
REALWASTE_MAPPING = {
    "Cardboard": "cardboard",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
    "Food Organics": None,       # unavailable in TrashNet
    "Miscellaneous Trash": None,  
    "Textile Trash": None,        # unavailable in TrashNet
    "Vegetation": None,           # unavailable in TrashNet
}


@dataclass
class ClassMappingAuditResult:
    """Audit result after mapping"""
    dataset_name: str
    counts_before: dict = field(default_factory=dict)  # {label_orig: count}
    counts_after: dict = field(default_factory=dict)   # {label_final: count}
    excluded_labels: list = field(default_factory=list)
    total_before: int = 0
    total_after: int = 0


def count_images_per_class(root_dir: str, mapping: dict) -> ClassMappingAuditResult:
    """
    Count after mapping.
    """
    root = Path(root_dir)
    dataset_name = root.parent.name if root.name in ("dataset-resized",) else root.name

    result = ClassMappingAuditResult(dataset_name=dataset_name)
    counts_after = {c: 0 for c in UNIFIED_CLASSES}

    valid_exts = {".jpg", ".jpeg", ".png"}

    for original_label, unified_label in mapping.items():
        class_dir = root / original_label
        if not class_dir.exists():
            print(f"[WARN] Cannot find: {class_dir}")
            continue

        n_images = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_exts
        )

        result.counts_before[original_label] = n_images
        result.total_before += n_images

        if unified_label is None:
            result.excluded_labels.append(original_label)
        else:
            counts_after[unified_label] += n_images

    result.counts_after = counts_after
    result.total_after = sum(counts_after.values())
    return result


def print_audit_report(result: ClassMappingAuditResult) -> None:
    """Print audit report to console/log"""
    print(f"\n{'=' * 60}")
    print(f"CLASS MAPPING AUDIT — {result.dataset_name}")
    print(f"{'=' * 60}")
    print(f"\nImage counts (original labels):")
    for label, count in result.counts_before.items():
        marker = " [EXCLUDED]" if label in result.excluded_labels else ""
        print(f"  {label:25s}: {count:5d}{marker}")
    print(f"  {'TOTAL (raw)':25s}: {result.total_before:5d}")

    print(f"\nImage counts (5 mutual classes):")
    for label, count in result.counts_after.items():
        print(f"  {label:25s}: {count:5d}")
    print(f"  {'TOTAL (filtered)':25s}: {result.total_after:5d}")

    print(f"\nExcluded labels: {result.excluded_labels}")
    print(f"Acceptance rate: {result.total_after / result.total_before * 100:.1f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print("Unified classes:", UNIFIED_CLASSES)
    print("\nTrashNet mapping:", TRASHNET_MAPPING)
    print("\nRealWaste mapping:", REALWASTE_MAPPING)