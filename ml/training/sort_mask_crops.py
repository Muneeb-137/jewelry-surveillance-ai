"""
Sort face crops in data/crop_export/sort/ into masked vs unmasked using mask_detector.keras.

Outputs (copies):
  data/crop_export/masked 2/
  data/crop_export/unmasked 2/
  data/crop_export/unknown 2/   (Keras uncertain — review manually)

Run from project root:
  python ml/training/sort_mask_crops.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[2]
SORT_DIR = PROJECT / "data" / "crop_export" / "sort"
OUT_MASKED = PROJECT / "data" / "crop_export" / "masked 2"
OUT_UNMASKED = PROJECT / "data" / "crop_export" / "unmasked 2"
OUT_UNKNOWN = PROJECT / "data" / "crop_export" / "unknown 2"
MODEL_PATH = PROJECT / "ml" / "models" / "mask_detector.keras"

MASK_IMG_SIZE = (160, 160)
MASKED_THRESHOLD = 0.10
UNMASKED_THRESHOLD = 0.55
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def classify_image(model, img_path: Path) -> tuple[str, float, float]:
    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        return "Invalid", 0.0, 0.0

    resized = cv2.resize(img, MASK_IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(rgb, axis=0).astype(np.float32)
    pred = float(model(arr, training=False)[0][0])

    if pred >= UNMASKED_THRESHOLD:
        return "Unmasked", pred, pred
    if pred <= MASKED_THRESHOLD:
        return "Masked", pred, float(1.0 - pred)
    return "Unknown", pred, 0.0


def unique_dest(folder: Path, name: str) -> Path:
    dest = folder / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 1
    while True:
        candidate = folder / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def main() -> int:
    os.chdir(PROJECT)
    SORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASKED.mkdir(parents=True, exist_ok=True)
    OUT_UNMASKED.mkdir(parents=True, exist_ok=True)
    OUT_UNKNOWN.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in SORT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not images:
        print(f"No images found in {SORT_DIR}")
        print("Add crops to data/crop_export/sort/ then run again.")
        return 1

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    counts = {"Masked": 0, "Unmasked": 0, "Unknown": 0, "Invalid": 0}
    print(f"Sorting {len(images)} images...\n")

    for img_path in sorted(images):
        label, pred, conf = classify_image(model, img_path)
        counts[label] = counts.get(label, 0) + 1

        if label == "Masked":
            out_dir = OUT_MASKED
        elif label == "Unmasked":
            out_dir = OUT_UNMASKED
        elif label == "Unknown":
            out_dir = OUT_UNKNOWN
        else:
            print(f"  SKIP (unreadable): {img_path.name}")
            continue

        dest = unique_dest(out_dir, img_path.name)
        shutil.copy2(img_path, dest)
        extra = f" conf={conf:.2f}" if label != "Unknown" else f" pred={pred:.3f}"
        print(f"  {label:8} {img_path.name}{extra} -> {dest.relative_to(PROJECT)}")

    print("\nDone.")
    print(f"  masked 2:   {counts.get('Masked', 0)}")
    print(f"  unmasked 2: {counts.get('Unmasked', 0)}")
    print(f"  unknown 2:  {counts.get('Unknown', 0)} (review these)")
    if counts.get("Invalid"):
        print(f"  skipped:    {counts.get('Invalid', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
