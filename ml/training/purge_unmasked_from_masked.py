"""
Remove face crops in a 'masked' folder that Keras classifies as clearly unmasked.

Default target: data/crop_export/roboflow_faces/masked

Moves (not permanently deletes) mislabeled files to:
  <folder>_removed_unmasked/

Run from project root:
  python ml/training/purge_unmasked_from_masked.py
  python ml/training/purge_unmasked_from_masked.py --folder "data/masked_clean"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT / "ml" / "models" / "mask_detector.keras"
DEFAULT_FOLDER = PROJECT / "data" / "crop_export" / "roboflow_faces" / "masked"

IMG_SIZE = (160, 160)
MASKED_THRESHOLD = 0.10
UNMASKED_THRESHOLD = 0.55
MIN_FACE_SIDE = 20
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def classify_crop(model, crop: np.ndarray) -> tuple[str, float]:
    if crop is None or crop.size == 0:
        return "Unknown", 0.0
    h, w = crop.shape[:2]
    if min(h, w) < MIN_FACE_SIDE:
        return "Unknown", 0.0
    resized = cv2.resize(crop, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(rgb, axis=0).astype(np.float32)
    pred = float(model(arr, training=False)[0][0])
    if pred >= UNMASKED_THRESHOLD:
        return "Unmasked", pred
    if pred <= MASKED_THRESHOLD:
        return "Masked", 1.0 - pred
    return "Unknown", pred


def main() -> int:
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete instead of moving to _removed_unmasked",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return 1

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return 1

    print(f"Loading {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    removed_dir = folder.parent / f"{folder.name}_removed_unmasked"
    if not args.delete:
        removed_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    kept = removed = unknown = 0

    for path in files:
        img = cv2.imread(str(path))
        label, conf = classify_crop(model, img)
        if label == "Unmasked":
            if args.delete:
                path.unlink()
            else:
                shutil.move(str(path), str(removed_dir / path.name))
            removed += 1
        elif label == "Masked":
            kept += 1
        else:
            unknown += 1

    print(f"\nFolder: {folder.resolve()}")
    print(f"  total:    {len(files)}")
    print(f"  kept:     {kept} (Keras says Masked)")
    print(f"  removed:  {removed} (Keras says Unmasked)")
    print(f"  unknown:  {unknown} (gray zone — left in place)")
    if removed and not args.delete:
        print(f"  moved to: {removed_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
