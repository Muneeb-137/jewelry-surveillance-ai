"""
Extract face crops from full-scene Roboflow (or any) image folders.

Uses the same OpenVINO face model as the live JewelGuard pipeline.
Each detected face is saved as its own JPG — ready for data/raw after review.

Run from project root:

  python ml/training/crop_folder_faces.py

Custom folders:

  python ml/training/crop_folder_faces.py ^
    --unmasked "data/final itt unmasked" ^
    --masked "data/finalize it mask" ^
    --out data/crop_export/roboflow_faces

Then quality-check and merge into raw:

  python ml/training/scan_crop_quality.py
  python ml/training/build_clean_crop_folders.py
  (copy masked_clean / unmasked_clean into data/raw/masked and data/raw/unmasked)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[2]
FACE_OV_MODEL_PATH = PROJECT / "ml" / "models" / "yolov8n-face_openvino_model"

DEFAULT_UNMASKED = PROJECT / "data" / "final itt unmasked"
DEFAULT_MASKED = PROJECT / "data" / "finalize it mask"
DEFAULT_OUT = PROJECT / "data" / "crop_export" / "roboflow_faces"

FACE_DET_CONF = 0.25
FACE_DET_IMGSZ = 640
FACE_DET_MAX_DET = 25
MIN_FACE_BOX_SIZE = 20
MAX_FACES_PER_IMAGE = 8  # avoid one crowd photo → hundreds of crops

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_face_model() -> YOLO:
    if not FACE_OV_MODEL_PATH.exists():
        print(f"Missing face model: {FACE_OV_MODEL_PATH}")
        sys.exit(1)
    print(f"Loading face model: {FACE_OV_MODEL_PATH}")
    return YOLO(str(FACE_OV_MODEL_PATH), task="pose")


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        print(f"Warning: folder not found: {folder}")
        return []
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return files


def detect_faces(model: YOLO, frame):
    h, w = frame.shape[:2]
    results = model(
        frame,
        imgsz=FACE_DET_IMGSZ,
        conf=FACE_DET_CONF,
        iou=0.45,
        max_det=FACE_DET_MAX_DET,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    faces = []
    for box in result.boxes:
        x1, y1, x2, y2 = (
            int(box.xyxy[0][0]),
            int(box.xyxy[0][1]),
            int(box.xyxy[0][2]),
            int(box.xyxy[0][3]),
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_FACE_BOX_SIZE or bh < MIN_FACE_BOX_SIZE:
            continue
        if bh > 0 and bw / bh < 0.30:
            continue
        conf = float(box.conf[0])
        faces.append((conf, x1, y1, x2, y2))

    faces.sort(reverse=True, key=lambda f: f[0])
    return faces[:MAX_FACES_PER_IMAGE]


def crop_and_save(
    model: YOLO,
    image_path: Path,
    out_dir: Path,
    label: str,
) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"  skip unreadable: {image_path.name}")
        return 0

    faces = detect_faces(model, frame)
    if not faces:
        return 0

    stem = image_path.stem.replace(" ", "_")[:80]
    saved = 0
    for i, (_conf, x1, y1, x2, y2) in enumerate(faces):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out_name = f"{label}_{stem}_f{i}.jpg"
        out_path = out_dir / out_name
        if out_path.exists():
            digest = hashlib.md5(crop.tobytes()).hexdigest()[:8]
            out_path = out_dir / f"{label}_{stem}_f{i}_{digest}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
    return saved


def process_folder(model: YOLO, src: Path, out_dir: Path, label: str) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(src)
    total_crops = 0
    with_faces = 0
    print(f"\n{label.upper()}: {src}")
    print(f"  source images: {len(images)}")
    print(f"  output:        {out_dir}")

    for n, img_path in enumerate(images, 1):
        n_saved = crop_and_save(model, img_path, out_dir, label)
        if n_saved:
            with_faces += 1
            total_crops += n_saved
        if n % 50 == 0 or n == len(images):
            print(f"  progress {n}/{len(images)}  crops so far: {total_crops}")

    print(f"  done: {total_crops} face crops from {with_faces}/{len(images)} images")
    return total_crops, len(images)


def main() -> int:
    os.chdir(PROJECT)
    parser = argparse.ArgumentParser(description="Crop faces from full-scene image folders")
    parser.add_argument("--unmasked", type=Path, default=DEFAULT_UNMASKED)
    parser.add_argument("--masked", type=Path, default=DEFAULT_MASKED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_unmasked = args.out / "unmasked"
    out_masked = args.out / "masked"

    model = load_face_model()
    u_crops, u_imgs = process_folder(model, args.unmasked, out_unmasked, "unmasked")
    m_crops, m_imgs = process_folder(model, args.masked, out_masked, "masked")

    print("\n=== SUMMARY ===")
    print(f"Unmasked crops: {u_crops} (from {u_imgs} images) -> {out_unmasked}")
    print(f"Masked crops:   {m_crops} (from {m_imgs} images) -> {out_masked}")
    print("\nNext steps:")
    print("  1. Spot-check a few crops in each folder (wrong labels, blur, no face)")
    print("  2. Merge with masked_clean / unmasked_clean if you like them")
    print("  3. Copy approved crops to data/raw/masked and data/raw/unmasked")
    print("  4. python ml/training/split_dataset.py")
    print("  5. python ml/training/train_mask_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
