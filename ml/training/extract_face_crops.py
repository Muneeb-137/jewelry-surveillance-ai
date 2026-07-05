"""
Extract face/head crops from sample videos for mask model retraining.

Outputs to data/crop_export/:
  auto_masked/     - high-confidence masked (spot-check before training)
  auto_unmasked/   - high-confidence unmasked
  needs_review/    - ambiguous / likely false positives (review these first)

Also writes manifest.csv with metadata for every crop.

Run from project root:
  python ml/training/extract_face_crops.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEOS = PROJECT / "data" / "sample_videos"
EXPORT_ROOT = PROJECT / "data" / "crop_export"
MASK_MODEL_PATH = PROJECT / "ml" / "models" / "mask_detector.keras"
FACE_OV_MODEL_PATH = PROJECT / "ml" / "models" / "yolov8n-face_openvino_model"
PERSON_OV_MODEL_PATH = PROJECT / "yolov8n_openvino_model"

for sub in ("auto_masked", "auto_unmasked", "needs_review"):
    (EXPORT_ROOT / sub).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Video context (from your testing)
# ---------------------------------------------------------------------------
HIKING_VIDEOS = {
    "03.06.2026_16.38.33_REC.mp4",
    "12.06.2026_19.34.10_REC.mp4",
}
ROBBERY_VIDEOS = {
    "18.06.2026_17.44.17_REC.mp4",
    "19.06.2026_17.46.49_REC.mp4",
    "19.06.2026_17.56.23_REC.mp4",
    "19.06.2026_17.58.31_REC.mp4",
}
STORE_VIDEOS = {
    "13.06.2026_16.32.32_REC.mp4",
    "19.06.2026_12.20.40_REC.mp4",
    "19.06.2026_12.26.49_REC.mp4",
}

# ---------------------------------------------------------------------------
# Detection / sampling settings (match video-mode vision_engine)
# ---------------------------------------------------------------------------
FACE_DET_CONF = 0.25
FACE_DET_IMGSZ = 640
FACE_DET_MAX_DET = 15
MIN_FACE_BOX_SIZE = 20
PERSON_CONF = 0.40
MASK_IMG_SIZE = (160, 160)
FRAME_SKIP = 10
MAX_CROPS_PER_VIDEO = 120
MAX_HEAD_CROPS_PER_VIDEO = 30

MASKED_THRESHOLD = 0.10
UNMASKED_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
print("Loading models...")
face_model = YOLO(str(FACE_OV_MODEL_PATH), task="pose")
person_model = YOLO(str(PERSON_OV_MODEL_PATH), task="detect")
mask_model = tf.keras.models.load_model(str(MASK_MODEL_PATH))


def video_category(name: str) -> str:
    if name in HIKING_VIDEOS:
        return "hiking"
    if name in ROBBERY_VIDEOS:
        return "robbery"
    if name in STORE_VIDEOS:
        return "store"
    return "other"


def size_bucket(side: int) -> str:
    if side < 30:
        return "tiny"
    if side < 45:
        return "small"
    if side < 70:
        return "medium"
    return "large"


def classify_crop(crop: np.ndarray) -> tuple[str, float, float]:
    if crop is None or crop.size == 0:
        return "Unknown", 0.0, 0.0
    h, w = crop.shape[:2]
    if min(h, w) < MIN_FACE_BOX_SIZE:
        return "Unknown", 0.0, 0.0

    resized = cv2.resize(crop, MASK_IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(rgb, axis=0).astype(np.float32)
    pred = float(mask_model(arr, training=False)[0][0])

    if pred >= UNMASKED_THRESHOLD:
        return "Unmasked", pred, pred
    if pred <= MASKED_THRESHOLD:
        return "Masked", pred, float(1.0 - pred)
    return "Unknown", pred, 0.0


def get_face_boxes(frame: np.ndarray) -> list[dict]:
    h, w = frame.shape[:2]
    results = face_model(
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

    has_kp = (
        result.keypoints is not None
        and result.keypoints.xy is not None
        and len(result.keypoints.xy) > 0
    )

    faces = []
    for i, box in enumerate(result.boxes):
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
        is_frontal = True
        mouth_visible = False

        if has_kp and i < len(result.keypoints.xy):
            kp = result.keypoints.xy[i].cpu().numpy()
            kp_conf = None
            if result.keypoints.conf is not None and i < len(result.keypoints.conf):
                kp_conf = result.keypoints.conf[i].cpu().numpy()

            if len(kp) >= 3:
                lx, ly = float(kp[0][0]), float(kp[0][1])
                rx, ry = float(kp[1][0]), float(kp[1][1])
                nx, ny = float(kp[2][0]), float(kp[2][1])
                eyes_ok = not (lx == 0 and ly == 0) and not (rx == 0 and ry == 0)
                nose_ok = not (nx == 0 and ny == 0)
                if eyes_ok and nose_ok and ny < (ly + ry) / 2.0 - 30:
                    is_frontal = False

            if len(kp) >= 5 and kp_conf is not None and len(kp_conf) >= 5:
                lm_ok = float(kp_conf[3]) >= 0.80
                rm_ok = float(kp_conf[4]) >= 0.80
                mouth_visible = lm_ok and rm_ok

        faces.append(
            {
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "is_frontal": is_frontal,
                "mouth_visible": mouth_visible,
                "crop_type": "face",
            }
        )
    return faces


def get_person_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    results = person_model(frame, conf=PERSON_CONF, verbose=False)
    result = results[0]
    if result.boxes is None:
        return []

    boxes = []
    for box in result.boxes:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = (
            int(box.xyxy[0][0]),
            int(box.xyxy[0][1]),
            int(box.xyxy[0][2]),
            int(box.xyxy[0][3]),
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def head_crop_from_person(frame: np.ndarray, person_box: tuple[int, int, int, int]) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    head_y2 = y1 + int(box_h * 0.28)
    cx = (x1 + x2) // 2
    head_w = int(box_w * 0.50)
    hx1 = max(0, cx - head_w // 2)
    hx2 = min(w, cx + head_w // 2)
    hy1 = max(0, y1)
    hy2 = min(h, head_y2)

    if hx2 <= hx1 or hy2 <= hy1:
        return None
    crop = frame[hy1:hy2, hx1:hx2]
    if crop.shape[0] < MIN_FACE_BOX_SIZE or crop.shape[1] < MIN_FACE_BOX_SIZE:
        return None
    return crop.copy()


def trait_tags(
    crop_type: str,
    size: int,
    is_frontal: bool,
    mouth_visible: bool,
    label: str,
    mask_conf: float,
    vcat: str,
) -> str:
    tags = [crop_type, size_bucket(size)]
    if not is_frontal:
        tags.append("side_profile")
    if mouth_visible:
        tags.append("mouth_visible")
    if vcat == "hiking" and label == "Masked":
        tags.append("likely_false_positive")
    if vcat == "robbery" and label == "Masked" and mask_conf >= 0.90:
        tags.append("balaclava_candidate")
    if label == "Masked" and mask_conf >= 0.95 and mouth_visible:
        tags.append("beard_candidate")
    if size < 35 and label == "Masked":
        tags.append("distant_masked")
    if size < 35 and label == "Unmasked":
        tags.append("distant_unmasked")
    return "|".join(tags)


def suggest_folder(
    vcat: str,
    label: str,
    pred: float,
    mask_conf: float,
    mouth_visible: bool,
    size: int,
) -> str:
    if label == "Unknown":
        return "needs_review"

    # Hiking videos: people are unmasked; Masked is usually a false positive.
    if vcat == "hiking":
        if label == "Unmasked" and pred >= 0.45:
            return "auto_unmasked"
        if label == "Masked":
            return "needs_review"  # beard / distant girl / shadow cases
        return "needs_review"

    # Robbery videos: balaclavas / ski masks.
    if vcat == "robbery":
        if label == "Masked" and pred <= 0.08 and mask_conf >= 0.85:
            return "auto_masked"
        if label == "Unmasked" and pred >= 0.60:
            return "auto_unmasked"
        if label == "Masked" and mouth_visible:
            return "needs_review"  # possible beard in news clip
        return "needs_review"

    # Store videos: mixed customers/staff/robbery clips.
    if vcat == "store":
        if label == "Masked" and pred <= 0.07 and mask_conf >= 0.88:
            return "auto_masked"
        if label == "Unmasked" and pred >= 0.55:
            return "auto_unmasked"
        if label == "Masked" and mouth_visible:
            return "needs_review"
        return "needs_review"

    return "needs_review"


def save_crop(
    crop: np.ndarray,
    folder: str,
    stem: str,
    row: dict,
    manifest: list[dict],
) -> None:
    out_dir = EXPORT_ROOT / folder
    path = out_dir / f"{stem}.jpg"
    if path.exists():
        return
    cv2.imwrite(str(path), crop)
    row["path"] = str(path.relative_to(PROJECT))
    row["folder"] = folder
    manifest.append(row)


def process_video(video_path: Path, manifest: list[dict]) -> None:
    vname = video_path.name
    vcat = video_category(vname)
    vstem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  SKIP (cannot open): {vname}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    saved = 0
    head_saved = 0
    seen_keys: set[str] = set()

    print(f"Processing {vname} ({vcat}, ~{total_frames} frames)")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        faces = get_face_boxes(frame)

        for fi, face in enumerate(faces):
            if saved >= MAX_CROPS_PER_VIDEO:
                break

            x1, y1, x2, y2 = face["box"]
            crop = frame[y1:y2, x1:x2].copy()
            side = min(crop.shape[0], crop.shape[1])

            label, pred, mask_conf = classify_crop(crop)
            folder = suggest_folder(
                vcat, label, pred, mask_conf, face["mouth_visible"], side
            )
            traits = trait_tags(
                face["crop_type"],
                side,
                face["is_frontal"],
                face["mouth_visible"],
                label,
                mask_conf,
                vcat,
            )

            # Deduplicate similar crops from nearby frames / same face slot.
            key = f"{vstem}_{size_bucket(side)}_{label}_{fi % 3}_{frame_idx // 30}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            stem = (
                f"{vstem}_f{frame_idx:06d}_face{fi}_"
                f"{size_bucket(side)}{side}_{'front' if face['is_frontal'] else 'side'}_"
                f"mv{1 if face['mouth_visible'] else 0}_"
                f"pred{pred:.3f}_{label.lower()}_{folder}"
            )
            row = {
                "video": vname,
                "video_category": vcat,
                "frame": frame_idx,
                "crop_type": "face",
                "face_conf": round(face["conf"], 3),
                "size_px": side,
                "size_bucket": size_bucket(side),
                "is_frontal": face["is_frontal"],
                "mouth_visible": face["mouth_visible"],
                "keras_pred": round(pred, 4),
                "keras_label": label,
                "mask_conf": round(mask_conf, 3),
                "traits": traits,
                "suggested_folder": folder,
            }
            save_crop(crop, folder, stem, row, manifest)
            saved += 1

        # Head crops for robbery/store when face detector may miss balaclava.
        if vcat in ("robbery", "store") and head_saved < MAX_HEAD_CROPS_PER_VIDEO:
            persons = get_person_boxes(frame)
            for pi, pbox in enumerate(persons):
                if head_saved >= MAX_HEAD_CROPS_PER_VIDEO:
                    break
                hcrop = head_crop_from_person(frame, pbox)
                if hcrop is None:
                    continue
                side = min(hcrop.shape[0], hcrop.shape[1])
                label, pred, mask_conf = classify_crop(hcrop)
                folder = suggest_folder(vcat, label, pred, mask_conf, False, side)
                key = f"{vstem}_head_{size_bucket(side)}_{pi}_{frame_idx // 45}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                stem = (
                    f"{vstem}_f{frame_idx:06d}_head{pi}_"
                    f"{size_bucket(side)}{side}_"
                    f"pred{pred:.3f}_{label.lower()}_{folder}"
                )
                row = {
                    "video": vname,
                    "video_category": vcat,
                    "frame": frame_idx,
                    "crop_type": "head",
                    "face_conf": "",
                    "size_px": side,
                    "size_bucket": size_bucket(side),
                    "is_frontal": "",
                    "mouth_visible": False,
                    "keras_pred": round(pred, 4),
                    "keras_label": label,
                    "mask_conf": round(mask_conf, 3),
                    "traits": trait_tags("head", side, True, False, label, mask_conf, vcat),
                    "suggested_folder": folder,
                }
                save_crop(hcrop, folder, stem, row, manifest)
                head_saved += 1

        frame_idx += 1

    cap.release()
    print(f"  saved face={saved}, head={head_saved}")


def write_manifest(manifest: list[dict]) -> None:
    if not manifest:
        return
    path = EXPORT_ROOT / "manifest.csv"
    fields = [
        "path", "folder", "video", "video_category", "frame", "crop_type",
        "face_conf", "size_px", "size_bucket", "is_frontal", "mouth_visible",
        "keras_pred", "keras_label", "mask_conf", "traits", "suggested_folder",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Manifest: {path}")


def write_readme(counts: dict[str, int]) -> None:
    readme = EXPORT_ROOT / "README.md"
    readme.write_text(
        """# Crop export for mask model retraining

## Folders

| Folder | Meaning |
|--------|---------|
| `auto_masked/` | High-confidence masked crops (balaclava/ski mask). Spot-check before training. |
| `auto_unmasked/` | High-confidence unmasked crops. Spot-check before training. |
| `needs_review/` | **Review first** — beards, distant faces, side profiles, likely false positives. |

## What to do

1. Open `needs_review/` first and move each image to the correct final label:
   - Real mask/balaclava → later move to `data/raw/masked/`
   - Visible face/beard/distant unmasked → `data/raw/unmasked/`

2. Spot-check `auto_masked/` and `auto_unmasked/` (quick scan).

3. Copy approved crops into:
   - `data/raw/masked/`
   - `data/raw/unmasked/`

4. Re-split and train:
   ```bash
   python ml/training/split_dataset.py
   python ml/training/train_mask_model.py
   ```

## Counts from last export

"""
        + "\n".join(f"- {k}: {v}" for k, v in sorted(counts.items()))
        + "\n\nSee `manifest.csv` for full metadata (size, side profile, mouth_visible, traits).\n",
        encoding="utf-8",
    )


def main() -> int:
    os.chdir(PROJECT)
    videos = sorted(SAMPLE_VIDEOS.glob("*.mp4"))
    if not videos:
        print(f"No videos found in {SAMPLE_VIDEOS}")
        return 1

    print(f"Found {len(videos)} videos")
    manifest: list[dict] = []

    for video in videos:
        process_video(video, manifest)

    write_manifest(manifest)
    counts: dict[str, int] = {}
    for row in manifest:
        counts[row["folder"]] = counts.get(row["folder"], 0) + 1
    write_readme(counts)

    print("\nDone.")
    print(f"Export root: {EXPORT_ROOT}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print("\nReview needs_review/ first, then copy approved crops to data/raw/masked and data/raw/unmasked/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
