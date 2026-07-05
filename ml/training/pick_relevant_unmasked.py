"""
Pick Roboflow unmasked images relevant for beard + distant false-positive fixes.

Reads:  data/crop_export/newby unmasked/
Writes:  data/crop_export/unmasked final/relevant_from_roboflow/
         data/crop_export/relevant_unmasked_roboflow_manifest.csv

Selection criteria:
  - distant: min side <= 80px (surveillance-scale crops)
  - medium_far: min side 80-120px
  - beard_hint: filename contains beard/facial/goatee/stubble OR dark lower-face
    texture in bottom 40% of crop (heuristic)
  - current_model_fp: current Keras says Masked with conf >= 0.85 (hard negatives)
  - dedupe: one file per Roboflow base id (.rf. augmentations collapsed)
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "data" / "crop_export" / "newby unmasked"
DEST = PROJECT / "data" / "crop_export" / "unmasked final" / "relevant_from_roboflow"
MANIFEST = PROJECT / "data" / "crop_export" / "relevant_unmasked_roboflow_manifest.csv"
MODEL_PATH = PROJECT / "ml" / "models" / "mask_detector.keras"

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MASKED_T = 0.10
UNMASKED_T = 0.55
IMG_SIZE = (160, 160)

BEARD_NAME_RE = re.compile(
    r"beard|facial.?hair|goatee|stubble|mustache|moustache|chin",
    re.I,
)


def roboflow_base(stem: str) -> str:
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem


def lower_face_texture_std(crop: np.ndarray) -> float:
    h, w = crop.shape[:2]
    if h < 10:
        return 0.0
    y0 = int(h * 0.55)
    region = crop[y0:h, :]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def keras_pred(model, crop: np.ndarray) -> tuple[str, float, float]:
    resized = cv2.resize(crop, IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(rgb, axis=0).astype(np.float32)
    p = float(model(arr, training=False)[0][0])
    if p >= UNMASKED_T:
        return "Unmasked", p, p
    if p <= MASKED_T:
        return "Masked", p, float(1 - p)
    return "Unknown", p, 0.0


def score_relevance(row: dict) -> int:
    s = 0
    if row["tag_distant"]:
        s += 3
    if row["tag_medium_far"]:
        s += 2
    if row["tag_beard_name"]:
        s += 3
    if row["tag_beard_texture"]:
        s += 2
    if row["tag_model_false_masked"]:
        s += 4
    return s


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source folder: {SRC}")

    DEST.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in EXTS)
    print(f"Scanning {len(paths)} images in {SRC.name}...")

    model = tf.keras.models.load_model(str(MODEL_PATH))

    # Best candidate per Roboflow base (prefer highest relevance score)
    best_by_base: dict[str, tuple[int, Path, dict]] = {}

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        h, w = img.shape[:2]
        side = min(h, w)
        base = roboflow_base(p.stem)
        tex = lower_face_texture_std(img)
        label, pred, conf = keras_pred(model, img)

        row = {
            "source": str(p.relative_to(PROJECT)),
            "filename": p.name,
            "roboflow_base": base,
            "width": w,
            "height": h,
            "min_side_px": side,
            "tag_distant": side <= 80,
            "tag_medium_far": 80 < side <= 120,
            "tag_beard_name": bool(BEARD_NAME_RE.search(p.stem)),
            "tag_beard_texture": tex >= 28.0,
            "keras_label": label,
            "keras_pred": round(pred, 4),
            "keras_conf": round(conf, 3),
            "tag_model_false_masked": label == "Masked" and conf >= 0.85,
            "lower_face_texture_std": round(tex, 1),
        }
        rel = score_relevance(row)
        row["relevance_score"] = rel

        if rel == 0:
            continue

        prev = best_by_base.get(base)
        if prev is None or rel > prev[0]:
            best_by_base[base] = (rel, p, row)

    # Keep score >= 2 (at least one strong signal)
    selected = [
        (score, path, row)
        for score, path, row in best_by_base.values()
        if score >= 2
    ]
    selected.sort(key=lambda x: (-x[0], x[2]["min_side_px"]))

    # Copy to destination
    copied = 0
    for score, path, row in selected:
        tags = []
        if row["tag_distant"]:
            tags.append("distant")
        if row["tag_medium_far"]:
            tags.append("medium")
        if row["tag_beard_name"]:
            tags.append("beard_name")
        if row["tag_beard_texture"]:
            tags.append("beard_tex")
        if row["tag_model_false_masked"]:
            tags.append("fp_masked")
        tag_str = "_".join(tags) if tags else "relevant"
        dest_name = f"rf_{tag_str}_s{row['min_side_px']}_{path.name}"
        dest_path = DEST / dest_name
        if not dest_path.exists():
            shutil.copy2(path, dest_path)
            copied += 1
        row["dest"] = str(dest_path.relative_to(PROJECT))
        row["tags"] = tag_str

    # Write manifest for all selected
    fields = [
        "relevance_score", "tags", "dest", "source", "filename", "roboflow_base",
        "min_side_px", "keras_label", "keras_pred", "keras_conf",
        "lower_face_texture_std", "tag_distant", "tag_medium_far",
        "tag_beard_name", "tag_beard_texture", "tag_model_false_masked",
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for _, _, row in selected:
            w.writerow(row)

    # Summary
    print(f"\nSelected {len(selected)} unique bases (deduped from {len(paths)} files)")
    print(f"Copied {copied} new files to:\n  {DEST}")
    print(f"Manifest: {MANIFEST}")

    def count_tag(key):
        return sum(1 for _, _, r in selected if r.get(key))

    print("\nBreakdown:")
    print(f"  distant (<=80px):     {count_tag('tag_distant')}")
    print(f"  medium_far (81-120):  {count_tag('tag_medium_far')}")
    print(f"  beard in filename:    {count_tag('tag_beard_name')}")
    print(f"  beard texture heuristic:{count_tag('tag_beard_texture')}")
    print(f"  current model FP:     {count_tag('tag_model_false_masked')}")
    print(f"  score >= 4:           {sum(1 for s,_,_ in selected if s >= 4)}")


if __name__ == "__main__":
    main()
