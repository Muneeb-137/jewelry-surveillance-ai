import os
import shutil
import random
from pathlib import Path

# Original images are stored here
RAW_DIR = Path("data/raw")

# Train/validation images will be created here
PROCESSED_DIR = Path("data/processed")

# 80% training, 20% validation
TRAIN_RATIO = 0.8

# Our two classes
CLASSES = ["masked", "unmasked"]


def make_dirs():
    """
    Creates the required processed dataset folders:
    data/processed/train/masked
    data/processed/train/unmasked
    data/processed/val/masked
    data/processed/val/unmasked
    """
    for split in ["train", "val"]:
        for class_name in CLASSES:
            folder = PROCESSED_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)


def clear_processed():
    """
    Deletes old processed dataset and recreates empty folders.
    This avoids duplicate files when you run the script multiple times.
    """
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    make_dirs()


def split_class_images(class_name):
    """
    Splits one class folder into train and validation sets.
    Example:
    data/raw/masked -> data/processed/train/masked and data/processed/val/masked
    """
    source_dir = RAW_DIR / class_name

    if not source_dir.exists():
        raise Exception(f"Missing folder: {source_dir}")

    # Only accept image files
    images = [
        file for file in source_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    # Shuffle so split is random
    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:]

    # Copy training images
    for img in train_images:
        shutil.copy(img, PROCESSED_DIR / "train" / class_name / img.name)

    # Copy validation images
    for img in val_images:
        shutil.copy(img, PROCESSED_DIR / "val" / class_name / img.name)

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")


def main():
    clear_processed()

    for class_name in CLASSES:
        split_class_images(class_name)

    print("Dataset split complete.")


if __name__ == "__main__":
    main()