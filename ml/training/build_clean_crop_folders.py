"""Copy keep + review crops into cleaned training folders (drops remove list)."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "data"
CSV_PATH = BASE / "crop_quality_review" / "quality_report.csv"

OUT = {
    "masked": BASE / "masked_clean",
    "unmasked": BASE / "unmasked_clean",
}


def main() -> int:
    if not CSV_PATH.exists():
        print(f"Run scan_crop_quality.py first. Missing: {CSV_PATH}")
        return 1

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    for folder in OUT.values():
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.iterdir():
            if old.is_file():
                old.unlink()

    copied = {"masked": 0, "unmasked": 0}
    removed = {"masked": 0, "unmasked": 0}

    for row in rows:
        label = row["folder"]
        if row["recommendation"] == "remove":
            removed[label] += 1
            continue
        src = Path(row["path"])
        if not src.exists():
            print(f"Missing source, skip: {src}")
            continue
        shutil.copy2(src, OUT[label] / src.name)
        copied[label] += 1

    print("Cleaned folders ready:")
    for label, dest in OUT.items():
        print(f"  {dest} -> {copied[label]} kept, {removed[label]} removed")
    print(f"Total: {sum(copied.values())} kept, {sum(removed.values())} removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
