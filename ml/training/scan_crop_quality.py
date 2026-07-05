"""Scan masked/unmasked crop folders and recommend keep / review / remove."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "data"
OUT = BASE / "crop_quality_review"

FOLDERS = {
    "masked": BASE / "masked 5",
    "unmasked": BASE / "unmasked 5",
}


def classify(mean: float, blur: float, min_side: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if min_side < 22:
        reasons.append("too_small")
    if mean < 35:
        reasons.append("too_dark")
    if mean > 235:
        reasons.append("too_bright")
    if blur < 12:
        reasons.append("too_blurry")

    hard = {"too_small", "too_dark", "too_blurry"}
    if hard.intersection(reasons):
        if reasons == ["too_small"] and blur >= 40 and mean >= 50:
            pass
        elif len([r for r in reasons if r in hard]) >= 2 or "too_dark" in reasons or blur < 8:
            return "remove", reasons
        return "review", reasons

    if blur < 28 or mean < 50 or min_side < 35:
        if blur < 28 and "too_blurry" not in reasons:
            reasons.append("soft_blur")
        if mean < 50 and "too_dark" not in reasons:
            reasons.append("dim")
        if min_side < 35 and "too_small" not in reasons:
            reasons.append("small")
        return "review", reasons

    return "keep", reasons


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for label, folder in FOLDERS.items():
        if not folder.exists():
            print(f"Missing: {folder}")
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                continue
            img = cv2.imread(str(f))
            if img is None:
                rows.append(
                    {
                        "folder": label,
                        "file": f.name,
                        "path": str(f.resolve()),
                        "recommendation": "remove",
                        "reasons": "unreadable",
                        "mean_brightness": "",
                        "blur_score": "",
                        "width": "",
                        "height": "",
                        "min_side": "",
                    }
                )
                continue

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            min_side = min(w, h)
            rec, reasons = classify(mean, blur, min_side)
            rows.append(
                {
                    "folder": label,
                    "file": f.name,
                    "path": str(f.resolve()),
                    "recommendation": rec,
                    "reasons": "|".join(reasons) if reasons else "ok",
                    "mean_brightness": round(mean, 1),
                    "blur_score": round(blur, 1),
                    "width": w,
                    "height": h,
                    "min_side": min_side,
                }
            )

    if not rows:
        print("No images found.")
        return 1

    csv_path = OUT / "quality_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for rec in ("keep", "review", "remove"):
        subset = [r for r in rows if r["recommendation"] == rec]
        (OUT / f"{rec}_files.txt").write_text(
            "\n".join(r["path"] for r in subset), encoding="utf-8"
        )

    print("=== QUALITY SCAN ===")
    for label in FOLDERS:
        sub = [r for r in rows if r["folder"] == label]
        if not sub:
            continue
        print(f"\n{label.upper()} ({len(sub)} images)")
        for rec in ("keep", "review", "remove"):
            n = sum(1 for r in sub if r["recommendation"] == rec)
            pct = 100 * n / len(sub)
            print(f"  {rec:6}: {n:3} ({pct:.0f}%)")

    print(f"\nReport: {csv_path}")
    print(f"Lists:  {OUT / 'keep_files.txt'}")
    print(f"        {OUT / 'review_files.txt'}")
    print(f"        {OUT / 'remove_files.txt'}")

    for label in ("masked", "unmasked"):
        print(f"\n=== SAMPLE REMOVE ({label}) ===")
        samples = [r for r in rows if r["folder"] == label and r["recommendation"] == "remove"][:6]
        for r in samples:
            print(
                f"  {r['file']}  bright={r['mean_brightness']} "
                f"blur={r['blur_score']} min={r['min_side']}  ({r['reasons']})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
