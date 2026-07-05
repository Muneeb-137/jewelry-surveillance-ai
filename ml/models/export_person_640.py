"""Export 640px OpenVINO person model for video/RTSP (better crowd recall)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "yolov8n_openvino_model_640"


def main() -> None:
    print("Exporting yolov8n @ 640 to OpenVINO (may take a few minutes)...")
    tmp_path = Path(tempfile.mkdtemp(prefix="vaultvision_ov_"))
    try:
        os.chdir(tmp_path)
        YOLO("yolov8n.pt").export(format="openvino", imgsz=640)
        work_dir = tmp_path / "yolov8n_openvino_model"
        if not work_dir.exists():
            raise SystemExit("Export failed — OpenVINO output folder not created")

        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        shutil.copytree(work_dir, OUT_DIR)
    finally:
        os.chdir(ROOT)
        shutil.rmtree(tmp_path, ignore_errors=True)

    print(f"Done. Person model saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
