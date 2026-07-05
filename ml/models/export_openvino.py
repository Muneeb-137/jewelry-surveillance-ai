"""Export all OpenVINO models used by VaultVision."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def _export_person(imgsz: int, out_name: str) -> None:
    out_dir = ROOT / out_name
    print(f"Exporting yolov8n @ {imgsz}px -> {out_name} ...")
    with tempfile.TemporaryDirectory(prefix="vaultvision_ov_") as tmp:
        tmp_path = Path(tmp)
        os.chdir(tmp_path)
        YOLO("yolov8n.pt").export(format="openvino", imgsz=imgsz)
        work_dir = tmp_path / "yolov8n_openvino_model"
        if not work_dir.exists():
            raise SystemExit(f"Export failed for {out_name}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(work_dir, out_dir)
    os.chdir(ROOT)
    print(f"  saved {out_dir}")


def main() -> None:
    _export_person(320, "yolov8n_openvino_model")
    _export_person(640, "yolov8n_openvino_model_640")

    print("Exporting yolov8n-pose @ 256 ...")
    with tempfile.TemporaryDirectory(prefix="vaultvision_ov_") as tmp:
        tmp_path = Path(tmp)
        os.chdir(tmp_path)
        YOLO("yolov8n-pose.pt").export(format="openvino", imgsz=256)
        work_dir = tmp_path / "yolov8n-pose_openvino_model"
        out_dir = ROOT / "yolov8n-pose_openvino_model"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(work_dir, out_dir)
    os.chdir(ROOT)

    print("Exporting yolov8n-face @ 640 ...")
    os.chdir(ROOT)
    YOLO("ml/models/yolov8n-face.pt").export(format="openvino", imgsz=640)
    print("All exports complete.")


if __name__ == "__main__":
    main()
