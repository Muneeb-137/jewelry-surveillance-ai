"""
Re-exports the YOLOv8n-face model to OpenVINO at imgsz=640.
640 is the native YOLO training resolution — best accuracy + detects faces
~60% further away than the original 320 export.
Run from the project root:  python ml/models/reexport_face_416.py
"""
from ultralytics import YOLO
import os

MODEL_PT   = "ml/models/yolov8n-face.pt"
OV_OUT_DIR = "ml/models/yolov8n-face_openvino_model"

print(f"Exporting {MODEL_PT} → OpenVINO @ imgsz=640 ...")
model = YOLO(MODEL_PT)
model.export(format="openvino", imgsz=640)

# ultralytics writes to  ml/models/yolov8n-face_openvino_model/
# (same folder, metadata.yaml will now say 416)
print(f"Done.  New metadata:")
with open(os.path.join(OV_OUT_DIR, "metadata.yaml")) as f:
    print(f.read())
