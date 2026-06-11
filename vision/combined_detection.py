import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# -----------------------------
# Load models
# -----------------------------
person_model = YOLO("yolov8n.pt")

MASK_MODEL_PATH = "ml/models/mask_detector.keras"
mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)

PERSON_CLASS_ID = 0

# IMPORTANT:
# Use the same size you used during training.
# If your train_mask_model.py used IMG_SIZE = (160, 160), keep this 160.
# If it used IMG_SIZE = (224, 224), change this to 224.
MASK_IMG_SIZE = (160, 160)


# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open webcam.")


def classify_face_covering(crop):
    """
    Receives a head/upper-body crop from the person box.
    Returns label, confidence, and color.
    """

    if crop is None or crop.size == 0:
        return "Unknown", 0.0, (255, 255, 255)

    # Resize to match training input
    resized = cv2.resize(crop, MASK_IMG_SIZE)

    # Convert BGR from OpenCV to RGB for TensorFlow
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Add batch dimension: (160,160,3) -> (1,160,160,3)
    input_img = np.expand_dims(rgb, axis=0)

    # Model output:
    # close to 0 = masked
    # close to 1 = unmasked
    prediction = mask_model.predict(input_img, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "Unmasked"
        confidence = float(prediction)
        color = (0, 0, 255)  # red
    else:
        label = "Masked"
        confidence = float(1 - prediction)
        color = (0, 255, 0)  # green

    return label, confidence, color


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    frame = cv2.resize(frame, (900, 600))

    # -----------------------------
    # YOLO person detection
    # -----------------------------
    results = person_model(frame, verbose=False)
    result = results[0]

    for box in result.boxes:
        class_id = int(box.cls[0])
        person_confidence = float(box.conf[0])

        if class_id == PERSON_CLASS_ID and person_confidence >= 0.50:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Keep coordinates inside frame boundaries
            h_frame, w_frame, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)

            person_width = x2 - x1
            person_height = y2 - y1

            if person_width <= 0 or person_height <= 0:
                continue

            # -----------------------------
            # Crop head/face-covering region
            # -----------------------------
            # This takes the top 45% of the person box.
            # Good for scarf, bandana, face covering, balaclava, etc.
            head_y1 = y1
            head_y2 = y1 + int(person_height * 0.45)

            # Add a little side padding so scarf/hood is included
            pad_x = int(person_width * 0.10)
            head_x1 = max(0, x1 - pad_x)
            head_x2 = min(w_frame, x2 + pad_x)

            head_crop = frame[head_y1:head_y2, head_x1:head_x2]

            mask_label, mask_confidence, mask_color = classify_face_covering(head_crop)


            # -----------------------------
            # Draw full person box
            # -----------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

            person_text = f"Person {person_confidence:.2f}"
            cv2.putText(
                frame,
                person_text,
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # -----------------------------
            # Draw head crop box
            # -----------------------------
            cv2.rectangle(
                frame,
                (head_x1, head_y1),
                (head_x2, head_y2),
                mask_color,
                3
            )

            mask_text = f"{mask_label} {mask_confidence:.2f}"
            cv2.putText(
                frame,
                mask_text,
                (head_x1, head_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                mask_color,
                2
            )

    cv2.imshow("JewelGuard AI - Person + Face Covering Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()