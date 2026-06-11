import cv2
import time
import numpy as np
import tensorflow as tf
import mediapipe as mp
import datetime

# ============================================================
# JewelGuard AI - MediaPipe + Keras Mask Detection
#
# Uses:
# - MediaPipe for stronger face detection
# - Your trained Keras MobileNetV2 model for masked/unmasked
# - Prediction smoothing to reduce flickering
# - Last label memory so label does not disappear instantly
# ============================================================

# -----------------------------
# Load your trained model
# -----------------------------
MASK_MODEL_PATH = "ml/models/mask_detector.keras"
mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)

# IMPORTANT:
# Use same size as training.
# You trained with IMG_SIZE = (160, 160)
MASK_IMG_SIZE = (160, 160)

# If labels ever appear reversed, change this to True
FLIP_LABELS = False

# -----------------------------
# MediaPipe Face Detection
# -----------------------------
mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,              # 0 = close webcam range, 1 = farther range
    min_detection_confidence=0.25
)

# -----------------------------
# Smoothing / memory
# -----------------------------
last_label = "Unknown"
last_confidence = 0.0
last_color = (255, 255, 255)
last_seen_time = 0

MEMORY_SECONDS = 2.0
smooth_predictions = []
SMOOTHING_WINDOW = 8


def classify_mask(face_crop):
    """
    Uses your Keras model.

    Your training labels:
    masked = 0
    unmasked = 1

    Model output:
    close to 0 = masked
    close to 1 = unmasked
    """

    if face_crop is None or face_crop.size == 0:
        return "Unknown", 0.0, (255, 255, 255), 0.5

    resized = cv2.resize(face_crop, MASK_IMG_SIZE)

    # OpenCV uses BGR. TensorFlow model expects RGB.
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    input_img = np.expand_dims(rgb, axis=0)

    prediction = float(mask_model.predict(input_img, verbose=0)[0][0])

    if FLIP_LABELS:
        prediction = 1.0 - prediction

    # Smooth predictions over last few frames
    smooth_predictions.append(prediction)

    if len(smooth_predictions) > SMOOTHING_WINDOW:
        smooth_predictions.pop(0)

    avg_prediction = sum(smooth_predictions) / len(smooth_predictions)

    if avg_prediction >= 0.5:
        label = "Unmasked"
        confidence = avg_prediction
        color = (0, 0, 255)  # red
    else:
        label = "Masked"
        confidence = 1 - avg_prediction
        color = (0, 255, 0)  # green

    return label, confidence, color, avg_prediction


def get_mediapipe_face_boxes(frame):
    """
    Uses MediaPipe to detect face boxes.
    Returns boxes in pixel coordinates:
    x1, y1, x2, y2, score
    """

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    boxes = []

    if not results.detections:
        return boxes

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        # Padding helps include mask/chin/scarf area
        pad_x = int(0.20 * box_w)
        pad_y_top = int(0.20 * box_h)
        pad_y_bottom = int(0.45 * box_h)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y_top)
        x2 = min(w, x + box_w + pad_x)
        y2 = min(h, y + box_h + pad_y_bottom)

        score = float(detection.score[0]) if detection.score else 0.0

        boxes.append((x1, y1, x2, y2, score))

    return boxes

def draw_timestamp(frame):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(
        frame,
        timestamp,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return timestamp


# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open webcam.")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    frame = cv2.resize(frame, (900, 600))
    date_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(
    frame,
    date_text,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2)

    boxes = get_mediapipe_face_boxes(frame)

    if len(boxes) > 0:
        for (x1, y1, x2, y2, face_score) in boxes:
            face_crop = frame[y1:y2, x1:x2]

            label, confidence, color, raw_prediction = classify_mask(face_crop)

            last_label = label
            last_confidence = confidence
            last_color = color
            last_seen_time = time.time()

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Face {face_score:.2f}",
                (x1, min(590, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    else:
        # If MediaPipe loses the face briefly, keep last result on screen
        if time.time() - last_seen_time <= MEMORY_SECONDS:
            cv2.putText(
                frame,
                f"Searching... last: {last_label} {last_confidence:.2f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                last_color,
                2
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.imshow("JewelGuard AI - MediaPipe Mask Detection", frame)

    # Press p to quit
    if cv2.waitKey(1) & 0xFF == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()
face_detector.close()