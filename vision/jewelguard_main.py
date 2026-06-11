import cv2
import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# JewelGuard AI - Main Combined System
#
# Combines:
# - YOLO person detection
# - Haar face detection
# - Keras mask/unmask model
# - Jewelry case ROI
# - Motion intensity in case ROI
# - Loitering near case
# - Risk score + reasons
# ============================================================

# -----------------------------
# Video source
# -----------------------------
# Use 0 for webcam
# Or use video file:
# VIDEO_SOURCE = "data/sample_videos/jewelry_store_test.mp4"
VIDEO_SOURCE = "data/sample_videos/03.06.2026_16.38.33_REC.mp4"
# -----------------------------
# Paths
# -----------------------------
ROI_PATH = Path("data/case_roi.json")
SCREENSHOT_DIR = Path("data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

MASK_MODEL_PATH = "ml/models/mask_detector.keras"

# -----------------------------
# Models
# -----------------------------
person_model = YOLO("yolov8n.pt")
mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise Exception("Could not load Haar face cascade.")

PERSON_CLASS_ID = 0

# -----------------------------
# Settings
# -----------------------------
FRAME_WIDTH = 900
FRAME_HEIGHT = 600

PERSON_CONFIDENCE_THRESHOLD = 0.35
MASK_IMG_SIZE = (160, 160)

LOW_MOTION_THRESHOLD = 800
MEDIUM_MOTION_THRESHOLD = 2000
HIGH_MOTION_THRESHOLD = 6000

MOTION_COOLDOWN_LIMIT = 20
ALERT_COOLDOWN_SECONDS = 5

# -----------------------------
# State variables
# -----------------------------
previous_gray_roi = None
person_near_start_time = None

high_motion_count = 0
motion_cooldown_frames = 0

last_alert_time = 0

last_mask_label = "Unknown"
last_mask_confidence = 0.0
last_face_covering_detected = False
last_mask_seen_time = 0
MASK_MEMORY_SECONDS = 2.0


# ============================================================
# ROI functions
# ============================================================

def save_roi(roi):
    ROI_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ROI_PATH, "w") as f:
        json.dump(roi, f, indent=4)

    print(f"ROI saved to {ROI_PATH}: {roi}")


def load_roi():
    if ROI_PATH.exists():
        with open(ROI_PATH, "r") as f:
            roi = json.load(f)

        print(f"Loaded ROI from {ROI_PATH}: {roi}")
        return roi

    return None


def select_roi_from_first_frame(video_source):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise Exception("Could not open video source for ROI selection.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Could not read first frame for ROI selection.")

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    print("Draw a box around the jewelry case.")
    print("Press ENTER or SPACE to confirm.")
    print("Press C to cancel.")

    x, y, w, h = cv2.selectROI(
        "Select Jewelry Case ROI",
        frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Select Jewelry Case ROI")

    if w == 0 or h == 0:
        raise Exception("No ROI selected.")

    roi = {
        "x1": int(x),
        "y1": int(y),
        "x2": int(x + w),
        "y2": int(y + h)
    }

    save_roi(roi)
    return roi


# ============================================================
# Detection helpers
# ============================================================

def classify_mask(face_crop):
    """
    Classifies clean face crop.
    Model output:
    close to 0 = masked
    close to 1 = unmasked
    """

    if face_crop is None or face_crop.size == 0:
        return "Unknown", 0.0, (255, 255, 255), False

    resized = cv2.resize(face_crop, MASK_IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(rgb, axis=0)

    prediction = mask_model.predict(input_img, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "Unmasked"
        confidence = float(prediction)
        color = (0, 0, 255)
        face_covering_detected = False
    else:
        label = "Masked"
        confidence = float(1 - prediction)
        color = (0, 255, 0)
        face_covering_detected = True

    return label, confidence, color, face_covering_detected


def detect_faces_and_masks(frame):
    """
    Detects faces using Haar and classifies mask/unmask using Keras.
    Returns:
    - face_covering_detected
    - best mask label/confidence
    Draws face boxes directly on frame.
    """
    global last_mask_label
    global last_mask_confidence
    global last_face_covering_detected
    global last_mask_seen_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    current_face_covering_detected = False
    best_label = "Unknown"
    best_confidence = 0.0

    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]

        label, confidence, color, face_covering_detected = classify_mask(face_crop)

        if confidence > best_confidence:
            best_label = label
            best_confidence = confidence
            current_face_covering_detected = face_covering_detected

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # Memory: if face disappears briefly, keep recent mask status
    if len(faces) > 0:
        last_mask_label = best_label
        last_mask_confidence = best_confidence
        last_face_covering_detected = current_face_covering_detected
        last_mask_seen_time = time.time()
    else:
        if time.time() - last_mask_seen_time <= MASK_MEMORY_SECONDS:
            best_label = last_mask_label
            best_confidence = last_mask_confidence
            current_face_covering_detected = last_face_covering_detected
        else:
            best_label = "Unknown"
            best_confidence = 0.0
            current_face_covering_detected = False

    return current_face_covering_detected, best_label, best_confidence


def get_overlap_ratio(person_box, roi_box):
    px1, py1, px2, py2 = person_box
    rx1, ry1, rx2, ry2 = roi_box

    overlap_x = max(0, min(px2, rx2) - max(px1, rx1))
    overlap_y = max(0, min(py2, ry2) - max(py1, ry1))

    overlap_area = overlap_x * overlap_y
    person_area = max(1, (px2 - px1) * (py2 - py1))

    return overlap_area / person_area


def calculate_motion_in_roi(frame, roi):
    global previous_gray_roi

    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

    roi_frame = frame[y1:y2, x1:x2]

    if roi_frame.size == 0:
        return 0

    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)

    if previous_gray_roi is None:
        previous_gray_roi = gray_roi
        return 0

    frame_diff = cv2.absdiff(previous_gray_roi, gray_roi)

    threshold_frame = cv2.threshold(
        frame_diff,
        25,
        255,
        cv2.THRESH_BINARY
    )[1]

    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

    motion_score = cv2.countNonZero(threshold_frame)

    previous_gray_roi = gray_roi

    return motion_score


def get_motion_level(motion_score):
    if motion_score > HIGH_MOTION_THRESHOLD:
        return "HIGH", (0, 0, 255)
    elif motion_score > MEDIUM_MOTION_THRESHOLD:
        return "MEDIUM", (0, 165, 255)
    elif motion_score > LOW_MOTION_THRESHOLD:
        return "LOW", (0, 255, 255)
    else:
        return "NONE", (0, 255, 0)


def calculate_risk(
    person_near_case,
    motion_level,
    high_motion_count,
    loitering_seconds,
    face_covering_detected
):
    risk = 0
    reasons = []

    if person_near_case:
        risk += 10
        reasons.append("Person near jewelry case")

    if motion_level == "LOW":
        risk += 5
        reasons.append("Low motion near case")
    elif motion_level == "MEDIUM":
        risk += 10
        reasons.append("Medium motion near case")
    elif motion_level == "HIGH":
        risk += 25
        reasons.append("High motion near case")

    if high_motion_count >= 3:
        risk += 25
        reasons.append("Repeated high motion near case")

    if loitering_seconds >= 25:
        risk += 15
        reasons.append("Loitering near case")

    if face_covering_detected:
        risk += 20
        reasons.append("Face covering detected")

    risk = min(risk, 100)

    if risk >= 70:
        level = "HIGH"
        color = (0, 0, 255)
    elif risk >= 40:
        level = "MEDIUM"
        color = (0, 165, 255)
    else:
        level = "LOW"
        color = (0, 255, 0)

    return risk, level, color, reasons


def save_alert_screenshot(frame, risk_score):
    global last_alert_time

    current_time = time.time()

    if current_time - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = SCREENSHOT_DIR / f"alert_risk_{risk_score}_{timestamp}.jpg"

    cv2.imwrite(str(filename), frame)
    print(f"Alert screenshot saved: {filename}")

    last_alert_time = current_time


# ============================================================
# Main
# ============================================================

def main():
    global previous_gray_roi
    global person_near_start_time
    global high_motion_count
    global motion_cooldown_frames

    roi = load_roi()

    if roi is None:
        roi = select_roi_from_first_frame(VIDEO_SOURCE)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise Exception("Could not open video source.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or failed to read frame.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        roi_x1 = roi["x1"]
        roi_y1 = roi["y1"]
        roi_x2 = roi["x2"]
        roi_y2 = roi["y2"]
        roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

        # -----------------------------
        # Draw ROI
        # -----------------------------
        cv2.rectangle(
            frame,
            (roi_x1, roi_y1),
            (roi_x2, roi_y2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            frame,
            "Jewelry Case ROI",
            (roi_x1, max(25, roi_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # -----------------------------
        # Face/mask detection
        # -----------------------------
        face_covering_detected, mask_label, mask_confidence = detect_faces_and_masks(frame)

        # -----------------------------
        # Motion in jewelry case ROI
        # -----------------------------
        motion_score = calculate_motion_in_roi(frame, roi)
        motion_level, motion_color = get_motion_level(motion_score)

        if motion_cooldown_frames > 0:
            motion_cooldown_frames -= 1

        if motion_level == "HIGH" and motion_cooldown_frames == 0:
            high_motion_count += 1
            motion_cooldown_frames = MOTION_COOLDOWN_LIMIT

        # -----------------------------
        # YOLO person detection
        # -----------------------------
        results = person_model(frame, verbose=False)
        result = results[0]

        person_near_case = False

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id != PERSON_CLASS_ID:
                    continue

                if confidence < PERSON_CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                person_box = (x1, y1, x2, y2)

                overlap_ratio = get_overlap_ratio(person_box, roi_box)

                if overlap_ratio > 0.02:
                    person_near_case = True
                    box_color = (0, 0, 255)
                    person_status = "Near Case"
                else:
                    box_color = (0, 255, 255)
                    person_status = "Person"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    3
                )

                cv2.putText(
                    frame,
                    f"{person_status} {confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

        # -----------------------------
        # Loitering
        # -----------------------------
        if person_near_case:
            if person_near_start_time is None:
                person_near_start_time = time.time()

            loitering_seconds = int(time.time() - person_near_start_time)
        else:
            person_near_start_time = None
            loitering_seconds = 0
            high_motion_count = 0
            motion_cooldown_frames = 0

        # -----------------------------
        # Risk
        # -----------------------------
        risk_score, risk_level, risk_color, reasons = calculate_risk(
            person_near_case,
            motion_level,
            high_motion_count,
            loitering_seconds,
            face_covering_detected
        )

        # -----------------------------
        # Status panel
        # -----------------------------
        cv2.rectangle(frame, (10, 10), (560, 310), (30, 30, 30), -1)

        cv2.putText(
            frame,
            f"Mask Status: {mask_label} {mask_confidence:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Person Near Case: {'YES' if person_near_case else 'NO'}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Motion Level: {motion_level} ({motion_score})",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            motion_color,
            2
        )

        cv2.putText(
            frame,
            f"Repeated High Motion: {high_motion_count}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Loitering: {loitering_seconds}s",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Risk: {risk_score}/100 {risk_level}",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            risk_color,
            2
        )

        y_reason = 225

        for reason in reasons[:3]:
            cv2.putText(
                frame,
                f"- {reason}",
                (20, y_reason),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            y_reason += 25

        if risk_score >= 70:
            cv2.putText(
                frame,
                "HIGH RISK ALERT",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            save_alert_screenshot(frame, risk_score)

        cv2.imshow("JewelGuard AI - Main Combined System", frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("p"):
            break

        if key == ord("r"):
            previous_gray_roi = None
            person_near_start_time = None
            high_motion_count = 0
            motion_cooldown_frames = 0
            roi = select_roi_from_first_frame(VIDEO_SOURCE)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()