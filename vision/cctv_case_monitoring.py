import cv2
import json
import time
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# JewelGuard AI - CCTV Jewelry Case Monitoring
#
# Features:
# - Loads CCTV/video file or webcam
# - Lets user draw/select jewelry case ROI
# - Saves and reloads ROI
# - Detects person near jewelry case
# - Detects motion inside jewelry case ROI
# - Converts motion into NONE / LOW / MEDIUM / HIGH
# - Counts repeated high-motion events
# - Tracks loitering near case
# - Calculates explainable risk score
# - Saves alert screenshots
# ============================================================

# -----------------------------
# 1. Video source
# -----------------------------
# Use 0 for webcam.
# Use a video path for CCTV footage.
VIDEO_SOURCE = "data/sample_videos/jewelry_store_test.mp4"

# -----------------------------
# 2. Paths
# -----------------------------
ROI_PATH = Path("data/case_roi.json")
SCREENSHOT_DIR = Path("data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 3. Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0

# -----------------------------
# 4. Frame settings
# -----------------------------
FRAME_WIDTH = 900
FRAME_HEIGHT = 600

# -----------------------------
# 5. Detection settings
# -----------------------------
PERSON_CONFIDENCE_THRESHOLD = 0.35

# Motion thresholds.
# You may tune these depending on your video/camera angle.
LOW_MOTION_THRESHOLD = 800
MEDIUM_MOTION_THRESHOLD = 2000
HIGH_MOTION_THRESHOLD = 6000

# Repeated high-motion logic.
high_motion_count = 0
motion_cooldown_frames = 0
MOTION_COOLDOWN_LIMIT = 20

# Alert screenshot cooldown.
ALERT_COOLDOWN_SECONDS = 5
last_alert_time = 0

# Previous ROI frame for motion detection.
previous_gray_roi = None

# Loitering timer.
person_near_start_time = None


def save_roi(roi):
    """
    Saves ROI coordinates to JSON.
    """
    ROI_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ROI_PATH, "w") as f:
        json.dump(roi, f, indent=4)

    print(f"ROI saved to {ROI_PATH}")
    print(roi)


def load_roi():
    """
    Loads ROI coordinates if they already exist.
    """
    if ROI_PATH.exists():
        with open(ROI_PATH, "r") as f:
            roi = json.load(f)

        print(f"Loaded ROI from {ROI_PATH}: {roi}")
        return roi

    return None


def select_roi_from_first_frame(video_source):
    """
    Opens the video/camera, grabs first frame, lets user draw ROI.
    Press ENTER or SPACE after selecting.
    Press C to cancel.
    """
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

    selected = cv2.selectROI(
        "Select Jewelry Case ROI",
        frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("Select Jewelry Case ROI")

    x, y, w, h = selected

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


def get_overlap_ratio(person_box, roi_box):
    """
    Calculates how much of the person box overlaps the jewelry case ROI.

    person_box format:
    (x1, y1, x2, y2)

    roi_box format:
    (x1, y1, x2, y2)
    """
    px1, py1, px2, py2 = person_box
    rx1, ry1, rx2, ry2 = roi_box

    overlap_x = max(0, min(px2, rx2) - max(px1, rx1))
    overlap_y = max(0, min(py2, ry2) - max(py1, ry1))

    overlap_area = overlap_x * overlap_y
    person_area = max(1, (px2 - px1) * (py2 - py1))

    return overlap_area / person_area


def calculate_motion_in_roi(frame, roi):
    """
    Detects motion inside the jewelry case ROI using frame differencing.
    Returns the raw motion score.
    """
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
    """
    Converts raw motion score into readable motion intensity.
    This prevents small normal browsing movements from becoming high risk.
    """
    if motion_score > HIGH_MOTION_THRESHOLD:
        return "HIGH", (0, 0, 255)       # red
    elif motion_score > MEDIUM_MOTION_THRESHOLD:
        return "MEDIUM", (0, 165, 255)   # orange
    elif motion_score > LOW_MOTION_THRESHOLD:
        return "LOW", (0, 255, 255)      # yellow
    else:
        return "NONE", (0, 255, 0)       # green


def calculate_risk(
    person_near_case,
    motion_level,
    high_motion_count,
    loitering_seconds,
    face_covering_detected=False
):
    """
    Calculates risk using multiple signals.

    This is better than:
        motion near case = high risk

    because normal customers can browse near the case.
    High risk should come from combined signals.
    """
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
        risk_level = "HIGH"
        color = (0, 0, 255)
    elif risk >= 40:
        risk_level = "MEDIUM"
        color = (0, 165, 255)
    else:
        risk_level = "LOW"
        color = (0, 255, 0)

    return risk, risk_level, color, reasons


def save_alert_screenshot(frame, risk_score):
    """
    Saves screenshot when high risk is detected.
    Uses cooldown so it does not save every frame.
    """
    global last_alert_time

    current_time = time.time()

    if current_time - last_alert_time < ALERT_COOLDOWN_SECONDS:
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = SCREENSHOT_DIR / f"alert_risk_{risk_score}_{timestamp}.jpg"

    cv2.imwrite(str(filename), frame)
    print(f"Alert screenshot saved: {filename}")

    last_alert_time = current_time


def main():
    global previous_gray_roi
    global high_motion_count
    global motion_cooldown_frames
    global person_near_start_time

    # -----------------------------
    # Load existing ROI or select new one
    # -----------------------------
    roi = load_roi()

    if roi is None:
        roi = select_roi_from_first_frame(VIDEO_SOURCE)

    # -----------------------------
    # Open CCTV/video feed
    # -----------------------------
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise Exception("Could not open video source.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or failed to read frame.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # -----------------------------
        # Draw jewelry case ROI
        # -----------------------------
        roi_x1 = roi["x1"]
        roi_y1 = roi["y1"]
        roi_x2 = roi["x2"]
        roi_y2 = roi["y2"]

        roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)

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
        # Motion detection inside ROI
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
        results = model(frame, verbose=False)
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

                # Person is near case if part of person overlaps ROI.
                # Tune 0.02 if needed.
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

                label = f"{person_status} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Overlap: {overlap_ratio:.2f}",
                    (x1, min(FRAME_HEIGHT - 10, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

        # -----------------------------
        # Loitering timer
        # -----------------------------
        if person_near_case:
            if person_near_start_time is None:
                person_near_start_time = time.time()

            loitering_seconds = int(time.time() - person_near_start_time)
        else:
            person_near_start_time = None
            loitering_seconds = 0

            # Reset repeated high motion when person leaves case area.
            high_motion_count = 0
            motion_cooldown_frames = 0

        # -----------------------------
        # Risk scoring
        # -----------------------------
        # Keep False for now.
        # Later we can connect this to your mask detection model.
        face_covering_detected = False

        risk_score, risk_level, risk_color, reasons = calculate_risk(
            person_near_case,
            motion_level,
            high_motion_count,
            loitering_seconds,
            face_covering_detected
        )

        # -----------------------------
        # Draw status panel
        # -----------------------------
        cv2.rectangle(frame, (10, 10), (540, 285), (30, 30, 30), -1)

        near_case_text = "YES" if person_near_case else "NO"

        cv2.putText(
            frame,
            f"Person Near Case: {near_case_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Motion Level: {motion_level} ({motion_score})",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            motion_color,
            2
        )

        cv2.putText(
            frame,
            f"Repeated High Motion: {high_motion_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Loitering: {loitering_seconds}s",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Face Covering: {'YES' if face_covering_detected else 'NO'}",
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
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            save_alert_screenshot(frame, risk_score)

        cv2.imshow("JewelGuard AI - CCTV Case Monitoring", frame)

        key = cv2.waitKey(30) & 0xFF

        # Press p to quit.
        if key == ord("p"):
            break

        # Press r to reset/reselect ROI.
        if key == ord("r"):
            previous_gray_roi = None
            high_motion_count = 0
            motion_cooldown_frames = 0
            person_near_start_time = None
            roi = select_roi_from_first_frame(VIDEO_SOURCE)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()