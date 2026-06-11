import cv2
import json
import time
import threading
import numpy as np
import tensorflow as tf
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp
import datetime
from backend.database import insert_incident

# ============================================================
# JewelGuard AI - Vision Engine
# Polygon security zones + entrance speed + case activity risk
# ============================================================

# -----------------------------
# Video source
# -----------------------------
# Use 0 for webcam
# Or use local MP4:
VIDEO_SOURCE = 0
# VIDEO_SOURCE = "data/sample_videos/jewelry_store_test.mp4"

# -----------------------------
# Paths
# -----------------------------
ZONES_PATH = Path("data/security_zones.json")
SCREENSHOT_DIR = Path("data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_FRAME_PATH = Path("backend/latest_frame.jpg")
MASK_MODEL_PATH = "ml/models/mask_detector.keras"

# -----------------------------
# Models
# -----------------------------
person_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")
mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.45
)

# -----------------------------
# Constants
# -----------------------------
PERSON_CLASS_ID = 0
FRAME_WIDTH = 900
FRAME_HEIGHT = 600
MIRROR_WEBCAM = True

PERSON_CONFIDENCE_THRESHOLD = 0.40
POSE_CONFIDENCE_THRESHOLD = 0.35
MASK_IMG_SIZE = (160, 160)

# COCO keypoint indexes from YOLOv8 pose
LEFT_WRIST = 9
RIGHT_WRIST = 10

POSE_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]

# -----------------------------
# Polygon / activity / risk settings
# -----------------------------
LOW_ACTIVITY_THRESHOLD = 1.5       # % changed pixels inside polygon
MEDIUM_ACTIVITY_THRESHOLD = 5.0
HIGH_ACTIVITY_THRESHOLD = 12.0

ACTIVITY_HISTORY_SIZE = 10
FAST_APPROACH_SPEED = 450          # pixels/sec; tune 350-650
WRIST_NEAR_CASE_MARGIN = 35

MOTION_COOLDOWN_LIMIT = 20
ALERT_COOLDOWN_SECONDS = 5

INCIDENT_SAVE_THRESHOLD = 40
INCIDENT_COOLDOWN_LIMIT = 90

# -----------------------------
# Runtime frame state
# -----------------------------
latest_frame_bytes = None
frame_lock = threading.Lock()

engine_running = False

# -----------------------------
# Stable person ID tracking
# -----------------------------
track_id_to_person_id = {}
next_person_number = 1

last_single_person_id = None
frames_without_person = 0
SAME_PERSON_GRACE_FRAMES = 90

person_speed_state = {}

# -----------------------------
# Zone activity state
# -----------------------------
previous_gray_by_zone = {}
activity_history_by_zone = {}

# -----------------------------
# Behavior state
# -----------------------------
person_near_start_time = None
high_motion_count = 0
motion_cooldown_frames = 0

incident_cooldown_frames = 0
last_alert_time = 0

# -----------------------------
# Mask memory state
# -----------------------------
last_mask_label = "Unknown"
last_mask_confidence = 0.0
last_face_covering_detected = False
last_mask_seen_time = 0
MASK_MEMORY_SECONDS = 2.0

latest_status = {
    "running": False,
    "currentPersonId": "None",
    "alertType": "NORMAL",
    "maskStatus": "Unknown",
    "maskConfidence": 0.0,
    "faceCoveringDetected": False,
    "maskedCount": 0,
    "unmaskedCount": 0,
    "totalPeople": 0,
    "peopleNearCase": 0,
    "personNearCase": False,
    "wristNearCase": False,
    "wristInsideCase": False,
    "entranceFastApproach": False,
    "entranceSpeed": 0.0,
    "caseActivityLevel": "NONE",
    "caseActivityScore": 0.0,
    "motionLevel": "NONE",
    "motionScore": 0.0,
    "repeatedHighMotion": 0,
    "loiteringSeconds": 0,
    "riskScore": 0,
    "riskLevel": "LOW",
    "reasons": [],
    "lastAlertImage": None,
}


# ============================================================
# Frame streaming helpers
# ============================================================

def update_latest_frame(frame):
    global latest_frame_bytes

    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return

    with frame_lock:
        latest_frame_bytes = buffer.tobytes()


def get_latest_frame_bytes():
    with frame_lock:
        return latest_frame_bytes


# ============================================================
# Security zone setup: polygon entrance + polygon display cases
# ============================================================

def load_security_zones():
    if ZONES_PATH.exists():
        with open(ZONES_PATH, "r") as f:
            return json.load(f)

    return {
        "entrance_zones": [],
        "case_zones": [],
        "staff_zones": []
    }


def save_security_zones(zones):
    ZONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ZONES_PATH, "w") as f:
        json.dump(zones, f, indent=4)


def select_polygon_zone(frame, window_name):
    """
    Click points around a zone.
    ENTER = save polygon
    R = reset
    ESC = cancel
    """
    points = []
    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()

        for i, point in enumerate(points):
            cv2.circle(display, tuple(point), 5, (0, 255, 255), -1)

            if i > 0:
                cv2.line(display, tuple(points[i - 1]), tuple(points[i]), (0, 255, 255), 2)

        if len(points) > 2:
            cv2.line(display, tuple(points[-1]), tuple(points[0]), (0, 255, 255), 2)

        cv2.putText(
            display,
            "Click polygon points. ENTER=save, R=reset, ESC=cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])
            redraw()

    redraw()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            break

        if key == ord("r"):
            points.clear()
            redraw()

        if key == 27:  # ESC
            points.clear()
            break

    cv2.destroyWindow(window_name)

    if len(points) < 3:
        return None

    return points


def setup_security_zones(video_source):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise Exception("Could not open video source for zone setup.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Could not read frame for zone setup.")

    if video_source == 0 and MIRROR_WEBCAM:
        frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    zones = {
        "entrance_zones": [],
        "case_zones": [],
        "staff_zones": []
    }

    print("Select ENTRANCE polygon zone.")
    print("Click around the entrance zone, then press ENTER.")
    entrance_polygon = select_polygon_zone(frame, "Select Entrance Zone")

    if entrance_polygon is not None:
        zones["entrance_zones"].append({
            "id": "ENTRANCE-001",
            "name": "Entrance Zone",
            "polygon": entrance_polygon
        })

    print("Select JEWELRY CASE polygon zone.")
    print("Click around the display case outline, then press ENTER.")
    case_polygon = select_polygon_zone(frame, "Select Jewelry Case Zone")

    if case_polygon is not None:
        zones["case_zones"].append({
            "id": "CASE-001",
            "name": "Jewelry Case 1",
            "polygon": case_polygon
        })

    save_security_zones(zones)
    return zones


# ============================================================
# Polygon geometry helpers
# ============================================================

def point_inside_polygon(point, polygon):
    contour = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, point, False)
    return result >= 0


def point_near_polygon(point, polygon, margin):
    contour = np.array(polygon, dtype=np.int32)
    distance = cv2.pointPolygonTest(contour, point, True)
    return distance >= -margin


def polygon_to_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    return {
        "x1": int(min(xs)),
        "y1": int(min(ys)),
        "x2": int(max(xs)),
        "y2": int(max(ys))
    }


def create_polygon_mask(frame_shape, polygon):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def draw_polygon_zone(frame, polygon, label, color):
    pts = np.array(polygon, dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    x, y = polygon[0]
    cv2.putText(
        frame,
        label,
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


def get_overlap_ratio(person_box, zone_box):
    px1, py1, px2, py2 = person_box
    rx1, ry1, rx2, ry2 = zone_box

    overlap_x = max(0, min(px2, rx2) - max(px1, rx1))
    overlap_y = max(0, min(py2, ry2) - max(py1, ry1))

    overlap_area = overlap_x * overlap_y
    person_area = max(1, (px2 - px1) * (py2 - py1))

    return overlap_area / person_area


# ============================================================
# Zone activity: smoothed percentage inside polygon
# ============================================================

def calculate_activity_in_polygon(frame, polygon, zone_id):
    """
    Calculates smoothed activity inside a polygon zone.
    This is not general motion. It is zone-specific case/entrance activity.
    """
    global previous_gray_by_zone
    global activity_history_by_zone

    mask = create_polygon_mask(frame.shape, polygon)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    if zone_id not in previous_gray_by_zone:
        previous_gray_by_zone[zone_id] = masked_gray
        return 0.0

    frame_diff = cv2.absdiff(previous_gray_by_zone[zone_id], masked_gray)

    threshold_frame = cv2.threshold(
        frame_diff,
        30,
        255,
        cv2.THRESH_BINARY
    )[1]

    threshold_frame = cv2.bitwise_and(threshold_frame, threshold_frame, mask=mask)
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

    changed_pixels = cv2.countNonZero(threshold_frame)
    total_pixels = cv2.countNonZero(mask)

    activity_percent = (changed_pixels / max(1, total_pixels)) * 100

    previous_gray_by_zone[zone_id] = masked_gray

    if zone_id not in activity_history_by_zone:
        activity_history_by_zone[zone_id] = []

    activity_history_by_zone[zone_id].append(activity_percent)

    if len(activity_history_by_zone[zone_id]) > ACTIVITY_HISTORY_SIZE:
        activity_history_by_zone[zone_id].pop(0)

    smoothed_activity = sum(activity_history_by_zone[zone_id]) / len(activity_history_by_zone[zone_id])

    return round(smoothed_activity, 2)


def get_activity_level(activity_score):
    if activity_score >= HIGH_ACTIVITY_THRESHOLD:
        return "HIGH"
    elif activity_score >= MEDIUM_ACTIVITY_THRESHOLD:
        return "MEDIUM"
    elif activity_score >= LOW_ACTIVITY_THRESHOLD:
        return "LOW"
    else:
        return "NONE"


# ============================================================
# Mask detection
# ============================================================

def classify_mask(face_crop):
    if face_crop is None or face_crop.size == 0:
        return "Unknown", 0.0, (255, 255, 255), False

    resized = cv2.resize(face_crop, MASK_IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_img = np.expand_dims(rgb, axis=0)

    prediction = mask_model.predict(input_img, verbose=0)[0][0]

    if prediction >= 0.5:
        return "Unmasked", float(prediction), (0, 0, 255), False
    else:
        return "Masked", float(1 - prediction), (0, 255, 0), True


def detect_faces_and_masks(frame):
    global last_mask_label
    global last_mask_confidence
    global last_face_covering_detected
    global last_mask_seen_time

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    best_label = "Unknown"
    best_confidence = 0.0
    face_covering_detected = False
    masked_count = 0
    unmasked_count = 0

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            pad_x = int(0.20 * box_w)
            pad_y_top = int(0.20 * box_h)
            pad_y_bottom = int(0.45 * box_h)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y_top)
            x2 = min(w, x + box_w + pad_x)
            y2 = min(h, y + box_h + pad_y_bottom)

            face_crop = frame[y1:y2, x1:x2]

            label, confidence, color, is_masked = classify_mask(face_crop)

            if label == "Masked":
                masked_count += 1
            elif label == "Unmasked":
                unmasked_count += 1

            if confidence > best_confidence:
                best_label = label
                best_confidence = confidence
                face_covering_detected = is_masked

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        last_mask_label = best_label
        last_mask_confidence = best_confidence
        last_face_covering_detected = face_covering_detected
        last_mask_seen_time = time.time()

    else:
        if time.time() - last_mask_seen_time <= MASK_MEMORY_SECONDS:
            best_label = last_mask_label
            best_confidence = last_mask_confidence
            face_covering_detected = last_face_covering_detected
        else:
            best_label = "Unknown"
            best_confidence = 0.0
            face_covering_detected = False

    return face_covering_detected, best_label, best_confidence, masked_count, unmasked_count


# ============================================================
# Person tracking + entrance speed
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def calculate_person_speed(person_id, center):
    global person_speed_state

    now = time.time()

    if person_id not in person_speed_state:
        person_speed_state[person_id] = {
            "center": center,
            "time": now,
            "speed": 0.0
        }
        return 0.0

    old_center = person_speed_state[person_id]["center"]
    old_time = person_speed_state[person_id]["time"]

    dt = max(0.001, now - old_time)

    dx = center[0] - old_center[0]
    dy = center[1] - old_center[1]

    raw_speed = ((dx * dx + dy * dy) ** 0.5) / dt

    old_speed = person_speed_state[person_id]["speed"]
    smoothed_speed = (0.75 * old_speed) + (0.25 * raw_speed)

    person_speed_state[person_id] = {
        "center": center,
        "time": now,
        "speed": smoothed_speed
    }

    return round(smoothed_speed, 2)


def get_person_id_for_track(raw_track_id, detection_count):
    global track_id_to_person_id
    global next_person_number
    global last_single_person_id
    global frames_without_person

    if raw_track_id in track_id_to_person_id:
        return track_id_to_person_id[raw_track_id]

    if detection_count == 1 and last_single_person_id is not None and frames_without_person < SAME_PERSON_GRACE_FRAMES:
        track_id_to_person_id[raw_track_id] = last_single_person_id
        return last_single_person_id

    person_id = f"P-{next_person_number:03d}"
    next_person_number += 1
    track_id_to_person_id[raw_track_id] = person_id

    return person_id


def process_person_tracking(frame, zones):
    """
    YOLO person tracking:
    - stable session person IDs
    - person near polygon case zones
    - fast approach through polygon entrance zones
    """
    global last_single_person_id
    global frames_without_person

    results = person_model.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS_ID],
        imgsz=416,
        conf=PERSON_CONFIDENCE_THRESHOLD,
        iou=0.60,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        frames_without_person += 1
        return {
            "total_people": 0,
            "people_near_case_count": 0,
            "person_near_case": False,
            "active_person_id": "None",
            "entrance_fast_approach": False,
            "entrance_speed": 0.0,
            "detections": []
        }

    boxes = result.boxes
    detection_count = len(boxes)

    entrance_fast_approach = False
    max_entrance_speed = 0.0

    for i, box in enumerate(boxes):
        confidence = float(box.conf[0])

        if confidence < PERSON_CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        person_box = (int(x1), int(y1), int(x2), int(y2))

        if box.id is not None:
            raw_track_id = int(box.id[0])
        else:
            raw_track_id = f"fallback-{i}"

        person_id = get_person_id_for_track(raw_track_id, detection_count)
        center = box_center(person_box)

        speed = calculate_person_speed(person_id, center)
        max_entrance_speed = max(max_entrance_speed, speed)

        in_entrance_zone = False

        for entrance_zone in zones["entrance_zones"]:
            if point_inside_polygon(center, entrance_zone["polygon"]):
                in_entrance_zone = True
                if speed >= FAST_APPROACH_SPEED:
                    entrance_fast_approach = True
                break

        person_near_case = False

        for case_zone in zones["case_zones"]:
            case_bbox = polygon_to_bbox(case_zone["polygon"])
            case_box = (
                case_bbox["x1"],
                case_bbox["y1"],
                case_bbox["x2"],
                case_bbox["y2"]
            )

            overlap_ratio = get_overlap_ratio(person_box, case_box)

            # bbox approximation for proximity to polygon zone
            if overlap_ratio > 0.02:
                person_near_case = True
                break

        detections.append({
            "person_id": person_id,
            "box": person_box,
            "person_near_case": person_near_case,
            "speed": speed,
            "in_entrance_zone": in_entrance_zone,
            "confidence": confidence
        })

    if len(detections) == 0:
        frames_without_person += 1
        return {
            "total_people": 0,
            "people_near_case_count": 0,
            "person_near_case": False,
            "active_person_id": "None",
            "entrance_fast_approach": False,
            "entrance_speed": 0.0,
            "detections": []
        }

    frames_without_person = 0

    if len(detections) == 1:
        last_single_person_id = detections[0]["person_id"]

    total_people = len(detections)
    people_near_case_count = sum(1 for d in detections if d["person_near_case"])
    person_near_case = people_near_case_count > 0

    near_case_people = [d for d in detections if d["person_near_case"]]

    if near_case_people:
        active_person_id = near_case_people[0]["person_id"]
    else:
        active_person_id = detections[0]["person_id"]

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        person_id = detection["person_id"]
        speed = detection["speed"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
        cv2.putText(
            frame,
            f"{person_id} | {speed:.0f}px/s",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    return {
        "total_people": total_people,
        "people_near_case_count": people_near_case_count,
        "person_near_case": person_near_case,
        "active_person_id": active_person_id,
        "entrance_fast_approach": entrance_fast_approach,
        "entrance_speed": max_entrance_speed,
        "detections": detections
    }


# ============================================================
# Pose / wrist detection with polygon case zones
# ============================================================

def draw_pose_skeleton(frame, keypoints):
    for start_idx, end_idx in POSE_CONNECTIONS:
        x1, y1 = keypoints[start_idx]
        x2, y2 = keypoints[end_idx]

        if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
            continue

        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

    for i, (x, y) in enumerate(keypoints):
        if x <= 0 or y <= 0:
            continue

        if i in [LEFT_WRIST, RIGHT_WRIST]:
            color = (0, 0, 255)
            radius = 7
        else:
            color = (0, 255, 255)
            radius = 5

        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def process_pose_detection(frame, zones):
    """
    YOLO pose:
    - draws skeleton
    - detects wrist near polygon case zones
    - detects wrist inside protected polygon case zones
    """
    results = pose_model(
        frame,
        imgsz=416,
        conf=POSE_CONFIDENCE_THRESHOLD,
        iou=0.60,
        max_det=10,
        verbose=False
    )

    result = results[0]

    wrist_near_case = False
    wrist_inside_case = False

    if result.boxes is None or result.keypoints is None:
        return {
            "wrist_near_case": False,
            "wrist_inside_case": False
        }

    keypoints_xy = result.keypoints.xy

    for i in range(len(keypoints_xy)):
        keypoints = keypoints_xy[i].cpu().numpy()
        draw_pose_skeleton(frame, keypoints)

        wrist_points = []

        left_wrist = keypoints[LEFT_WRIST]
        right_wrist = keypoints[RIGHT_WRIST]

        if left_wrist[0] > 0 and left_wrist[1] > 0:
            wrist_points.append((int(left_wrist[0]), int(left_wrist[1])))

        if right_wrist[0] > 0 and right_wrist[1] > 0:
            wrist_points.append((int(right_wrist[0]), int(right_wrist[1])))

        for wrist_point in wrist_points:
            wx, wy = wrist_point

            for case_zone in zones["case_zones"]:
                case_polygon = case_zone["polygon"]

                if point_inside_polygon(wrist_point, case_polygon):
                    wrist_inside_case = True
                    wrist_near_case = True

                    cv2.circle(frame, wrist_point, 9, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "Wrist inside case zone",
                        (wx + 10, wy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                elif point_near_polygon(wrist_point, case_polygon, WRIST_NEAR_CASE_MARGIN):
                    wrist_near_case = True

                    cv2.circle(frame, wrist_point, 8, (0, 165, 255), -1)
                    cv2.putText(
                        frame,
                        "Wrist near case",
                        (wx + 10, wy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 165, 255),
                        2
                    )

    return {
        "wrist_near_case": wrist_near_case,
        "wrist_inside_case": wrist_inside_case
    }


# ============================================================
# Risk model
# ============================================================

def calculate_risk(
    face_covering_detected,
    entrance_fast_approach,
    person_near_case,
    wrist_near_case,
    wrist_inside_case,
    case_activity_level,
    repeated_case_activity,
    loitering_seconds
):
    risk = 0
    reasons = []
    alert_type = "NORMAL"

    # Early prevention: identity / entrance
    if face_covering_detected:
        risk += 40
        reasons.append("Reduced identity visibility / face covering detected")
        alert_type = "IDENTITY_WARNING"

    if entrance_fast_approach:
        risk += 30
        reasons.append("Fast approach detected near entrance")
        if alert_type == "NORMAL":
            alert_type = "ENTRANCE_ACTIVITY"

    if face_covering_detected and entrance_fast_approach:
        risk += 15
        reasons.append("Face covering with fast entrance movement")
        alert_type = "CASE_WATCH"

    # Display case context
    if person_near_case:
        risk += 15
        reasons.append("Person near jewelry display case")

    if wrist_near_case:
        risk += 25
        reasons.append("Wrist/hand near display case boundary")

    if wrist_inside_case:
        risk += 45
        reasons.append("Wrist/hand entered protected case zone")
        alert_type = "CRITICAL_ALERT"

    # Case activity is weak alone, strong with context
    if case_activity_level == "LOW":
        risk += 2
        reasons.append("Low case activity")
    elif case_activity_level == "MEDIUM":
        risk += 5
        reasons.append("Medium case activity")
    elif case_activity_level == "HIGH":
        risk += 8
        reasons.append("High case activity")

    if case_activity_level in ["MEDIUM", "HIGH"] and person_near_case:
        risk += 10
        reasons.append("Case activity while person is near display")
        alert_type = "CASE_WATCH"

    if case_activity_level in ["MEDIUM", "HIGH"] and wrist_near_case:
        risk += 25
        reasons.append("Hand movement near jewelry case")
        alert_type = "CRITICAL_ALERT"

    if repeated_case_activity >= 3 and (person_near_case or wrist_near_case):
        risk += 15
        reasons.append("Repeated case activity near customer")
        alert_type = "CASE_WATCH"

    # Loitering
    if loitering_seconds >= 20:
        risk += 10
        reasons.append("Loitering near jewelry display")

    # Strong combinations
    if face_covering_detected and person_near_case:
        risk += 15
        reasons.append("Face covering while near jewelry case")
        alert_type = "CASE_WATCH"

    if face_covering_detected and wrist_near_case:
        risk += 25
        reasons.append("Face covering with hand near display case")
        alert_type = "CRITICAL_ALERT"

    if face_covering_detected and wrist_inside_case:
        risk += 20
        reasons.append("Face covering with hand inside protected case zone")
        alert_type = "CRITICAL_ALERT"

    risk = min(risk, 100)

    if risk >= 75:
        return risk, "HIGH", "CRITICAL_ALERT", reasons
    elif risk >= 55:
        return risk, "MEDIUM", "CASE_WATCH", reasons
    elif risk >= 40:
        return risk, "MEDIUM", alert_type, reasons
    else:
        return risk, "LOW", "NORMAL", reasons


# ============================================================
# Incident logging
# ============================================================

def save_incident_screenshot(frame, risk_score, person_id):
    screenshot_dir = Path("data/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{person_id}_risk_{risk_score}_{timestamp}.jpg"
    screenshot_path = screenshot_dir / filename

    cv2.imwrite(str(screenshot_path), frame)
    return str(screenshot_path)


def build_risk_description(reasons):
    if not reasons:
        return "No active risk reasons."
    return " | ".join(reasons)


def log_incident_if_needed(
    frame,
    person_id,
    risk_score,
    risk_level,
    reasons,
    mask_label,
    mask_confidence,
    face_covering_detected,
    people_near_case_count,
    wrist_near_case,
    motion_level,
    motion_score,
    high_motion_count,
    loitering_seconds,
):
    global incident_cooldown_frames

    if incident_cooldown_frames > 0:
        incident_cooldown_frames -= 1
        return None

    if risk_score < INCIDENT_SAVE_THRESHOLD:
        return None

    screenshot_path = save_incident_screenshot(frame, risk_score, person_id)
    risk_description = build_risk_description(reasons)

    insert_incident(
        person_id=person_id,
        risk_score=int(risk_score),
        risk_level=risk_level,
        risk_description=risk_description,
        mask_status=mask_label,
        mask_confidence=float(mask_confidence),
        face_covering_detected=face_covering_detected,
        people_near_case=int(people_near_case_count),
        wrist_near_case=wrist_near_case,
        motion_level=motion_level,
        motion_score=float(motion_score),
        repeated_high_motion=int(high_motion_count),
        loitering_seconds=int(loitering_seconds),
        screenshot_path=screenshot_path,
    )

    print(f"Incident saved for {person_id}: {risk_level} risk")

    incident_cooldown_frames = INCIDENT_COOLDOWN_LIMIT
    return screenshot_path


# ============================================================
# Visual overlay
# ============================================================

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


# def draw_status_badge(frame, risk_level, alert_type, risk_score):
#     if risk_level == "HIGH":
#         color = (0, 0, 255)
#     elif risk_level == "MEDIUM":
#         color = (0, 165, 255)
#     else:
#         color = (0, 255, 0)

#     text = f"{alert_type} | Risk {risk_score}"

#     cv2.rectangle(frame, (10, 45), (430, 85), (0, 0, 0), -1)
#     cv2.putText(
#         frame,
#         text,
#         (20, 73),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.75,
#         color,
#         2
#     )


# ============================================================
# Main vision loop
# ============================================================

def run_vision_loop():
    global engine_running
    global latest_status
    global person_near_start_time
    global high_motion_count
    global motion_cooldown_frames
    global incident_cooldown_frames
    global track_id_to_person_id
    global next_person_number
    global last_single_person_id
    global frames_without_person

    zones = load_security_zones()

    if len(zones.get("entrance_zones", [])) == 0 or len(zones.get("case_zones", [])) == 0:
        zones = setup_security_zones(VIDEO_SOURCE)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        latest_status["running"] = False
        latest_status["error"] = f"Could not open video source: {VIDEO_SOURCE}"
        return

    engine_running = True
    latest_status["running"] = True

    while engine_running:
        ret, frame = cap.read()

        # Skip extra frames for video files so playback feels faster
        if VIDEO_SOURCE != 0:
            cap.grab()
            cap.grab()

        if not ret:
            if VIDEO_SOURCE != 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                previous_gray_by_zone.clear()
                activity_history_by_zone.clear()
                continue
            break

        if VIDEO_SOURCE == 0 and MIRROR_WEBCAM:
            frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        draw_timestamp(frame)

        # Draw calibrated polygon zones
        for entrance_zone in zones["entrance_zones"]:
            draw_polygon_zone(
                frame,
                entrance_zone["polygon"],
                entrance_zone["name"],
                (255, 0, 255)
            )

        for case_zone in zones["case_zones"]:
            draw_polygon_zone(
                frame,
                case_zone["polygon"],
                case_zone["name"],
                (255, 0, 0)
            )

        # Face / mask detection
        face_covering_detected, mask_label, mask_confidence, masked_count, unmasked_count = detect_faces_and_masks(frame)

        # Case activity inside polygon zones
        case_activity_scores = []

        for case_zone in zones["case_zones"]:
            score = calculate_activity_in_polygon(
                frame,
                case_zone["polygon"],
                case_zone["id"]
            )
            case_activity_scores.append(score)

        case_activity_score = max(case_activity_scores) if case_activity_scores else 0.0
        case_activity_level = get_activity_level(case_activity_score)

        # Person tracking for IDs, near-case, and entrance speed
        tracking_result = process_person_tracking(frame, zones)

        total_people = tracking_result["total_people"]
        people_near_case_count = tracking_result["people_near_case_count"]
        person_near_case = tracking_result["person_near_case"]
        active_person_id = tracking_result["active_person_id"]
        entrance_fast_approach = tracking_result["entrance_fast_approach"]
        entrance_speed = tracking_result["entrance_speed"]

        # Pose for wrist near/inside polygon case
        pose_status = process_pose_detection(frame, zones)

        wrist_near_case = pose_status["wrist_near_case"]
        wrist_inside_case = pose_status["wrist_inside_case"]

        # Loitering logic
        if person_near_case:
            if person_near_start_time is None:
                person_near_start_time = time.time()
            loitering_seconds = int(time.time() - person_near_start_time)
        else:
            person_near_start_time = None
            loitering_seconds = 0
            high_motion_count = 0
            motion_cooldown_frames = 0

        # Repeated case activity only matters when person/wrist is involved
        if motion_cooldown_frames > 0:
            motion_cooldown_frames -= 1

        if (
            case_activity_level == "HIGH"
            and (person_near_case or wrist_near_case)
            and motion_cooldown_frames == 0
        ):
            high_motion_count += 1
            motion_cooldown_frames = MOTION_COOLDOWN_LIMIT

        # Risk calculation
        risk_score, risk_level, alert_type, reasons = calculate_risk(
            face_covering_detected=face_covering_detected,
            entrance_fast_approach=entrance_fast_approach,
            person_near_case=person_near_case,
            wrist_near_case=wrist_near_case,
            wrist_inside_case=wrist_inside_case,
            case_activity_level=case_activity_level,
            repeated_case_activity=high_motion_count,
            loitering_seconds=loitering_seconds
        )

        # draw_status_badge(frame, risk_level, alert_type, risk_score)

        alert_image = None

        if active_person_id != "None":
            alert_image = log_incident_if_needed(
                frame=frame,
                person_id=active_person_id,
                risk_score=risk_score,
                risk_level=risk_level,
                reasons=reasons,
                mask_label=mask_label,
                mask_confidence=mask_confidence,
                face_covering_detected=face_covering_detected,
                people_near_case_count=people_near_case_count,
                wrist_near_case=wrist_near_case,
                motion_level=case_activity_level,
                motion_score=case_activity_score,
                high_motion_count=high_motion_count,
                loitering_seconds=loitering_seconds,
            )

        latest_status = {
            "running": True,
            "currentPersonId": active_person_id,
            "alertType": alert_type,
            "maskStatus": mask_label,
            "maskConfidence": round(float(mask_confidence), 2),
            "faceCoveringDetected": bool(face_covering_detected),
            "maskedCount": int(masked_count),
            "unmaskedCount": int(unmasked_count),
            "totalPeople": int(total_people),
            "peopleNearCase": int(people_near_case_count),
            "personNearCase": bool(person_near_case),
            "wristNearCase": bool(wrist_near_case),
            "wristInsideCase": bool(wrist_inside_case),
            "entranceFastApproach": bool(entrance_fast_approach),
            "entranceSpeed": float(entrance_speed),
            "caseActivityLevel": case_activity_level,
            "caseActivityScore": float(case_activity_score),

            # old names kept so your frontend does not break
            "motionLevel": case_activity_level,
            "motionScore": float(case_activity_score),

            "repeatedHighMotion": int(high_motion_count),
            "loiteringSeconds": int(loitering_seconds),
            "riskScore": int(risk_score),
            "riskLevel": risk_level,
            "reasons": reasons,
            "lastAlertImage": alert_image,
        }

        update_latest_frame(frame)
        time.sleep(0.01)

    cap.release()
    latest_status["running"] = False


# ============================================================
# Public controls used by FastAPI
# ============================================================

def start_engine():
    global engine_running
    global track_id_to_person_id
    global next_person_number
    global last_single_person_id
    global frames_without_person
    global person_speed_state
    global previous_gray_by_zone
    global activity_history_by_zone

    if engine_running:
        return

    track_id_to_person_id = {}
    next_person_number = 1
    last_single_person_id = None
    frames_without_person = 0

    person_speed_state = {}
    previous_gray_by_zone = {}
    activity_history_by_zone = {}

    thread = threading.Thread(target=run_vision_loop, daemon=True)
    thread.start()


def stop_engine():
    global engine_running
    engine_running = False


def get_latest_status():
    return latest_status
