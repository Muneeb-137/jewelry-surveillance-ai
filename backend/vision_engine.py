import cv2
import json
import time
import queue
import threading
import numpy as np
import tensorflow as tf
from pathlib import Path
from ultralytics import YOLO
import datetime
from backend.database import insert_incident

# ============================================================
# VaultVision - Vision Engine
# Mask flagging + wrist-at-case pose alerts + case ROI interaction
# ============================================================

from backend.source_config import (
    ACTIVE,
    IS_WEBCAM,
    IS_RTSP,
    IS_VIDEO_FAR,
    MODE,
    VIDEO_PROFILE,
    WEBCAM,
    VIDEO_SOURCE,
    describe as describe_source,
)
from backend.notifications import notify_incident_created
from backend.retail_config import (
    RUNTIME_IS_DEMO,
    RUNTIME_IS_LIVE,
    RUNTIME_LABEL,
    RUNTIME_PROFILE,
    SCENE_CUT_CORREL_THRESHOLD,
    SCENE_CUT_MAD_STRONG,
    SCENE_CUT_MAD_THRESHOLD,
    SCENE_CUT_MIN_INTERVAL_SEC,
    STORE_NAME,
    VIDEO_PLAYBACK_SPEED,
    VIDEO_SCENE_CUT_RESET,
)

# -----------------------------
# Paths
# -----------------------------
CASE_ROI_PATH = Path("data/case_roi.json")
LEGACY_ZONES_PATH = Path("data/security_zones.json")
SCREENSHOT_DIR = Path("data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_FRAME_PATH = Path("backend/latest_frame.jpg")
MASK_MODEL_PATH = "ml/models/mask_detector.keras"
FACE_MODEL_PATH        = "ml/models/yolov8n-face.pt"
FACE_OV_MODEL_PATH     = "ml/models/yolov8n-face_openvino_model/"
PERSON_OV_MODEL_320    = Path("yolov8n_openvino_model")
PERSON_OV_MODEL_640    = Path("yolov8n_openvino_model_640")


def _resolve_person_model_spec():
    """Video/RTSP use 640 IR when exported; webcam stays on 320 for speed."""
    use_640 = MODE in ("video", "rtsp") and PERSON_OV_MODEL_640.exists()
    if use_640:
        return str(PERSON_OV_MODEL_640) + "/", 640
    return str(PERSON_OV_MODEL_320) + "/", 320


_PERSON_MODEL_PATH, _PERSON_MODEL_IMGSZ = _resolve_person_model_spec()

# -----------------------------
# Models — all on OpenVINO IR for Intel CPU/iGPU performance.
# Face OpenVINO model also outputs 5-point landmarks (eyes, nose, mouth)
# which re-enables the frontal-face orientation gate at no extra cost.
# -----------------------------
person_model = YOLO(_PERSON_MODEL_PATH, task="detect")
pose_model   = YOLO("yolov8n-pose_openvino_model/", task="pose")
face_model   = YOLO(FACE_OV_MODEL_PATH,             task="pose")   # pose = detects + 5 kp
mask_model   = tf.keras.models.load_model(MASK_MODEL_PATH)

# -----------------------------
# Constants
# -----------------------------
PERSON_CLASS_ID = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
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
# Pose / wrist / risk settings
# -----------------------------
WRIST_NEAR_CASE_MARGIN = 35
POSE_RUN_INTERVAL = 2  # run pose every N frames while someone is near the case

# Motion inside case ROI (interaction / touch proxy when pose is occluded)
CASE_MOTION_LOW = 800
CASE_MOTION_MEDIUM = 2000
CASE_MOTION_HIGH = 6000
CASE_MOTION_RUN_INTERVAL = 1

INCIDENT_SAVE_THRESHOLD = 40
INCIDENT_COOLDOWN_LIMIT = 90
FLAG_INCIDENT_COOLDOWN = 45
WRIST_INCIDENT_COOLDOWN = 30
MASK_PERSON_INCIDENT_RISK = 40
MASK_PERSON_INCIDENT_LEVEL = "MEDIUM"
CROWD_FLAGGED_INCIDENT_RISK_WATCH = 85
CROWD_FLAGGED_INCIDENT_RISK_HIGH = 90
CROWD_FLAGGED_INCIDENT_LEVEL = "HIGH"

# Person box colors (BGR) — avoid green/red (used on face mask labels)
PERSON_BOX_NORMAL = (255, 255, 0)     # cyan — neutral tracked person
PERSON_BOX_FLAGGED = (0, 165, 255)    # orange
PERSON_BOX_WRIST_NEAR = (0, 140, 255) # orange-red
PERSON_BOX_WRIST_INSIDE = (0, 0, 255) # red

# -----------------------------
# Runtime frame state
# -----------------------------
latest_entrance_frame_bytes = None
latest_store_frame_bytes = None
frame_lock = threading.Lock()

engine_running = False

# Internal tracking keys (not shown on video — mask pipeline only).
# -----------------------------
byte_track_to_person_id = {}
person_last_boxes = {}
person_last_seen_at = {}
person_first_seen_at = {}
person_last_centers = {}
person_last_velocity = {}
track_seen_outside_entrance = {}
next_person_number = 1

# Person-box tag (distinct from face-box "Masked"/"Unmasked").
# FLAGGED only while confirmed mask is above FLAG_MIN_CONF — clears on unmask vote.
PERSON_FLAG_LABEL = "FLAGGED"

last_single_person_id = None
frames_without_person = 0
SAME_PERSON_GRACE_FRAMES = 90
PERSON_REASSOC_IOU = 0.12       # min IoU to treat a detection as same person
PERSON_REASSOC_CENTER = 1.2     # max center shift (× box size) when IoU is low
PERSON_REASSOC_MAX_AGE = 2.0    # seconds — only match recently seen people (live)
VIDEO_REASSOC_MAX_AGE = 15.0    # longer hold for file playback / slow demo
VIDEO_REASSOC_CENTER = 3.5      # tolerate movement while YOLO box flickers
VIDEO_REASSOC_IOU = 0.06
TRACK_ID_VERIFY_MIN_SCORE = 0.05  # reject reused ByteTrack ids on wrong person
STICKY_REASSOC_MAX_AGE = 12.0   # longer window to reattach a flagged P-ID after tracker flicker
PERSON_DEDUPE_IOU = 0.35        # merge overlapping YOLO person boxes (same individual)
PERSON_DEDUPE_CENTER = 0.42     # center proximity merge when boxes partially overlap
PERSON_DEDUPE_MIN_IOU = 0.20    # require some overlap before center-based merge

# -----------------------------
# Behavior state
# -----------------------------
person_near_start_time = None
previous_gray_case_roi = None

incident_cooldown_frames = 0
flag_incident_cooldown_frames = 0
wrist_incident_cooldown_frames = 0
_flag_incident_logged_pids = set()
_crowd_incident_logged = False
_sticky_flagged_pids = set()
_sticky_flag_snapshot = {}
_dismissed_pids = set()
_flag_hold_until = {}
_sticky_unmask_votes = {}
_mask_vote_masked = {}
_mask_vote_unmasked = {}

_staff_pids = set()
_staff_mode_active = False
_crowd_active_since = None

# Customer crowd thresholds (registered persons excluding staff / dismissed)
CROWD_ELEVATED = 2
CROWD_WATCH = 3
CROWD_HIGH = 4
CROWD_SUSTAIN_SEC = 2.0

# -----------------------------
# Mask memory state
# -----------------------------
last_mask_label = "Unknown"
last_mask_confidence = 0.0
last_face_covering_detected = False
last_mask_seen_time = 0
MASK_MEMORY_SECONDS = 0.35

# -----------------------------
# Face detection — two background threads
#
# Thread A (_face_yolo_worker): runs YOLO face detection on frames.
#   Frees ~35-50 ms from the main loop every frame.
#
# Thread B (_face_detection_worker): runs Keras mask classification on crops.
#   Already existed; crops now come from Thread A results.
# -----------------------------

# Mask pipeline settings → edit backend/source_config.py (WEBCAM vs VIDEO)
_CFG = ACTIVE

FACE_DET_CONF          = _CFG["FACE_DET_CONF"]
FACE_DET_IMGSZ         = _CFG["FACE_DET_IMGSZ"]
MIN_FACE_BOX_SIZE      = _CFG["MIN_FACE_BOX_SIZE"]
FACE_CLASSIFY_MIN_CONF = _CFG["FACE_CLASSIFY_MIN_CONF"]
MASKED_THRESHOLD       = _CFG["MASKED_THRESHOLD"]
UNMASKED_THRESHOLD     = _CFG["UNMASKED_THRESHOLD"]
FACE_DET_MAX_DET       = _CFG["FACE_DET_MAX_DET"]
PERSON_TRACK_INTERVAL  = _CFG["PERSON_TRACK_INTERVAL"]
PERSON_CONFIDENCE_THRESHOLD = _CFG.get(
    "PERSON_CONF", 0.40 if IS_WEBCAM else 0.28
)
# OpenVINO person IR is fixed-size — must match the exported model (320 or 640).
PERSON_TRACK_IMGSZ     = _PERSON_MODEL_IMGSZ
PERSON_TRACK_MAX_DET   = _CFG.get("PERSON_MAX_DET", 15 if IS_WEBCAM else 25)
ENTRANCE_GATED_IDS     = _CFG.get("ENTRANCE_GATED_IDS", False)
FLAG_MIN_CONF          = _CFG.get("FLAG_MIN_CONF", 0.75)
FLAG_MIN_PERSON_HEIGHT_RATIO = _CFG.get("FLAG_MIN_PERSON_HEIGHT_RATIO", 0.0)
FLAG_REQUIRE_FACE_LINK = _CFG.get("FLAG_REQUIRE_FACE_LINK", False)
FLAG_SUSTAIN_SEC       = _CFG.get("FLAG_SUSTAIN_SEC", 0.0)
MIN_TRACK_SEC_BEFORE_FLAG = _CFG.get("MIN_TRACK_SEC_BEFORE_FLAG", 0.0)
MIN_TRACK_SEC_BEFORE_INCIDENT = _CFG.get("MIN_TRACK_SEC_BEFORE_INCIDENT", 0.0)
MASK_RESULT_TTL        = _CFG.get("MASK_RESULT_TTL", 5.0)
FLAG_STICKY_MODE       = _CFG.get("FLAG_STICKY_MODE", "off")
FLAG_HOLD_SEC          = _CFG.get("FLAG_HOLD_SEC", 2.0)
FLAG_STICKY_MIN_CONF   = _CFG.get("FLAG_STICKY_MIN_CONF", 0.90)
STICKY_UNMASK_CLEAR    = _CFG.get("STICKY_UNMASK_CLEAR", 4)

_flag_qualify_since = {}


def _is_dismissed(pid):
    """Staff marked this P-ID as a false-positive mask flag for the session."""
    return bool(pid and pid in _dismissed_pids)


def _is_staff(pid):
    """Working staff — excluded from mask pipeline, flags, and customer crowd count."""
    if not pid:
        return False
    if _staff_mode_active:
        return True
    return pid in _staff_pids


def _skip_mask_pipeline(pid):
    return _is_dismissed(pid) or _is_staff(pid)


def person_flag_status(person_results, pid):
    """Return whether person box should show FLAGGED (per person — independent of others)."""
    if _skip_mask_pipeline(pid):
        return False, "Unmasked", 0.0

    result = person_results.get(pid)
    now = time.monotonic()
    if not result:
        _flag_qualify_since.pop(pid, None)
        return False, "Unknown", 0.0
    _is_masked, label, conf = result
    qualifies = label == "Masked" and conf >= FLAG_MIN_CONF

    # Webcam: instant orange box once mask is confirmed at flag confidence.
    if IS_WEBCAM:
        if qualifies:
            if pid not in _flag_qualify_since:
                _flag_qualify_since[pid] = now
        else:
            _flag_qualify_since.pop(pid, None)
        return qualifies, label, conf

    track_age = now - person_first_seen_at.get(pid, now)
    if track_age < MIN_TRACK_SEC_BEFORE_FLAG:
        _flag_qualify_since.pop(pid, None)
        return False, label, conf

    if qualifies:
        if pid not in _flag_qualify_since:
            _flag_qualify_since[pid] = now
        sustained = now - _flag_qualify_since[pid]
        flagged = sustained >= FLAG_SUSTAIN_SEC
    else:
        _flag_qualify_since.pop(pid, None)
        flagged = False

    return flagged, label, conf


def _display_mask_result(pid, raw):
    if _is_staff(pid):
        return None
    if _is_dismissed(pid):
        return (False, "Unmasked", 1.0)
    return raw


def build_display_person_results(active_pids, person_results):
    return {
        pid: _display_mask_result(pid, person_results.get(pid))
        for pid in active_pids
    }


def person_can_be_flagged(pid, person_results, face_by_person, person_box, frame_height):
    if _is_staff(pid):
        return False, "Staff", 0.0
    if _is_dismissed(pid):
        return False, "Unmasked", 0.0

    now = time.monotonic()

    if FLAG_STICKY_MODE != "off" and pid in _sticky_flagged_pids:
        label, conf = _sticky_flag_snapshot.get(pid, ("Masked", FLAG_MIN_CONF))
        return True, label, conf

    hold_until = _flag_hold_until.get(pid)
    if hold_until and now < hold_until:
        label, conf = _sticky_flag_snapshot.get(pid, ("Masked", FLAG_MIN_CONF))
        return True, label, conf

    flagged, label, conf = person_flag_status(person_results, pid)
    if not flagged:
        return False, label, conf
    if FLAG_REQUIRE_FACE_LINK and pid not in face_by_person:
        return False, label, conf
    if FLAG_MIN_PERSON_HEIGHT_RATIO > 0 and frame_height > 0:
        py1, py2 = person_box[1], person_box[3]
        if (py2 - py1) / frame_height < FLAG_MIN_PERSON_HEIGHT_RATIO:
            return False, label, conf

    _sticky_flag_snapshot[pid] = (label, conf)
    _flag_hold_until[pid] = now + FLAG_HOLD_SEC

    if FLAG_STICKY_MODE == "high_conf" and conf >= FLAG_STICKY_MIN_CONF:
        _lock_sticky_flag(pid, label, conf)
    elif FLAG_STICKY_MODE == "session":
        _lock_sticky_flag(pid, label, conf)

    return True, label, conf

if IS_WEBCAM:
    WEBCAM_MASK_CONFIRM      = WEBCAM["MASK_CONFIRM"]
    WEBCAM_MASK_CONFIRM_TILT = WEBCAM["MASK_CONFIRM_TILT"]
    WEBCAM_UNMASK_CLEAR      = WEBCAM["UNMASK_CLEAR"]
    WEBCAM_UNMASK_CLEAR_FROM_MASKED = WEBCAM.get("UNMASK_CLEAR_FROM_MASKED", 2)
    WEBCAM_TARGET_FPS        = WEBCAM["TARGET_FPS"]
    WEBCAM_BLANK_RECONNECT   = WEBCAM["BLANK_RECONNECT_AFTER"]
    FACE_REQUEUE_INTERVAL    = 0.20
else:
    VIDEO_MASK_CONFIRM       = ACTIVE["MASK_CONFIRM"]
    VIDEO_UNMASK_CLEAR       = ACTIVE["UNMASK_CLEAR"]
    FACE_REQUEUE_INTERVAL    = ACTIVE["FACE_REQUEUE_INTERVAL"]
    VIDEO_TARGET_FPS         = ACTIVE.get("TARGET_FPS", 25)
    VIDEO_STREAM_FPS         = ACTIVE.get("STREAM_FPS", 15)
    _stream_jpeg_min_interval = 1.0 / max(VIDEO_STREAM_FPS, 1)

USE_FRONTAL_GATE = False

# Thread A: raw frames in → face box list out
_face_yolo_input_queue = queue.Queue(maxsize=1)
_face_yolo_result      = []          # list of (x1,y1,x2,y2,conf,is_frontal)
_face_yolo_lock        = threading.Lock()

# Thread B (video only): throttled crop queue
_face_input_queue        = queue.Queue(maxsize=ACTIVE["INPUT_QUEUE_SIZE"])
_face_results_per_person = {}        # { face_key: (is_masked, label, conf) }
_face_result_last_seen   = {}        # { face_key: float } — last time person was in active tracking
_face_result_lock        = threading.Lock()
_face_frame_counter      = 0
_face_key_last_classified = {}       # { face_key: timestamp } — throttle re-queuing (video)

# Webcam-only: latest crop worker (fast, non-blocking) + tilt-aware voting.
_webcam_latest_crop = (None, None, True)  # crop, person_id, is_frontal
_webcam_crop_lock     = threading.Lock()
_webcam_mask_event    = threading.Event()


latest_status = {
    "running": False,
    "mode": MODE,
    "videoProfile": VIDEO_PROFILE if MODE == "video" else None,
    "currentPersonId": "None",
    "currentSubjectTag": "None",
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
    "caseMotionLevel": "NONE",
    "caseMotionScore": 0,
    "caseInteraction": False,
    "flaggedCount": 0,
    "alarmLevel": "NONE",
    "alarmActive": False,

    "loiteringSeconds": 0,
    "riskScore": 0,
    "riskLevel": "LOW",
    "reasons": [],
    "entranceReasons": [],
    "storeReasons": [],
    "lastAlertImage": None,
    "flaggedPeople": [],
    "dismissedIds": [],
    "staffIds": [],
    "staffModeActive": False,
    "customerCount": 0,
    "staffCount": 0,
    "crowdLevel": "NONE",
    "crowdActive": False,
    "crowdSeconds": 0,
    "flaggedCustomerCount": 0,
    "trackedPersonIds": [],
    "playbackSpeed": 1.0,
    "sceneCutResetEnabled": False,
    "sceneCutCount": 0,
    "runtimeProfile": RUNTIME_PROFILE,
    "runtimeLabel": RUNTIME_LABEL,
    "flagPolicy": {
        "stickyMode": FLAG_STICKY_MODE,
        "holdSeconds": FLAG_HOLD_SEC,
        "minConf": FLAG_MIN_CONF,
        "incidentDelaySec": MIN_TRACK_SEC_BEFORE_INCIDENT,
    },
}


# ============================================================
# Frame streaming helpers
# ============================================================

_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 45]
_PLACEHOLDER_JPEG = None
_last_stream_jpeg_at = 0.0
_stream_jpeg_min_interval = 1.0 / 15.0


def _placeholder_frame_bytes(message="Click Start on dashboard"):
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is None:
        img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            img,
            message,
            (80, FRAME_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (200, 200, 200),
            2,
        )
        ok, buffer = cv2.imencode(".jpg", img, _JPEG_PARAMS)
        if ok:
            _PLACEHOLDER_JPEG = buffer.tobytes()
    return _PLACEHOLDER_JPEG


def update_latest_frames(entrance_frame, store_frame):
    global latest_entrance_frame_bytes
    global latest_store_frame_bytes
    global _last_stream_jpeg_at

    now = time.time()
    if now - _last_stream_jpeg_at < _stream_jpeg_min_interval:
        return
    _last_stream_jpeg_at = now

    entrance_success, entrance_buffer = cv2.imencode(".jpg", entrance_frame, _JPEG_PARAMS)
    store_success, store_buffer = cv2.imencode(".jpg", store_frame, _JPEG_PARAMS)

    if not entrance_success or not store_success:
        return

    with frame_lock:
        latest_entrance_frame_bytes = entrance_buffer.tobytes()
        latest_store_frame_bytes = store_buffer.tobytes()


def get_latest_frame_bytes(view="store"):
    with frame_lock:
        if view == "entrance":
            return latest_entrance_frame_bytes or _placeholder_frame_bytes()

        return latest_store_frame_bytes or _placeholder_frame_bytes()


# ============================================================
# Jewelry case ROI — rectangle selection (replaces polygon zones)
# ============================================================

def save_case_roi(roi):
    CASE_ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CASE_ROI_PATH, "w") as f:
        json.dump(roi, f, indent=4)


def _roi_from_legacy_zones():
    """One-time migration from old polygon security_zones.json."""
    if not LEGACY_ZONES_PATH.exists():
        return None
    try:
        with open(LEGACY_ZONES_PATH, "r") as f:
            zones = json.load(f)
        case_zones = zones.get("case_zones") or []
        if not case_zones:
            return None
        polygon = case_zones[0].get("polygon") or []
        if len(polygon) < 3:
            return None
        bbox = polygon_to_bbox(polygon)
        return {
            "id": case_zones[0].get("id", "CASE-001"),
            "name": case_zones[0].get("name", "Jewelry Case"),
            **bbox,
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def load_case_roi():
    if CASE_ROI_PATH.exists():
        with open(CASE_ROI_PATH, "r") as f:
            data = json.load(f)
        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            return data

    legacy = _roi_from_legacy_zones()
    if legacy is not None:
        save_case_roi(legacy)
        print(f"Migrated legacy case polygon → rectangle ROI ({CASE_ROI_PATH})")
        return legacy

    return None


def select_case_roi(frame):
    """Draw a rectangle around the display case. ENTER/SPACE confirm, C cancel."""
    print("Draw a rectangle around the jewelry display case.")
    print("Drag to select, then press ENTER or SPACE. Press C to cancel.")

    selected = cv2.selectROI(
        "Select Jewelry Case",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Select Jewelry Case")

    x, y, w, h = selected
    if w == 0 or h == 0:
        return None

    return {
        "id": "CASE-001",
        "name": "Jewelry Case",
        "x1": int(x),
        "y1": int(y),
        "x2": int(x + w),
        "y2": int(y + h),
    }


def setup_case_roi(video_source):
    cap = open_video_capture(video_source)

    if not cap.isOpened():
        raise Exception("Could not open video source for case ROI setup.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Could not read frame for case ROI setup.")

    if video_source == 0 and MIRROR_WEBCAM:
        frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    roi = select_case_roi(frame)

    if roi is None:
        raise Exception("No case ROI selected.")

    save_case_roi(roi)
    print(f"Case ROI saved to {CASE_ROI_PATH}")
    return roi


def roi_to_box(roi):
    return (int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"]))


def point_inside_rect(point, roi):
    x, y = point
    return roi["x1"] <= x <= roi["x2"] and roi["y1"] <= y <= roi["y2"]


def point_near_rect(point, roi, margin):
    x, y = point
    return (
        roi["x1"] - margin <= x <= roi["x2"] + margin
        and roi["y1"] - margin <= y <= roi["y2"] + margin
    )


def draw_case_roi(frame, roi, color=(0, 140, 255), label=None):
    x1, y1, x2, y2 = roi_to_box(roi)
    label = label or roi.get("name", "Jewelry Case")

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    label_y = max(22, y1 - 8)
    cv2.rectangle(frame, (x1 - 2, label_y - th - 6), (x1 + tw + 8, label_y + 4), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1, label_y), font, scale, color, thickness)


def calculate_motion_in_case_roi(frame, roi):
    """Frame differencing inside the case rectangle — touch / handling proxy."""
    global previous_gray_case_roi

    x1, y1, x2, y2 = roi_to_box(roi)
    roi_frame = frame[y1:y2, x1:x2]

    if roi_frame.size == 0:
        return 0

    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (21, 21), 0)

    if previous_gray_case_roi is None:
        previous_gray_case_roi = gray_roi
        return 0

    frame_diff = cv2.absdiff(previous_gray_case_roi, gray_roi)
    threshold_frame = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
    motion_score = int(cv2.countNonZero(threshold_frame))

    previous_gray_case_roi = gray_roi
    return motion_score


def classify_case_motion(motion_score):
    if motion_score >= CASE_MOTION_HIGH:
        return "HIGH"
    if motion_score >= CASE_MOTION_MEDIUM:
        return "MEDIUM"
    if motion_score >= CASE_MOTION_LOW:
        return "LOW"
    return "NONE"


# ============================================================
# Video capture
# ============================================================

def open_video_capture(source):
    """Open webcam, RTSP/IP stream, or video file."""
    if source == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        return cap
    if isinstance(source, str) and source.lower().startswith("rtsp"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(source)


def _webcam_frame_is_valid(frame):
    """Detect all-black / frozen frames Windows sends when the camera sleeps."""
    if frame is None or frame.size == 0:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > 12.0 and float(gray.std()) > 8.0


def _reopen_webcam_capture(cap):
    """Release and reopen the webcam (DirectShow)."""
    try:
        cap.release()
    except Exception:
        pass
    time.sleep(0.6)
    new_cap = open_video_capture(0)
    if new_cap.isOpened():
        print("Webcam reopened.")
    else:
        print("Webcam reopen failed.")
    return new_cap


# ============================================================
# Geometry helpers
# ============================================================

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


def draw_polygon_zone(frame, polygon, label, color, show_fill=False):
    pts = np.array(polygon, dtype=np.int32)

    if show_fill:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.04, frame, 0.96, 0, frame)

    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

    x, y = polygon[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)

    label_x = int(x)
    label_y = max(25, int(y) - 8)

    cv2.rectangle(
        frame,
        (label_x - 4, label_y - th - 6),
        (label_x + tw + 6, label_y + 4),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        font,
        scale,
        color,
        thickness
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
# Wrist ↔ person association
# ============================================================

def _point_in_box(point, box, margin=0):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin


def apply_wrist_flags_to_detections(detections, wrist_points, case_roi):
    """Tag each tracked person with wrist-near / wrist-inside case flags."""
    for detection in detections:
        detection["wrist_near"] = False
        detection["wrist_inside"] = False

    if not case_roi:
        return False, False

    wrist_near_case = False
    wrist_inside_case = False

    for wx, wy in wrist_points:
        wrist_point = (wx, wy)
        owner = None
        best_area = None

        for detection in detections:
            if not detection.get("registered") or not detection.get("person_near_case"):
                continue
            box = detection["box"]
            if _point_in_box(wrist_point, box, margin=12):
                area = (box[2] - box[0]) * (box[3] - box[1])
                if best_area is None or area < best_area:
                    best_area = area
                    owner = detection

        inside = point_inside_rect(wrist_point, case_roi)
        near = point_near_rect(wrist_point, case_roi, WRIST_NEAR_CASE_MARGIN)

        if inside:
            wrist_inside_case = True
            wrist_near_case = True
            if owner is not None:
                owner["wrist_inside"] = True
                owner["wrist_near"] = True
        elif near:
            wrist_near_case = True
            if owner is not None:
                owner["wrist_near"] = True

    return wrist_near_case, wrist_inside_case


def compute_alarm_level(
    flagged_count,
    wrist_near_case,
    wrist_inside_case,
    crowd_active=False,
    flagged_in_crowd=False,
    staff_mode=False,
):
    if staff_mode:
        return "NONE"
    if flagged_in_crowd and crowd_active:
        return "CROWD_FLAG"
    if wrist_inside_case:
        return "WRIST_INSIDE"
    if wrist_near_case and flagged_count > 0:
        return "WRIST_FLAG"
    if wrist_near_case:
        return "WRIST_NEAR"
    if flagged_count > 0:
        return "FLAG"
    return "NONE"


def draw_person_boxes(frame, detections, person_results, face_by_person=None):
    """Cyan by default; orange FLAGGED; red when wrist is at the case."""
    flagged_count = 0
    frame_h = frame.shape[0]
    if face_by_person is None:
        face_by_person = {}

    for detection in detections:
        if not detection.get("registered"):
            continue

        x1, y1, x2, y2 = detection["box"]
        pid = detection.get("person_id")

        if _is_staff(pid):
            box_color = (160, 160, 160)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            staff_label = f"{pid} STAFF"
            cv2.putText(
                frame,
                staff_label,
                (x1, max(18, y1 - 6)),
                font,
                0.5,
                (200, 200, 200),
                2,
            )
            continue

        if pid:
            font = cv2.FONT_HERSHEY_SIMPLEX
            pid_scale, pid_th = 0.5, 2
            cv2.putText(
                frame,
                pid,
                (x1, max(18, y1 - 6)),
                font,
                pid_scale,
                (255, 255, 255),
                pid_th,
            )

        flagged, _plabel, _pconf = (
            person_can_be_flagged(
                pid, person_results, face_by_person, detection["box"], frame_h
            )
            if pid else (False, "Unknown", 0.0)
        )
        if flagged:
            flagged_count += 1

        wrist_inside = detection.get("wrist_inside", False)
        wrist_near = detection.get("wrist_near", False)

        if wrist_inside:
            box_color = PERSON_BOX_WRIST_INSIDE
            thickness = 3
            label = "WRIST IN CASE"
        elif flagged and wrist_near:
            box_color = PERSON_BOX_WRIST_INSIDE
            thickness = 3
            label = "FLAGGED + WRIST"
        elif wrist_near:
            box_color = PERSON_BOX_WRIST_NEAR
            thickness = 2
            label = "WRIST NEAR CASE"
        elif flagged:
            box_color = PERSON_BOX_FLAGGED
            thickness = 2
            label = f"{pid} {PERSON_FLAG_LABEL}" if pid else PERSON_FLAG_LABEL
        else:
            box_color = PERSON_BOX_NORMAL
            thickness = 2
            label = None

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, th = 0.55, 2
            (tw, tex_h), _ = cv2.getTextSize(label, font, scale, th)
            lx = x1
            ly = min(frame.shape[0] - 8, y2 + tex_h + 8)
            cv2.rectangle(frame, (lx, ly - tex_h - 6), (lx + tw + 8, ly + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (lx + 4, ly), font, scale, box_color, th)

    return flagged_count


# ============================================================
# Mask detection
# ============================================================

def extract_face_crop_for_mask(frame, x1, y1, x2, y2, person_box=None):
    """Face crop for Keras. Webcam extends down to include scarf/cloth below the face box."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if IS_WEBCAM:
        box_h = max(1, y2 - y1)
        box_w = max(1, x2 - x1)
        pad_x = int(box_w * 0.10)
        x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
        y2 = min(h, y2 + int(box_h * 0.55))
        if person_box is not None:
            py1, py2 = person_box[1], person_box[3]
            head_bottom = py1 + int(max(1, py2 - py1) * 0.45)
            y2 = min(h, max(y2, head_bottom))
    else:
        # Slight padding on small face boxes — FAR profile only.
        if IS_VIDEO_FAR:
            fw, fh = max(1, x2 - x1), max(1, y2 - y1)
            if min(fw, fh) < 40:
                pad = max(2, int(min(fw, fh) * 0.12))
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return frame[y1:y2, x1:x2].copy()


def _classify_crop_to_result(crop):
    label, conf, _color, is_masked = classify_mask(crop)
    if label in ("Masked", "Unmasked"):
        return is_masked, label, round(conf, 2)
    return False, "Unknown", 0.0


@tf.function(reduce_retracing=True)
def _mask_model_predict_batch(x):
    return mask_model(x, training=False)


def classify_mask(face_crop):
    if face_crop is None or face_crop.size == 0:
        return "Unknown", 0.0, (255, 255, 255), False

    h, w = face_crop.shape[:2]

    if w < MIN_FACE_BOX_SIZE or h < MIN_FACE_BOX_SIZE:
        return "Unknown", 0.0, (255, 255, 255), False

    resized = cv2.resize(face_crop, MASK_IMG_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_arr = np.expand_dims(rgb, axis=0).astype(np.float32)

    prediction = float(_mask_model_predict_batch(input_arr)[0][0])

    if prediction >= UNMASKED_THRESHOLD:
        return "Unmasked", float(prediction), (0, 0, 255), False
    if prediction <= MASKED_THRESHOLD:
        return "Masked", float(1 - prediction), (0, 255, 0), True
    return "Unknown", 0.0, (255, 255, 255), False


def get_head_crop_from_person(frame, person_box):
    """
    Fallback when face detector fails.

    Uses the top part of the YOLO person box as an estimated head/face region.
    This helps when the face is far away and the face YOLO misses it.
    Returns None if the estimated crop is too small to be useful.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_box

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # Head is usually the top 28% of the person box.
    head_y1 = y1
    head_y2 = y1 + int(box_h * 0.28)

    # Use centre upper region, not full shoulder width.
    cx = (x1 + x2) // 2
    head_w = int(box_w * 0.50)

    hx1 = max(0, cx - head_w // 2)
    hx2 = min(w, cx + head_w // 2)
    hy1 = max(0, head_y1)
    hy2 = min(h, head_y2)

    if hx2 <= hx1 or hy2 <= hy1:
        return None, None

    crop = frame[hy1:hy2, hx1:hx2]
    if crop.shape[0] < MIN_FACE_BOX_SIZE or crop.shape[1] < MIN_FACE_BOX_SIZE:
        return None, None
    return crop, (hx1, hy1, hx2, hy2)

def detect_faces_and_masks(frame):
    global last_mask_label
    global last_mask_confidence
    global last_face_covering_detected
    global last_mask_seen_time

    best_label = "Unknown"
    best_confidence = 0.0
    face_covering_detected = False
    masked_count = 0
    unmasked_count = 0
    annotations = []   # (x1,y1,x2,y2, label, color, conf, primary)

    h, w = frame.shape[:2]

    # imgsz=640 — 4x fewer pixels than 1280, same quality for typical distances.
    # conf=0.50 — rejects bottles/cups/reflections that plagued lower thresholds.
    results = face_model(
        frame,
        imgsz=640,
        conf=0.50,
        iou=0.45,
        max_det=10,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        # Collect valid face boxes; reject tiny crops and tall-narrow shapes
        # (bottles, cups, hands) before touching Keras at all.
        valid = []
        for box in result.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1
            if bw < 40 or bh < 40:
                continue
            if bh > 0 and bw / bh < 0.35:
                continue
            valid.append((float(box.conf[0]), x1, y1, x2, y2))

        if valid:
            # Sort by YOLO confidence; run Keras only on the best face.
            valid.sort(reverse=True)
            _, tx1, ty1, tx2, ty2 = valid[0]
            face_crop = frame[ty1:ty2, tx1:tx2]
            label, mask_confidence, color, is_masked = classify_mask(face_crop)

            best_label = label
            best_confidence = mask_confidence
            face_covering_detected = is_masked
            masked_count  = 1 if label == "Masked"   else 0
            unmasked_count = 1 if label == "Unmasked" else 0

            # Collect annotations to be drawn by the main loop on its live frame
            annotations.append((tx1, ty1, tx2, ty2, label, color, mask_confidence, True))
            for _, x1, y1, x2, y2 in valid[1:]:
                annotations.append((x1, y1, x2, y2, "", (160, 160, 160), 0.0, False))

    if best_label in ["Masked", "Unmasked"] and best_confidence >= 0.80:
        last_mask_label = best_label
        last_mask_confidence = best_confidence
        last_face_covering_detected = face_covering_detected
        last_mask_seen_time = time.time()

    else:
        if time.time() - last_mask_seen_time <= MASK_MEMORY_SECONDS and last_mask_confidence >= 0.85:
            best_label = last_mask_label
            best_confidence = last_mask_confidence
            face_covering_detected = last_face_covering_detected
        else:
            best_label = "Unknown"
            best_confidence = 0.0
            face_covering_detected = False

    return face_covering_detected, best_label, best_confidence, masked_count, unmasked_count, annotations


def get_face_boxes(frame):
    """Detect all faces in the frame using OpenVINO face model.

    Returns a list of (x1, y1, x2, y2, conf, is_frontal) for every valid face.
    """
    h, w = frame.shape[:2]
    results = face_model(frame, imgsz=FACE_DET_IMGSZ, conf=FACE_DET_CONF, iou=0.45, max_det=FACE_DET_MAX_DET, verbose=False)
    result  = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    has_kp = (
        result.keypoints is not None
        and result.keypoints.xy is not None
        and len(result.keypoints.xy) > 0
    )

    faces = []
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_FACE_BOX_SIZE or bh < MIN_FACE_BOX_SIZE:
            continue
        min_aspect = 0.25 if IS_VIDEO_FAR else 0.30
        if bh > 0 and bw / bh < min_aspect:
            continue

        conf = float(box.conf[0])

        # 5-point keypoints: frontal gate for webcam tilt voting only.
        is_frontal = True
        if has_kp and i < len(result.keypoints.xy):
            kp = result.keypoints.xy[i].cpu().numpy()
            if len(kp) >= 3:
                lx, ly = float(kp[0][0]), float(kp[0][1])  # left_eye
                rx, ry = float(kp[1][0]), float(kp[1][1])  # right_eye
                nx, ny = float(kp[2][0]), float(kp[2][1])  # nose
                eyes_ok = not (lx == 0 and ly == 0) and not (rx == 0 and ry == 0)
                nose_ok = not (nx == 0 and ny == 0)
                if eyes_ok and nose_ok:
                    avg_eye_y = (ly + ry) / 2.0
                    if ny < avg_eye_y - 30:
                        is_frontal = False
                    eye_span = abs(rx - lx)
                    if eye_span >= 8:
                        nose_off = abs(nx - (lx + rx) / 2.0) / eye_span
                        if nose_off > 0.22:
                            is_frontal = False

        faces.append((x1, y1, x2, y2, conf, is_frontal))

    return faces


def _person_head_region_box(person_box):
    """Upper-body region where a face is expected (for face↔person matching)."""
    x1, y1, x2, y2 = person_box
    box_h = max(1, y2 - y1)
    pad_x = int((x2 - x1) * 0.08)
    head_y2 = y1 + int(box_h * 0.58)
    return (x1 - pad_x, y1, x2 + pad_x, head_y2)


def _boxes_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_y = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_x * inter_y
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _associate_faces_center(faces, detections):
    """Strict centre-in-box matching (NEAR profile)."""
    assignments = {}
    for face in sorted(faces, key=lambda f: f[4], reverse=True):
        fx1, fy1, fx2, fy2 = face[0], face[1], face[2], face[3]
        fcx = (fx1 + fx2) // 2
        fcy = (fy1 + fy2) // 2
        for det in detections:
            pid = det.get("person_id")
            if not pid or pid in assignments:
                continue
            px1, py1, px2, py2 = det["box"]
            if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                assignments[pid] = face
                break
    return assignments


def _associate_faces_head_region(faces, detections):
    """Head-region IoU matching (FAR profile)."""
    assignments = {}
    registered = [d for d in detections if d.get("person_id") and d.get("registered")]

    for face in sorted(faces, key=lambda f: f[4], reverse=True):
        fx1, fy1, fx2, fy2 = face[0], face[1], face[2], face[3]
        face_box = (fx1, fy1, fx2, fy2)
        fcx = (fx1 + fx2) // 2
        fcy = (fy1 + fy2) // 2
        best_pid = None
        best_score = 0.0

        for det in registered:
            pid = det["person_id"]
            if pid in assignments:
                continue
            head_box = _person_head_region_box(det["box"])
            iou = _boxes_iou(face_box, head_box)
            hx1, hy1, hx2, hy2 = head_box
            center_in = hx1 <= fcx <= hx2 and hy1 <= fcy <= hy2
            score = max(iou, 0.12 if center_in else 0.0)
            if score > best_score:
                best_score = score
                best_pid = pid

        if best_pid and best_score >= 0.05:
            assignments[best_pid] = face

    return assignments


def associate_faces_to_persons(faces, detections):
    if IS_VIDEO_FAR:
        return _associate_faces_head_region(faces, detections)
    assignments = _associate_faces_center(faces, detections)
    if IS_WEBCAM:
        # Tilted faces can sit outside strict centre-in-box — head IoU fallback.
        head_assign = _associate_faces_head_region(faces, detections)
        for pid, face in head_assign.items():
            if pid not in assignments:
                assignments[pid] = face
    return assignments

def _lock_sticky_flag(person_id, label, conf):
    if label != "Masked" or _is_dismissed(person_id):
        return
    if FLAG_STICKY_MODE == "off":
        return
    if FLAG_STICKY_MODE == "high_conf" and conf < FLAG_STICKY_MIN_CONF:
        return
    _sticky_flagged_pids.add(person_id)
    _sticky_flag_snapshot[person_id] = (label, conf)


def _maybe_clear_sticky_flag(person_id, label):
    if person_id not in _sticky_flagged_pids:
        _sticky_unmask_votes.pop(person_id, None)
        return
    if label != "Unmasked":
        _sticky_unmask_votes[person_id] = 0
        return
    _sticky_unmask_votes[person_id] = _sticky_unmask_votes.get(person_id, 0) + 1
    if _sticky_unmask_votes[person_id] >= STICKY_UNMASK_CLEAR:
        _sticky_flagged_pids.discard(person_id)
        _sticky_flag_snapshot.pop(person_id, None)
        _sticky_unmask_votes.pop(person_id, None)


def _clear_mask_votes_for_person(person_id):
    _mask_vote_masked.pop(person_id, None)
    _mask_vote_unmasked.pop(person_id, None)


def _purge_face_queue_for_person(person_id):
    if IS_WEBCAM:
        return
    kept = []
    while True:
        try:
            item = _face_input_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            kept.append(item)
            continue
        _crop, pid = item
        if pid != person_id:
            kept.append(item)
    for item in kept:
        try:
            _face_input_queue.put_nowait(item)
        except queue.Full:
            break


def _store_mask_result(person_id, result_tuple):
    if _is_dismissed(person_id):
        return
    with _face_result_lock:
        _face_results_per_person[person_id] = result_tuple
    _is_masked, label, conf = result_tuple
    if label == "Unmasked":
        _maybe_clear_sticky_flag(person_id, label)


def _queue_video_mask_crop(frame, person_id, person_box, face=None):
    """Enqueue face or head crop for async Keras (video mode)."""
    if _is_dismissed(person_id):
        return False
    now = time.time()
    if now - _face_key_last_classified.get(person_id, 0) < FACE_REQUEUE_INTERVAL:
        return False

    if face is not None:
        fx1, fy1, fx2, fy2, fconf = face[:5]
        if fconf < FACE_CLASSIFY_MIN_CONF:
            return False
        crop = extract_face_crop_for_mask(frame, fx1, fy1, fx2, fy2, person_box)
    else:
        crop, _ = get_head_crop_from_person(frame, person_box)
        if crop is None:
            return False

    _face_key_last_classified[person_id] = now
    item = (crop, person_id)
    if _face_input_queue.full():
        try:
            _face_input_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        _face_input_queue.put_nowait(item)
    except queue.Full:
        return False
    return True


def make_face_key(x1, y1, x2, y2):
    """Stable position-based key for a face box.

    Grid step of 60px means a face centre must move >30px before the key
    changes. At typical walking speed (~60 px/s in a 960-wide frame) a face
    stays in the same cell for ~0.5 s, which safely covers the 0.2 s requeue
    interval so results don't get orphaned by small movements.
    Keys are "F<col>x<row>" — separate namespace from person-ID keys ("P-NNN").
    """
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return f"F{cx // 60}x{cy // 60}"


def _check_head_frontal(frame):
    """Use the pose model's COCO keypoints to decide if the face is frontal.

    COCO indices: 0=nose, 1=left_eye, 2=right_eye.
    In image coordinates y increases downward, so for a frontal face
    the nose must be at or BELOW the eye midpoint (nose_y >= avg_eye_y).
    Tilting up pushes the nose above the eyes in the image → returns False.

    Falls back to True when keypoints are missing or have zero confidence.
    Updates _last_pose_frontal so process_pose_detection doesn't need to
    repeat the inference on the same frame.
    """
    global _last_pose_frontal
    try:
        results = pose_model(frame, imgsz=256, conf=0.35, max_det=1, verbose=False)
        result  = results[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            _last_pose_frontal = True
            return True
        kp = result.keypoints.xy[0].cpu().numpy()
        if len(kp) < 3:
            _last_pose_frontal = True
            return True
        nx, ny = float(kp[0][0]), float(kp[0][1])   # nose
        lx, ly = float(kp[1][0]), float(kp[1][1])   # left_eye
        rx, ry = float(kp[2][0]), float(kp[2][1])   # right_eye
        # Zero-value coords mean the keypoint wasn't detected
        eyes_ok = not (lx == 0 and ly == 0) and not (rx == 0 and ry == 0)
        nose_ok = not (nx == 0 and ny == 0)
        if not eyes_ok or not nose_ok:
            _last_pose_frontal = True
            return True
        avg_eye_y = (ly + ry) / 2.0
        # Nose must be at or below eyes; allow 8 px tolerance
        frontal = ny >= avg_eye_y - 8
        _last_pose_frontal = frontal
        return frontal
    except Exception:
        _last_pose_frontal = True
        return True


def _face_yolo_worker():
    """Thread A: runs YOLO face detection on frames asynchronously."""
    global _face_yolo_result
    while True:
        try:
            frame = _face_yolo_input_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if frame is None:
            break
        boxes = get_face_boxes(frame)
        with _face_yolo_lock:
            _face_yolo_result = boxes
        # Brief yield so Thread A doesn't monopolise the CPU and cause main
        # loop FPS spikes when face YOLO inference is unexpectedly slow.
        time.sleep(0.02)


def _webcam_trust_unmasked(is_frontal):
    """Only label Unmasked when facing the camera."""
    return is_frontal


def _webcam_mask_worker():
    """Webcam: classify latest crop in background with tilt-aware voting."""
    global _face_results_per_person

    while True:
        _webcam_mask_event.wait()
        while True:
            _webcam_mask_event.clear()
            with _webcam_crop_lock:
                crop, person_id, is_frontal = _webcam_latest_crop
            if crop is None or person_id is None:
                break
            if _skip_mask_pipeline(person_id):
                if not _webcam_mask_event.is_set():
                    break
                continue

            label, mask_conf, _color, is_masked = classify_mask(crop)
            if label not in ("Masked", "Unmasked"):
                if not _webcam_mask_event.is_set():
                    break
                continue

            if label == "Masked":
                _mask_vote_masked[person_id] = _mask_vote_masked.get(person_id, 0) + 1
                _mask_vote_unmasked[person_id] = 0
                need = (
                    WEBCAM_MASK_CONFIRM_TILT if not is_frontal
                    else WEBCAM_MASK_CONFIRM
                )
                if _mask_vote_masked[person_id] >= need:
                    _store_mask_result(
                        person_id,
                        (True, "Masked", round(mask_conf, 2)),
                    )
            elif _webcam_trust_unmasked(is_frontal):
                with _face_result_lock:
                    prev = _face_results_per_person.get(person_id)
                if prev and prev[1] == "Masked" and not is_frontal:
                    pass
                else:
                    clear_need = WEBCAM_UNMASK_CLEAR
                    if prev and prev[1] == "Masked":
                        clear_need = WEBCAM_UNMASK_CLEAR_FROM_MASKED
                    _mask_vote_unmasked[person_id] = _mask_vote_unmasked.get(person_id, 0) + 1
                    _mask_vote_masked[person_id] = 0
                    if _mask_vote_unmasked[person_id] >= clear_need:
                        _store_mask_result(
                            person_id,
                            (False, "Unmasked", round(mask_conf, 2)),
                        )
            # tilted: keep current label

            if not _webcam_mask_event.is_set():
                break


def _face_detection_worker():
    """Thread B: per-person Keras mask classification with confirmation buffer.

    CONFIRM_WIN consecutive "Masked" readings are required before the label
    is committed. This eliminates false positives from beards, shadows, and
    momentary misclassifications.

    Fewer consecutive "Unmasked" readings clear a false Masked label so
    unmasked customers are not left flagged for long.
    """
    global _face_results_per_person

    CONFIRM_WIN = VIDEO_MASK_CONFIRM
    CLEAR_WIN   = VIDEO_UNMASK_CLEAR

    while True:
        try:
            item = _face_input_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break

        face_crop = item[0]
        person_id = item[1]
        if _skip_mask_pipeline(person_id):
            continue
        label, mask_conf, _color, is_masked = classify_mask(face_crop)

        if label not in ("Masked", "Unmasked"):
            continue

        if label == "Masked":
            _mask_vote_masked[person_id] = _mask_vote_masked.get(person_id, 0) + 1
            _mask_vote_unmasked[person_id] = 0
            if _mask_vote_masked[person_id] >= CONFIRM_WIN:
                _store_mask_result(
                    person_id, (True, "Masked", round(mask_conf, 2))
                )
        else:
            _mask_vote_unmasked[person_id] = _mask_vote_unmasked.get(person_id, 0) + 1
            _mask_vote_masked[person_id] = 0
            if _mask_vote_unmasked[person_id] >= CLEAR_WIN:
                _store_mask_result(
                    person_id, (False, "Unmasked", round(mask_conf, 2))
                )


# ============================================================
# Person tracking + entrance speed
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_y = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_x * inter_y
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _should_run_throttled(tick, interval):
    """Run every frame when interval<=1; otherwise every Nth frame (first tick runs)."""
    if interval <= 1:
        return True
    return tick % interval == 1


def _reset_yolo_person_tracker():
    """Reset ByteTrack internal state without breaking ultralytics' tracker list.

    Never assign predictor.trackers = [] — on_predict_start skips re-init when
    persist=True and the next track() call raises IndexError.
    """
    predictor = getattr(person_model, "predictor", None)
    if predictor is None:
        return

    trackers = getattr(predictor, "trackers", None)
    if isinstance(trackers, list) and len(trackers) > 0:
        trackers[0].reset()
        return

    if hasattr(predictor, "trackers"):
        del predictor.trackers


def _ensure_person_tracker_ready():
    """Recover from a broken empty tracker list left by a bad reset."""
    predictor = getattr(person_model, "predictor", None)
    if predictor is None:
        return
    trackers = getattr(predictor, "trackers", None)
    if isinstance(trackers, list) and len(trackers) == 0:
        del predictor.trackers


def _person_model_track(frame):
    """Run YOLO person tracking; auto-repair and retry once on IndexError."""
    track_kwargs = dict(
        persist=True,
        classes=[PERSON_CLASS_ID],
        imgsz=PERSON_TRACK_IMGSZ,
        conf=PERSON_CONFIDENCE_THRESHOLD,
        iou=0.55 if IS_VIDEO_FILE else 0.60,
        max_det=PERSON_TRACK_MAX_DET,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    _ensure_person_tracker_ready()
    try:
        return person_model.track(frame, **track_kwargs)
    except IndexError:
        print("Person tracker reset — recovering after video loop")
        _reset_yolo_person_tracker()
        _ensure_person_tracker_ready()
        try:
            return person_model.track(frame, **track_kwargs)
        except IndexError:
            print("Person tracker recovery failed — skipping frame")
            return None


def person_match_score(box_a, box_b, *, center_scale=1.0):
    """Combine IoU and center proximity — robust when ByteTrack IDs flicker."""
    iou = box_iou(box_a, box_b)
    if iou >= PERSON_REASSOC_IOU:
        return iou

    ca, cb = box_center(box_a), box_center(box_b)
    ah = max(1, box_a[3] - box_a[1])
    aw = max(1, box_a[2] - box_a[1])
    scale = max(ah, aw)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    center_limit = PERSON_REASSOC_CENTER * center_scale
    if dist <= scale * center_limit:
        return max(iou, PERSON_REASSOC_IOU + 0.01)
    return iou


def _get_reassoc_limits():
    if IS_VIDEO_FILE:
        return {
            "max_age": VIDEO_REASSOC_MAX_AGE,
            "center_scale": VIDEO_REASSOC_CENTER / PERSON_REASSOC_CENTER,
            "min_score": VIDEO_REASSOC_IOU,
            "mint_score": VIDEO_REASSOC_IOU,
        }
    return {
        "max_age": PERSON_REASSOC_MAX_AGE,
        "center_scale": 1.0,
        "min_score": PERSON_REASSOC_IOU,
        "mint_score": PERSON_REASSOC_IOU,
    }


def _update_person_motion(person_id, box, now):
    cx, cy = box_center(box)
    prev_center = person_last_centers.get(person_id)
    prev_seen = person_last_seen_at.get(person_id)
    if prev_center is not None and prev_seen is not None:
        dt = max(now - prev_seen, 0.001)
        vx = (cx - prev_center[0]) / dt
        vy = (cy - prev_center[1]) / dt
        old_vx, old_vy = person_last_velocity.get(person_id, (0.0, 0.0))
        person_last_velocity[person_id] = (0.65 * vx + 0.35 * old_vx, 0.65 * vy + 0.35 * old_vy)
    person_last_centers[person_id] = (cx, cy)


def _predicted_person_box(person_id, now):
    last_box = person_last_boxes.get(person_id)
    if last_box is None:
        return None
    x1, y1, x2, y2 = last_box
    dt = max(now - person_last_seen_at.get(person_id, now), 0.0)
    vx, vy = person_last_velocity.get(person_id, (0.0, 0.0))
    cx, cy = box_center(last_box)
    pcx = cx + vx * dt
    pcy = cy + vy * dt
    half_w = max(1, (x2 - x1) // 2)
    half_h = max(1, (y2 - y1) // 2)
    return (
        int(pcx - half_w),
        int(pcy - half_h),
        int(pcx + half_w),
        int(pcy + half_h),
    )


def _person_match_for_reassoc(box, person_id, now, limits=None):
    limits = limits or _get_reassoc_limits()
    last_box = person_last_boxes.get(person_id)
    if last_box is None:
        return 0.0
    predicted = _predicted_person_box(person_id, now)
    score_last = person_match_score(
        box, last_box, center_scale=limits["center_scale"]
    )
    score_pred = 0.0
    if predicted is not None:
        score_pred = person_match_score(
            box, predicted, center_scale=limits["center_scale"]
        )
    return max(score_last, score_pred)


def _touch_visible_person_seen_times(detections):
    """Keep last_seen fresh on cached tracking frames so re-link windows don't expire."""
    now = time.monotonic()
    for det in detections:
        pid = det.get("person_id")
        box = det.get("box")
        if not pid or not box:
            continue
        person_last_boxes[pid] = box
        person_last_seen_at[pid] = now
        _update_person_motion(pid, box, now)


def _purge_stale_track_mappings(active_track_ids):
    for track_id in list(byte_track_to_person_id.keys()):
        if track_id not in active_track_ids:
            del byte_track_to_person_id[track_id]


def _duplicate_person_boxes(box_a, box_b):
    """True when two YOLO boxes likely belong to the same individual."""
    iou = box_iou(box_a, box_b)
    if iou >= PERSON_DEDUPE_IOU:
        return True
    if iou < PERSON_DEDUPE_MIN_IOU:
        return False

    ca, cb = box_center(box_a), box_center(box_b)
    ah = max(1, box_a[3] - box_a[1])
    bh = max(1, box_b[3] - box_b[1])
    scale = (ah + bh) / 2.0
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    return dist <= scale * PERSON_DEDUPE_CENTER


def dedupe_person_detections(items):
    """Drop duplicate YOLO boxes on the same individual (keeps highest confidence)."""
    kept = []
    for item in sorted(items, key=lambda row: -row["confidence"]):
        if any(_duplicate_person_boxes(item["box"], other["box"]) for other in kept):
            continue
        kept.append(item)
    return kept


def reset_person_identity_session(*, reset_tracker=False):
    """Reset ordered person IDs (P-001 …) for a new playback pass or engine start."""
    global byte_track_to_person_id
    global person_last_boxes
    global person_last_seen_at
    global person_first_seen_at
    global track_seen_outside_entrance
    global next_person_number
    global last_single_person_id
    global frames_without_person
    global _face_results_per_person
    global _face_key_last_classified
    global _flag_incident_logged_pids
    global _sticky_flagged_pids
    global _sticky_flag_snapshot
    global _dismissed_pids
    global _flag_hold_until
    global _sticky_unmask_votes
    global _mask_vote_masked
    global _mask_vote_unmasked
    global person_last_centers
    global person_last_velocity

    byte_track_to_person_id = {}
    person_last_boxes = {}
    person_last_seen_at = {}
    person_first_seen_at = {}
    person_last_centers = {}
    person_last_velocity = {}
    track_seen_outside_entrance = {}
    next_person_number = 1
    last_single_person_id = None
    frames_without_person = 0

    _face_results_per_person = {}
    _face_key_last_classified = {}
    _flag_qualify_since = {}
    # Keep _flag_incident_logged_pids across video loop / scene-cut resets so each
    # P-ID is logged at most once per engine run (not again on remask or rewind).
    _sticky_flagged_pids = set()
    _sticky_flag_snapshot = {}
    _dismissed_pids = set()
    _staff_pids = set()
    _flag_hold_until = {}
    _sticky_unmask_votes = {}
    _mask_vote_masked = {}
    _mask_vote_unmasked = {}

    if reset_tracker:
        _reset_yolo_person_tracker()


def is_center_in_entrance(center, zones):
    return False


def person_touches_entrance(person_box, zones):
    return False


def _match_dormant_person_id(box, used_person_ids, now):
    """Re-link a box to a P-ID whose YOLO track briefly dropped or moved."""
    limits = _get_reassoc_limits()
    best_pid = None
    best_score = limits["mint_score"]

    for person_id in person_last_boxes:
        if person_id in used_person_ids:
            continue
        last_seen = person_last_seen_at.get(person_id, 0.0)
        max_age = (
            STICKY_REASSOC_MAX_AGE
            if person_id in _sticky_flagged_pids
            else limits["max_age"]
        )
        if now - last_seen > max_age:
            continue
        score = _person_match_for_reassoc(box, person_id, now, limits)
        if score > best_score:
            best_score = score
            best_pid = person_id

    return best_pid


def _match_sticky_person_id(box, used_person_ids, now):
    """Reuse a flagged P-ID when YOLO/ByteTrack briefly drops and re-detects the same person."""
    best_pid = None
    best_score = PERSON_REASSOC_IOU

    for pid in _sticky_flagged_pids:
        if pid in used_person_ids:
            continue
        last_box = person_last_boxes.get(pid)
        if last_box is None:
            continue
        last_seen = person_last_seen_at.get(pid, 0.0)
        if now - last_seen > STICKY_REASSOC_MAX_AGE:
            continue
        score = _person_match_for_reassoc(box, pid, now)
        if score > best_score:
            best_score = score
            best_pid = pid

    return best_pid


def _assign_legacy_person_ids(tracked_items):
    """Legacy: assign P-001… on first YOLO sighting (no entrance gate)."""
    global byte_track_to_person_id
    global person_last_boxes
    global person_last_seen_at
    global person_first_seen_at
    global next_person_number
    global last_single_person_id
    global frames_without_person

    if not tracked_items:
        return []

    tracked_items = dedupe_person_detections(tracked_items)
    limits = _get_reassoc_limits()
    active_track_ids = {
        item.get("track_id")
        for item in tracked_items
        if item.get("track_id") is not None
    }
    _purge_stale_track_mappings(active_track_ids)

    now = time.monotonic()
    n = len(tracked_items)
    assigned = [None] * n
    used_person_ids = set()

    def _mark_seen(person_id, box):
        person_last_boxes[person_id] = box
        person_last_seen_at[person_id] = now
        person_first_seen_at.setdefault(person_id, now)
        _update_person_motion(person_id, box, now)

    for idx, item in enumerate(tracked_items):
        track_id = item.get("track_id")
        if track_id is None or track_id not in byte_track_to_person_id:
            continue
        person_id = byte_track_to_person_id[track_id]
        last_box = person_last_boxes.get(person_id)
        if last_box is not None and person_last_seen_at.get(person_id, 0.0) > 0:
            verify_score = _person_match_for_reassoc(item["box"], person_id, now, limits)
            if verify_score < TRACK_ID_VERIFY_MIN_SCORE:
                del byte_track_to_person_id[track_id]
                continue
        assigned[idx] = person_id
        used_person_ids.add(person_id)
        _mark_seen(person_id, item["box"])

    candidates = []
    for idx, item in enumerate(tracked_items):
        if assigned[idx] is not None:
            continue
        box = item["box"]
        for person_id in person_last_boxes:
            if person_id in used_person_ids:
                continue
            last_seen = person_last_seen_at.get(person_id, 0.0)
            max_age = (
                STICKY_REASSOC_MAX_AGE
                if person_id in _sticky_flagged_pids
                else limits["max_age"]
            )
            if now - last_seen > max_age:
                continue
            score = _person_match_for_reassoc(box, person_id, now, limits)
            if score >= limits["min_score"]:
                candidates.append((idx, person_id, score))

    candidates.sort(key=lambda row: -row[2])
    for idx, person_id, _score in candidates:
        if assigned[idx] is not None or person_id in used_person_ids:
            continue
        assigned[idx] = person_id
        used_person_ids.add(person_id)
        track_id = tracked_items[idx].get("track_id")
        if track_id is not None:
            byte_track_to_person_id[track_id] = person_id
        _mark_seen(person_id, tracked_items[idx]["box"])

    for idx, item in enumerate(tracked_items):
        if assigned[idx] is not None:
            continue
        sticky_match = _match_sticky_person_id(item["box"], used_person_ids, now)
        if sticky_match is not None:
            person_id = sticky_match
        elif (
            n == 1
            and last_single_person_id is not None
            and last_single_person_id not in used_person_ids
            and frames_without_person < SAME_PERSON_GRACE_FRAMES
        ):
            person_id = last_single_person_id
        else:
            dormant_match = _match_dormant_person_id(item["box"], used_person_ids, now)
            if dormant_match is not None:
                reclaim_score = _person_match_for_reassoc(
                    item["box"], dormant_match, now, limits
                )
                if reclaim_score < limits["min_score"]:
                    dormant_match = None
            if dormant_match is not None:
                person_id = dormant_match
            else:
                person_id = f"P-{next_person_number:03d}"
                next_person_number += 1
        assigned[idx] = person_id
        used_person_ids.add(person_id)
        track_id = item.get("track_id")
        if track_id is not None:
            byte_track_to_person_id[track_id] = person_id
        _mark_seen(person_id, item["box"])

    return assigned


def assign_session_person_ids(tracked_items, zones=None):
    """Assign P-001… IDs for this session.

    When ENTRANCE_GATED_IDS is on (default), a new ID is issued only when a
    person enters the entrance polygon. After that the ID sticks via ByteTrack
    + spatial matching even if YOLO flickers.
    """
    global byte_track_to_person_id
    global person_last_boxes
    global person_last_seen_at
    global person_first_seen_at
    global track_seen_outside_entrance
    global next_person_number
    global last_single_person_id
    global frames_without_person

    use_entrance_gate = (
        ENTRANCE_GATED_IDS
        and zones is not None
        and len(zones.get("entrance_zones", [])) > 0
    )
    if not use_entrance_gate:
        return _assign_legacy_person_ids(tracked_items)

    if not tracked_items:
        return []

    tracked_items = dedupe_person_detections(tracked_items)
    now = time.monotonic()
    assigned = [None] * len(tracked_items)
    used_person_ids = set()

    def _mark_seen(person_id, box):
        person_last_boxes[person_id] = box
        person_last_seen_at[person_id] = now
        person_first_seen_at.setdefault(person_id, now)

    # 1) Known ByteTrack id → existing session id
    for idx, item in enumerate(tracked_items):
        track_id = item.get("track_id")
        if track_id is None:
            continue
        if not person_touches_entrance(item["box"], zones):
            track_seen_outside_entrance[track_id] = True
        person_id = byte_track_to_person_id.get(track_id)
        if person_id is None:
            continue
        assigned[idx] = person_id
        used_person_ids.add(person_id)
        _mark_seen(person_id, item["box"])

    # 2) Spatial re-link to a registered person (track flicker / occlusion)
    candidates = []
    for idx, item in enumerate(tracked_items):
        if assigned[idx] is not None:
            continue
        box = item["box"]
        for person_id, last_box in person_last_boxes.items():
            if person_id in used_person_ids:
                continue
            if now - person_last_seen_at.get(person_id, 0.0) > PERSON_REASSOC_MAX_AGE:
                continue
            score = person_match_score(box, last_box)
            if score >= PERSON_REASSOC_IOU:
                candidates.append((idx, person_id, score))

    candidates.sort(key=lambda row: -row[2])
    for idx, person_id, _score in candidates:
        if assigned[idx] is not None or person_id in used_person_ids:
            continue
        assigned[idx] = person_id
        used_person_ids.add(person_id)
        track_id = tracked_items[idx].get("track_id")
        if track_id is not None:
            byte_track_to_person_id[track_id] = person_id
        _mark_seen(person_id, tracked_items[idx]["box"])

    # 3) New ID only when entering the entrance zone
    for idx, item in enumerate(tracked_items):
        if assigned[idx] is not None:
            continue

        track_id = item.get("track_id")
        in_entrance = person_touches_entrance(item["box"], zones)

        if track_id is not None and not in_entrance:
            track_seen_outside_entrance[track_id] = True

        if not in_entrance:
            continue

        # Require having been outside first when possible (avoids loop false-starts)
        if track_id is not None and track_id not in track_seen_outside_entrance:
            if len(person_last_boxes) > 0:
                continue

        person_id = f"P-{next_person_number:03d}"
        next_person_number += 1
        assigned[idx] = person_id
        used_person_ids.add(person_id)
        if track_id is not None:
            byte_track_to_person_id[track_id] = person_id
        _mark_seen(person_id, item["box"])

    return assigned


def process_person_tracking(frame, case_roi):
    """
    YOLO person tracking:
    - stable session person IDs
    - person near jewelry case rectangle
    """
    global last_single_person_id
    global frames_without_person

    empty = {
        "total_people": 0,
        "people_near_case_count": 0,
        "person_near_case": False,
        "active_person_id": "None",
        "detections": [],
    }

    results = _person_model_track(frame)
    if results is None:
        frames_without_person += 1
        return empty

    result = results[0]
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        frames_without_person += 1
        return empty

    boxes = result.boxes
    raw_items = []

    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < PERSON_CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        track_id = int(box.id[0]) if box.id is not None else None
        raw_items.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": confidence,
            "track_id": track_id,
        })

    if not raw_items:
        frames_without_person += 1
        return empty

    tracked_items = dedupe_person_detections(raw_items)
    person_ids = assign_session_person_ids(tracked_items, None)

    case_box = roi_to_box(case_roi) if case_roi else None

    for item, person_id in zip(tracked_items, person_ids):
        person_box = item["box"]
        confidence = item["confidence"]

        person_near_case = False

        if case_box is not None:
            overlap_ratio = get_overlap_ratio(person_box, case_box)
            if overlap_ratio > 0.02:
                person_near_case = True

        detections.append({
            "person_id": person_id,
            "registered": person_id is not None,
            "box": person_box,
            "track_id": item.get("track_id"),
            "person_near_case": person_near_case,
            "wrist_near": False,
            "wrist_inside": False,
            "confidence": confidence,
        })

    if len(detections) == 0:
        frames_without_person += 1
        return empty

    frames_without_person = 0

    registered = [d for d in detections if d.get("registered")]
    if len(registered) == 1:
        last_single_person_id = registered[0]["person_id"]

    total_people = len(registered)
    people_near_case_count = sum(1 for d in registered if d["person_near_case"])
    person_near_case = people_near_case_count > 0

    near_case_people = [d for d in registered if d["person_near_case"]]

    if near_case_people:
        active_person_id = near_case_people[0]["person_id"]
    elif registered:
        active_person_id = registered[0]["person_id"]
    else:
        active_person_id = "None"

    return {
        "total_people": total_people,
        "total_detected": len(detections),
        "people_near_case_count": people_near_case_count,
        "person_near_case": person_near_case,
        "active_person_id": active_person_id,
        "detections": detections,
    }


# ============================================================
# Pose / wrist detection with case rectangle ROI
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

def process_pose_detection(frame, case_roi, detections=None):
    """
    YOLO pose for wrist-at-case checks against the case rectangle.
    Called only when at least one person is near the jewelry case.
    """
    results = pose_model(
        frame,
        imgsz=256,
        conf=POSE_CONFIDENCE_THRESHOLD,
        iou=0.60,
        max_det=10,
        verbose=False,
    )

    result = results[0]
    wrist_points = []

    if result.boxes is None or result.keypoints is None:
        if detections is not None:
            return apply_wrist_flags_to_detections(detections, wrist_points, case_roi)
        return False, False

    keypoints_xy = result.keypoints.xy

    for i in range(len(keypoints_xy)):
        keypoints = keypoints_xy[i].cpu().numpy()
        draw_pose_skeleton(frame, keypoints)

        if i == 0 and len(keypoints) >= 3:
            global _last_pose_frontal
            nx, ny = float(keypoints[0][0]), float(keypoints[0][1])
            lx, ly = float(keypoints[1][0]), float(keypoints[1][1])
            rx, ry = float(keypoints[2][0]), float(keypoints[2][1])
            eyes_ok = not (lx == 0 and ly == 0) and not (rx == 0 and ry == 0)
            nose_ok = not (nx == 0 and ny == 0)
            if eyes_ok and nose_ok:
                _last_pose_frontal = ny >= (ly + ry) / 2.0 - 8

        for wrist_idx in (LEFT_WRIST, RIGHT_WRIST):
            wx, wy = keypoints[wrist_idx]
            if wx > 0 and wy > 0:
                wrist_points.append((int(wx), int(wy)))

    for wx, wy in wrist_points:
        if not case_roi:
            continue
        wrist_point = (wx, wy)

        if point_inside_rect(wrist_point, case_roi):
            cv2.circle(frame, wrist_point, 9, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "Wrist inside case",
                (wx + 10, wy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        elif point_near_rect(wrist_point, case_roi, WRIST_NEAR_CASE_MARGIN):
            cv2.circle(frame, wrist_point, 8, (0, 165, 255), -1)
            cv2.putText(
                frame,
                "Wrist near case",
                (wx + 10, wy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )

    if detections is not None:
        return apply_wrist_flags_to_detections(detections, wrist_points, case_roi)

    wrist_near_case = False
    wrist_inside_case = False
    if case_roi:
        for wx, wy in wrist_points:
            if point_inside_rect((wx, wy), case_roi):
                wrist_inside_case = True
                wrist_near_case = True
            elif point_near_rect((wx, wy), case_roi, WRIST_NEAR_CASE_MARGIN):
                wrist_near_case = True

    return wrist_near_case, wrist_inside_case


# ============================================================
# Customer crowd (excludes staff + dismissed)
# ============================================================

def compute_crowd_metrics(detections, person_results, face_by_person, frame_height):
    global _crowd_active_since

    now = time.monotonic()
    registered = [d for d in detections if d.get("registered") and d.get("person_id")]
    tracked_person_ids = sorted({d["person_id"] for d in registered})

    customers = [
        d for d in registered
        if not _is_staff(d["person_id"]) and not _is_dismissed(d["person_id"])
    ]
    staff_count = sum(1 for d in registered if _is_staff(d["person_id"]))
    customer_count = len(customers)

    flagged_customer_count = 0
    for detection in customers:
        pid = detection["person_id"]
        flagged, _, _ = person_can_be_flagged(
            pid, person_results, face_by_person, detection["box"], frame_height
        )
        if flagged:
            flagged_customer_count += 1

    if customer_count >= CROWD_HIGH:
        crowd_level = "HIGH"
    elif customer_count >= CROWD_WATCH:
        crowd_level = "WATCH"
    elif customer_count >= CROWD_ELEVATED:
        crowd_level = "ELEVATED"
    else:
        crowd_level = "NONE"

    if crowd_level in ("WATCH", "HIGH"):
        if _crowd_active_since is None:
            _crowd_active_since = now
        crowd_seconds = int(now - _crowd_active_since)
        crowd_active = (now - _crowd_active_since) >= CROWD_SUSTAIN_SEC
    else:
        _crowd_active_since = None
        crowd_seconds = 0
        crowd_active = False

    return {
        "customerCount": customer_count,
        "staffCount": staff_count,
        "crowdLevel": crowd_level,
        "crowdActive": crowd_active,
        "crowdSeconds": crowd_seconds,
        "flaggedCustomerCount": flagged_customer_count,
        "trackedPersonIds": tracked_person_ids,
    }


def summarize_flagged_customer_mask(detections, person_results, face_by_person, frame_height):
    """Mask label/conf for CROWD_FLAG — from flagged customers, not scene aggregate."""
    best_label = "Unknown"
    best_conf = 0.0
    flagged_pids = []

    for detection in detections:
        if not detection.get("registered"):
            continue
        pid = detection.get("person_id")
        if not pid or _is_staff(pid) or _is_dismissed(pid):
            continue

        flagged, label, conf = person_can_be_flagged(
            pid, person_results, face_by_person, detection["box"], frame_height
        )
        if not flagged:
            continue

        flagged_pids.append(pid)
        if label == "Masked" and (best_label != "Masked" or conf > best_conf):
            best_label, best_conf = label, conf
        elif best_label == "Unknown" and label in ("Masked", "Unmasked") and conf > best_conf:
            best_label, best_conf = label, conf

    return best_label, best_conf, sorted(flagged_pids)


def crowd_flagged_incident_risk(crowd_level):
    if crowd_level == "HIGH":
        return CROWD_FLAGGED_INCIDENT_RISK_HIGH
    if crowd_level == "WATCH":
        return CROWD_FLAGGED_INCIDENT_RISK_WATCH
    return CROWD_FLAGGED_INCIDENT_RISK_WATCH


def build_crowd_flagged_incident_reasons(crowd_metrics, flagged_pids):
    reasons = ["Face covering / masked person flagged"]
    customer_count = crowd_metrics["customerCount"]
    crowd_level = crowd_metrics["crowdLevel"]
    if crowd_level == "HIGH":
        reasons.append(f"Large customer group ({customer_count} people)")
    else:
        reasons.append(f"Customer group ({customer_count} people)")
    reasons.append("Flagged person in customer group")
    if flagged_pids:
        reasons.append(f"Flagged IDs: {', '.join(flagged_pids)}")
    return reasons


# ============================================================
# Risk model
# ============================================================

def calculate_risk(
    face_covering_detected,
    person_near_case,
    wrist_near_case,
    wrist_inside_case,
    loitering_seconds,
    flagged_count=0,
    case_motion_level="NONE",
    case_interaction=False,
    customer_count=0,
    crowd_level="NONE",
    crowd_active=False,
    flagged_customer_count=0,
    staff_mode_active=False,
):
    risk = 0
    reasons = []
    alert_type = "NORMAL"

    if staff_mode_active:
        return 0, "LOW", "STAFF_MODE", ["Staff mode active — customer alerts paused"]

    if face_covering_detected or flagged_count > 0:
        risk += 40
        reasons.append("Face covering / masked person flagged")
        alert_type = "IDENTITY_WARNING"

    if person_near_case:
        risk += 15
        reasons.append("Person near jewelry display case")

    if wrist_near_case:
        risk += 25
        reasons.append("Wrist/hand near display case boundary")
        if alert_type == "NORMAL":
            alert_type = "CASE_WATCH"

    if wrist_inside_case:
        risk += 45
        reasons.append("Wrist/hand inside display case — direct interaction")
        alert_type = "CRITICAL_ALERT"

    if case_motion_level == "HIGH":
        risk += 20
        reasons.append("High activity inside display case")
        if alert_type in ("NORMAL", "CASE_WATCH"):
            alert_type = "CASE_WATCH"
    elif case_motion_level == "MEDIUM":
        risk += 10
        reasons.append("Movement detected inside display case")

    if case_interaction and not wrist_inside_case:
        risk += 15
        reasons.append("Case interaction detected")
        if alert_type == "NORMAL":
            alert_type = "CASE_WATCH"

    if loitering_seconds >= 20:
        risk += 10
        reasons.append("Loitering near jewelry display")

    if (face_covering_detected or flagged_count > 0) and person_near_case:
        risk += 15
        reasons.append("Flagged person near jewelry case")
        alert_type = "CASE_WATCH"

    if (face_covering_detected or flagged_count > 0) and wrist_near_case:
        risk += 25
        reasons.append("Flagged person with hand near display case")
        alert_type = "CRITICAL_ALERT"

    if (face_covering_detected or flagged_count > 0) and wrist_inside_case:
        risk += 20
        reasons.append("Flagged person with hand inside display case")
        alert_type = "CRITICAL_ALERT"

    if (face_covering_detected or flagged_count > 0) and case_interaction:
        risk += 15
        reasons.append("Flagged person interacting with display case")
        alert_type = "CRITICAL_ALERT"

    if crowd_active:
        if crowd_level == "WATCH":
            risk += 10
            reasons.append(f"Customer group ({customer_count} people)")
            if alert_type == "NORMAL":
                alert_type = "CROWD_WATCH"
        elif crowd_level == "HIGH":
            risk += 20
            reasons.append(f"Large customer group ({customer_count} people)")
            if alert_type == "NORMAL":
                alert_type = "CROWD_WATCH"

        if flagged_customer_count >= 1 and crowd_level in ("WATCH", "HIGH"):
            risk += 25
            reasons.append("Flagged person in customer group")
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

def save_incident_screenshot(frame, risk_score, subject_tag, alert_zone="general"):
    screenshot_dir = Path("data/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_tag = subject_tag.lower().replace(" ", "_")
    safe_zone = alert_zone.lower().replace(" ", "_")
    filename = f"{safe_tag}_{safe_zone}_risk_{risk_score}_{timestamp}.jpg"
    screenshot_path = screenshot_dir / filename

    cv2.imwrite(str(screenshot_path), frame)
    return str(screenshot_path)


def build_risk_description(reasons):
    if not reasons:
        return "No active risk reasons."
    return " | ".join(reasons)


def log_incident_if_needed(
    frame,
    subject_tag,
    risk_score,
    risk_level,
    reasons,
    mask_label,
    mask_confidence,
    face_covering_detected,
    people_near_case_count,
    wrist_near_case,
    loitering_seconds,
    cooldown_frames,
    cooldown_limit,
    alert_zone="general",
):
    global incident_cooldown_frames
    global flag_incident_cooldown_frames
    global wrist_incident_cooldown_frames

    if cooldown_frames > 0:
        return None

    if risk_score < INCIDENT_SAVE_THRESHOLD:
        return None

    screenshot_path = save_incident_screenshot(frame, risk_score, subject_tag, alert_zone)
    risk_description = build_risk_description(reasons)

    incident_id = insert_incident(
        person_id=subject_tag,
        risk_score=int(risk_score),
        risk_level=risk_level,
        risk_description=risk_description,
        mask_status=mask_label,
        mask_confidence=float(mask_confidence),
        face_covering_detected=face_covering_detected,
        people_near_case=int(people_near_case_count),
        wrist_near_case=wrist_near_case,
        motion_level="NONE",
        motion_score=0.0,
        repeated_high_motion=0,
        loitering_seconds=int(loitering_seconds),
        screenshot_path=screenshot_path,
        alert_zone=alert_zone,
    )

    notify_incident_created(
        {
            "id": incident_id,
            "person_id": subject_tag,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score": int(risk_score),
            "risk_level": risk_level,
            "risk_description": risk_description,
            "alert_zone": alert_zone,
            "store": STORE_NAME,
        }
    )

    print(f"Incident saved ({subject_tag}): {risk_level} risk")

    if cooldown_limit == WRIST_INCIDENT_COOLDOWN:
        wrist_incident_cooldown_frames = cooldown_limit
    elif cooldown_limit == FLAG_INCIDENT_COOLDOWN:
        flag_incident_cooldown_frames = cooldown_limit
    else:
        incident_cooldown_frames = cooldown_limit

    return screenshot_path


def log_flag_incidents_for_people(
    entrance_frame,
    store_frame,
    detections,
    person_results,
    face_by_person,
    risk_score,
    risk_level,
    people_near_case_count,
    wrist_near_case,
    loitering_seconds,
):
    """Log one mask-flag incident per person ID (P-001 …), not on every unmask/remask toggle."""
    global _flag_incident_logged_pids

    alert_image = None
    frame_h = entrance_frame.shape[0]

    for detection in detections:
        if not detection.get("registered"):
            continue

        pid = detection.get("person_id")
        if not pid or pid in _flag_incident_logged_pids:
            continue

        flagged, mask_label, mask_confidence = person_can_be_flagged(
            pid, person_results, face_by_person, detection["box"], frame_h
        )
        if not flagged:
            continue

        if MIN_TRACK_SEC_BEFORE_INCIDENT > 0:
            flagged_since = _flag_qualify_since.get(pid)
            if flagged_since is not None:
                if time.monotonic() - flagged_since < MIN_TRACK_SEC_BEFORE_INCIDENT:
                    continue
            elif pid not in _sticky_flagged_pids:
                continue

        near_case = detection.get("person_near_case", False)
        alert_zone = "store" if (near_case or wrist_near_case) else "entrance"
        screenshot_frame = store_frame if alert_zone == "store" else entrance_frame

        person_reasons = ["Face covering / masked person flagged"]
        if near_case:
            person_reasons.append("Person near jewelry display case")

        path = log_incident_if_needed(
            frame=screenshot_frame,
            subject_tag=pid,
            risk_score=MASK_PERSON_INCIDENT_RISK,
            risk_level=MASK_PERSON_INCIDENT_LEVEL,
            reasons=person_reasons,
            mask_label=mask_label,
            mask_confidence=mask_confidence,
            face_covering_detected=True,
            people_near_case_count=people_near_case_count,
            wrist_near_case=wrist_near_case,
            loitering_seconds=loitering_seconds,
            cooldown_frames=0,
            cooldown_limit=0,
            alert_zone=alert_zone,
        )
        if path:
            _flag_incident_logged_pids.add(pid)
            alert_image = path

    return alert_image


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
# Video playback helpers (file mode only)
# ============================================================

IS_VIDEO_FILE = MODE == "video"


def _frame_scene_signature(frame):
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return {"hist": hist, "gray": gray}


def _is_scene_cut(prev_sig, curr_sig):
    if prev_sig is None or curr_sig is None:
        return False, 1.0, 0.0
    corr = cv2.compareHist(prev_sig["hist"], curr_sig["hist"], cv2.HISTCMP_CORREL)
    mad = float(
        np.mean(
            np.abs(
                prev_sig["gray"].astype(np.float32) - curr_sig["gray"].astype(np.float32)
            )
        )
    )
    hard_cut = corr < 0.52 or mad >= SCENE_CUT_MAD_STRONG
    soft_cut = corr < SCENE_CUT_CORREL_THRESHOLD and mad >= SCENE_CUT_MAD_THRESHOLD
    return hard_cut or soft_cut, corr, mad


# ============================================================
# Main vision loop
# ============================================================

def run_vision_loop():
    global engine_running
    global latest_status
    global person_near_start_time
    global incident_cooldown_frames
    global flag_incident_cooldown_frames
    global wrist_incident_cooldown_frames
    global _crowd_incident_logged

    global previous_gray_case_roi

    case_roi = load_case_roi()
    if case_roi is None:
        case_roi = setup_case_roi(VIDEO_SOURCE)

    cap = open_video_capture(VIDEO_SOURCE)

    if not cap.isOpened():
        latest_status["running"] = False
        latest_status["error"] = f"Could not open video source: {VIDEO_SOURCE}"
        return

    video_target_fps = WEBCAM_TARGET_FPS if IS_WEBCAM else VIDEO_TARGET_FPS
    if not IS_WEBCAM:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps and 5 < native_fps <= 60:
            video_target_fps = float(native_fps)

    playback_speed = VIDEO_PLAYBACK_SPEED if IS_VIDEO_FILE else 1.0
    scene_cut_reset = bool(IS_VIDEO_FILE and RUNTIME_IS_DEMO and VIDEO_SCENE_CUT_RESET)
    frame_interval = (1.0 / video_target_fps) / max(playback_speed, 0.05)

    if IS_VIDEO_FILE and playback_speed != 1.0:
        print(
            f"Video playback speed: {playback_speed:.2f}x "
            f"(~{video_target_fps * playback_speed:.1f} effective FPS)"
        )
    print(f"Runtime profile: {RUNTIME_LABEL} ({RUNTIME_PROFILE})")
    if MODE in ("video", "rtsp"):
        print(
            f"Person detector: OpenVINO {_PERSON_MODEL_IMGSZ}px "
            f"({_PERSON_MODEL_PATH})"
        )
    if IS_VIDEO_FILE and scene_cut_reset:
        print(
            f"Scene-cut P-ID reset: ON (demo only — "
            f"corr<{SCENE_CUT_CORREL_THRESHOLD:.2f} or "
            f"mad>={SCENE_CUT_MAD_THRESHOLD:.0f})"
        )
    elif IS_VIDEO_FILE and RUNTIME_IS_LIVE:
        print("Scene-cut P-ID reset: OFF (live file — static session IDs)")

    engine_running = True
    latest_status["running"] = True
    latest_status["playbackSpeed"] = playback_speed
    latest_status["sceneCutResetEnabled"] = scene_cut_reset
    latest_status["sceneCutCount"] = 0
    latest_status["runtimeProfile"] = RUNTIME_PROFILE
    latest_status["runtimeLabel"] = RUNTIME_LABEL

    _loop_tick          = 0
    _cached_tracking    = None
    _cached_pose        = (False, False)
    _cached_case_motion = 0
    _webcam_blank_streak = 0
    _prev_scene_sig    = None
    _last_scene_cut_at  = 0.0
    _scene_cut_count    = 0

    while engine_running:
        _loop_start = time.time()
        ret, frame = cap.read()


        if not ret:
            if IS_RTSP:
                print("RTSP stream lost — reconnecting...")
                cap.release()
                time.sleep(1.0)
                cap = open_video_capture(VIDEO_SOURCE)
                if not cap.isOpened():
                    print("RTSP reconnect failed — stopping engine.")
                    break
                continue
            if not IS_WEBCAM:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reset_person_identity_session(reset_tracker=False)
                _reset_yolo_person_tracker()
                _cached_tracking = None
                _prev_scene_sig = None
                _last_scene_cut_at = 0.0
                time.sleep(0.05)
                continue
            recovered = False
            for _ in range(5):
                time.sleep(0.04)
                ret, frame = cap.read()
                if ret:
                    recovered = True
                    break
            if not recovered:
                print("Webcam lost — attempting reconnect...")
                cap = _reopen_webcam_capture(cap)
                if not cap.isOpened():
                    print("Webcam reconnect failed — stopping engine.")
                    break
            time.sleep(0.04)
            continue

        _loop_tick += 1

        if IS_WEBCAM and MIRROR_WEBCAM:
            frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if IS_VIDEO_FILE and scene_cut_reset:
            curr_scene_sig = _frame_scene_signature(frame)
            now_mono = time.monotonic()
            cut, corr, mad = _is_scene_cut(_prev_scene_sig, curr_scene_sig)
            if cut and now_mono - _last_scene_cut_at >= SCENE_CUT_MIN_INTERVAL_SEC:
                reset_person_identity_session(reset_tracker=True)
                _reset_yolo_person_tracker()
                _cached_tracking = None
                previous_gray_case_roi = None
                _last_scene_cut_at = now_mono
                _scene_cut_count += 1
                latest_status["sceneCutCount"] = _scene_cut_count
                print(
                    f"[VaultVision] Scene cut detected — person IDs reset "
                    f"(#{_scene_cut_count}, corr={corr:.2f}, mad={mad:.0f})"
                )
            _prev_scene_sig = curr_scene_sig

        if IS_WEBCAM and not _webcam_frame_is_valid(frame):
            _webcam_blank_streak += 1
            if _webcam_blank_streak >= WEBCAM_BLANK_RECONNECT:
                print("Webcam blank frames — reopening camera...")
                cap = _reopen_webcam_capture(cap)
                _webcam_blank_streak = 0
            time.sleep(1.0 / WEBCAM_TARGET_FPS)
            continue
        _webcam_blank_streak = 0

        draw_timestamp(frame)

        _t0 = time.time()

        # Person tracking — interval set in _WEBCAM / _VIDEO config above.
        _track_interval = PERSON_TRACK_INTERVAL
        if _should_run_throttled(_loop_tick, _track_interval) or _cached_tracking is None:
            tracking_result  = process_person_tracking(frame, case_roi)
            _cached_tracking = tracking_result
        else:
            tracking_result  = _cached_tracking

        _touch_visible_person_seen_times(tracking_result.get("detections", []))
        _t_person = time.time()

        total_people = tracking_result["total_people"]
        people_near_case_count = tracking_result["people_near_case_count"]
        person_near_case = tracking_result["person_near_case"]
        active_person_id = tracking_result["active_person_id"]
        # ---- Per-person face / mask detection (fully non-blocking) ----------
        # Thread A detects all face boxes; Thread B runs Keras classification.
        # Main loop just drops frames / reads results — zero waiting.
        global _face_frame_counter, _webcam_latest_crop
        _face_frame_counter += 1

        detections = tracking_result.get("detections", [])

        # Drop frame to Thread A (face YOLO) — skip entirely in staff mode.
        if total_people > 0 and not _staff_mode_active:
            try:
                _face_yolo_input_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        # Read latest face boxes from Thread A
        with _face_yolo_lock:
            all_face_boxes = [] if _staff_mode_active else list(_face_yolo_result)

        # Thread A runs continuously — no sync fallback needed.
        face_by_person = associate_faces_to_persons(all_face_boxes, detections)

        # Purge stale mask results (video only — webcam keeps state while tracked).
        active_pids = {d["person_id"] for d in detections if d.get("person_id")}
        _now = time.time()
        for pid in active_pids:
            _face_result_last_seen[pid] = _now
        if not IS_WEBCAM:
            with _face_result_lock:
                stale = [pid for pid in _face_results_per_person
                         if pid not in active_pids
                         and (_now - _face_result_last_seen.get(pid, 0)) > MASK_RESULT_TTL]
                for pid in stale:
                    del _face_results_per_person[pid]
                    _face_result_last_seen.pop(pid, None)
                    _face_key_last_classified.pop(pid, None)

        person_boxes = {
            d["person_id"]: d["box"]
            for d in detections
            if d.get("person_id")
        }

        # Mask detection — see _WEBCAM / _VIDEO config blocks at top of file.
        with _face_result_lock:
            person_results = dict(_face_results_per_person)

        if IS_WEBCAM:
            # Webcam: sync only on first frame + instant mask-on while facing camera.
            classify_pid = active_person_id
            if classify_pid not in face_by_person and face_by_person:
                classify_pid = next(iter(face_by_person))
            if classify_pid in face_by_person:
                face = face_by_person[classify_pid]
                fx1, fy1, fx2, fy2, fconf = face[:5]
                is_frontal = face[5] if len(face) > 5 else True
                if (
                    fconf >= FACE_CLASSIFY_MIN_CONF
                    and not _skip_mask_pipeline(classify_pid)
                ):
                    crop = extract_face_crop_for_mask(
                        frame, fx1, fy1, fx2, fy2, person_boxes.get(classify_pid)
                    )
                    current = person_results.get(classify_pid)
                    run_sync = (
                        current is None
                        or (
                            current[1] == "Unmasked"
                            and is_frontal
                        )
                    )
                    if run_sync:
                        result_tuple = _classify_crop_to_result(crop)
                        _label = result_tuple[1]
                        if current is None:
                            if _label == "Masked":
                                _store_mask_result(classify_pid, result_tuple)
                                if not _skip_mask_pipeline(classify_pid):
                                    person_results[classify_pid] = result_tuple
                            elif _label == "Unmasked" and _webcam_trust_unmasked(is_frontal):
                                _store_mask_result(classify_pid, result_tuple)
                                if not _skip_mask_pipeline(classify_pid):
                                    person_results[classify_pid] = result_tuple
                        elif _label == "Masked":
                            _store_mask_result(classify_pid, result_tuple)
                            if not _skip_mask_pipeline(classify_pid):
                                person_results[classify_pid] = result_tuple
                    with _webcam_crop_lock:
                        _webcam_latest_crop = (crop, classify_pid, is_frontal)
                    _webcam_mask_event.set()
        else:
            if IS_VIDEO_FAR:
                # FAR: queue face/head crops only — worker voting + sustain gate flags.
                for pid, box in person_boxes.items():
                    face = face_by_person.get(pid)
                    if face is not None:
                        _queue_video_mask_crop(frame, pid, box, face=face)
                    else:
                        _queue_video_mask_crop(frame, pid, box, face=None)
            else:
                # NEAR: all linked faces use async voting (no sync bypass — reduces false flags).
                for pid, face in face_by_person.items():
                    if _skip_mask_pipeline(pid):
                        continue
                    person_box = person_boxes.get(pid)
                    fx1, fy1, fx2, fy2, fconf = face[:5]
                    if fconf < FACE_CLASSIFY_MIN_CONF:
                        continue
                    if _now - _face_key_last_classified.get(pid, 0) < FACE_REQUEUE_INTERVAL:
                        continue
                    _face_key_last_classified[pid] = _now
                    crop = extract_face_crop_for_mask(
                        frame, fx1, fy1, fx2, fy2, person_box
                    )
                    item = (crop, pid)
                    if _face_input_queue.full():
                        try:
                            _face_input_queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        _face_input_queue.put_nowait(item)
                    except queue.Full:
                        pass

        with _face_result_lock:
            person_results = dict(_face_results_per_person)
        mask_results = build_display_person_results(active_pids, person_results)

        # Draw face box + label for each tracked person
        masked_count   = 0
        unmasked_count = 0
        face_covering_detected = False

        for pid, face in face_by_person.items():
            if _is_staff(pid):
                continue

            fx1, fy1, fx2, fy2, fconf = face[:5]
            result_tuple = mask_results.get(pid)

            if result_tuple:
                is_masked, plabel, pconf = result_tuple
                if plabel == "Masked":
                    _clr = (0, 255, 0)
                    masked_count += 1
                else:
                    _clr = (0, 0, 255)
                    unmasked_count += 1
                _txt = f"{plabel} {pconf:.2f}"
            else:
                _clr = (160, 160, 160)
                _txt = "Detecting..."

            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), _clr, 2)
            (_tw, _th), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (fx1 - 2, fy1 - _th - 10), (fx1 + _tw + 4, fy1 - 2), (0, 0, 0), -1)
            cv2.putText(frame, _txt, (fx1, fy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _clr, 2)

        counted_pids = set(face_by_person.keys())
        if IS_VIDEO_FAR:
            for pid in person_boxes:
                if pid in counted_pids or _skip_mask_pipeline(pid):
                    continue
                result_tuple = mask_results.get(pid)
                if not result_tuple:
                    continue
                _is_masked, plabel, _pconf = result_tuple
                if plabel == "Masked":
                    masked_count += 1
                elif plabel == "Unmasked":
                    unmasked_count += 1

        # Active-person mask state for risk calculation and status API
        active_result = mask_results.get(active_person_id)
        if active_result:
            _, mask_label, mask_confidence = active_result
        else:
            mask_label, mask_confidence = "Unknown", 0.0

        active_subject_tag = "None"
        if active_person_id != "None":
            active_box = person_boxes.get(active_person_id)
            if active_box is not None:
                can_flag, _label, _conf = person_can_be_flagged(
                    active_person_id,
                    mask_results,
                    face_by_person,
                    active_box,
                    frame.shape[0],
                )
                if can_flag:
                    active_subject_tag = active_person_id

        _t_face = time.time()

        detections = tracking_result.get("detections", [])

        if total_people == 0 or not person_near_case:
            wrist_near_case, wrist_inside_case = False, False
            case_motion_score = 0
            case_motion_level = "NONE"
            _cached_case_motion = 0
            previous_gray_case_roi = None
            _cached_pose = (False, False)
            for detection in detections:
                detection["wrist_near"] = False
                detection["wrist_inside"] = False
        else:
            if _loop_tick % POSE_RUN_INTERVAL == 0:
                wrist_near_case, wrist_inside_case = process_pose_detection(
                    frame, case_roi, detections
                )
                _cached_pose = (wrist_near_case, wrist_inside_case)
            else:
                wrist_near_case, wrist_inside_case = _cached_pose

            if case_roi and _loop_tick % CASE_MOTION_RUN_INTERVAL == 0:
                _cached_case_motion = calculate_motion_in_case_roi(frame, case_roi)
            case_motion_score = _cached_case_motion
            case_motion_level = classify_case_motion(case_motion_score)

        case_interaction = bool(
            wrist_inside_case or case_motion_level in ("MEDIUM", "HIGH")
        )
        _t_pose = time.time()

        if person_near_case:
            if person_near_start_time is None:
                person_near_start_time = time.time()
            loitering_seconds = int(time.time() - person_near_start_time)
        else:
            person_near_start_time = None
            loitering_seconds = 0

        flagged_count = draw_person_boxes(
            frame, detections, mask_results, face_by_person
        )
        crowd_metrics = compute_crowd_metrics(
            detections, mask_results, face_by_person, frame.shape[0]
        )
        face_covering_detected = flagged_count > 0 and not _staff_mode_active
        flagged_people = build_flagged_people(
            detections, mask_results, face_by_person, frame.shape[0]
        )

        alarm_level = compute_alarm_level(
            flagged_count,
            wrist_near_case,
            wrist_inside_case,
            crowd_active=crowd_metrics["crowdActive"],
            flagged_in_crowd=crowd_metrics["flaggedCustomerCount"] > 0,
            staff_mode=_staff_mode_active,
        )
        alarm_active = alarm_level != "NONE" and not _staff_mode_active

        if incident_cooldown_frames > 0:
            incident_cooldown_frames -= 1
        if flag_incident_cooldown_frames > 0:
            flag_incident_cooldown_frames -= 1
        if wrist_incident_cooldown_frames > 0:
            wrist_incident_cooldown_frames -= 1

        risk_score, risk_level, alert_type, reasons = calculate_risk(
            face_covering_detected=face_covering_detected,
            person_near_case=person_near_case,
            wrist_near_case=wrist_near_case,
            wrist_inside_case=wrist_inside_case,
            loitering_seconds=loitering_seconds,
            flagged_count=flagged_count,
            case_motion_level=case_motion_level,
            case_interaction=case_interaction,
            customer_count=crowd_metrics["customerCount"],
            crowd_level=crowd_metrics["crowdLevel"],
            crowd_active=crowd_metrics["crowdActive"],
            flagged_customer_count=crowd_metrics["flaggedCustomerCount"],
            staff_mode_active=_staff_mode_active,
        )

        entrance_reasons = []
        store_reasons = []

        if _staff_mode_active:
            entrance_reasons.append("Staff mode — customer monitoring paused")

        if face_covering_detected:
            entrance_reasons.append("Face covering / masked person flagged")

        if crowd_metrics["crowdActive"] and not _staff_mode_active:
            entrance_reasons.append(
                f"Customer group: {crowd_metrics['customerCount']} people ({crowd_metrics['crowdLevel']})"
            )
            store_reasons.append(
                f"Customer group: {crowd_metrics['customerCount']} people ({crowd_metrics['crowdLevel']})"
            )

        if crowd_metrics["flaggedCustomerCount"] > 0 and crowd_metrics["crowdActive"]:
            entrance_reasons.append("Flagged person in customer group")
            store_reasons.append("Flagged person in customer group")

        if person_near_case:
            store_reasons.append("Person near jewelry display case")

        if wrist_near_case:
            store_reasons.append("Wrist/hand near display case boundary")

        if wrist_inside_case:
            store_reasons.append("Wrist/hand inside display case")

        if case_motion_level in ("MEDIUM", "HIGH"):
            store_reasons.append(f"Case activity: {case_motion_level}")

        if case_interaction and not wrist_inside_case:
            store_reasons.append("Interaction detected at display case")

        if loitering_seconds >= 20:
            store_reasons.append("Loitering near jewelry display")

        if face_covering_detected and person_near_case:
            store_reasons.append("Flagged person near jewelry case")

        if face_covering_detected and wrist_near_case:
            store_reasons.append("Flagged person with hand near display case")

        if face_covering_detected and wrist_inside_case:
            store_reasons.append("Flagged person with hand inside protected case zone")

        # Tab-specific display frames (entrance = clean; store = case rectangle overlay).
        entrance_display_frame = frame.copy()
        store_display_frame = frame.copy()

        if case_roi:
            case_color = (0, 0, 255) if wrist_inside_case else (0, 140, 255)
            draw_case_roi(store_display_frame, case_roi, color=case_color)

        alert_image = None

        if (
            not _staff_mode_active
            and crowd_metrics["flaggedCustomerCount"] > 0
            and crowd_metrics["crowdActive"]
            and not _crowd_incident_logged
        ):
            crowd_mask_label, crowd_mask_conf, crowd_flagged_pids = summarize_flagged_customer_mask(
                detections,
                mask_results,
                face_by_person,
                frame.shape[0],
            )
            crowd_incident_risk = crowd_flagged_incident_risk(crowd_metrics["crowdLevel"])
            crowd_incident_reasons = build_crowd_flagged_incident_reasons(
                crowd_metrics, crowd_flagged_pids
            )
            alert_image = log_incident_if_needed(
                frame=entrance_display_frame,
                subject_tag="CROWD_FLAG",
                risk_score=crowd_incident_risk,
                risk_level=CROWD_FLAGGED_INCIDENT_LEVEL,
                reasons=crowd_incident_reasons,
                mask_label=crowd_mask_label,
                mask_confidence=crowd_mask_conf,
                face_covering_detected=True,
                people_near_case_count=people_near_case_count,
                wrist_near_case=wrist_near_case,
                loitering_seconds=loitering_seconds,
                cooldown_frames=0,
                cooldown_limit=0,
                alert_zone="entrance",
            )
            if alert_image:
                _crowd_incident_logged = True
        elif not _staff_mode_active and (wrist_inside_case or (wrist_near_case and flagged_count > 0)):
            alert_image = log_incident_if_needed(
                frame=store_display_frame,
                subject_tag="WRIST_ALERT",
                risk_score=max(risk_score, 55),
                risk_level=risk_level if risk_score >= 55 else "MEDIUM",
                reasons=reasons,
                mask_label=mask_label,
                mask_confidence=mask_confidence,
                face_covering_detected=face_covering_detected,
                people_near_case_count=people_near_case_count,
                wrist_near_case=wrist_near_case,
                loitering_seconds=loitering_seconds,
                cooldown_frames=wrist_incident_cooldown_frames,
                cooldown_limit=WRIST_INCIDENT_COOLDOWN,
                alert_zone="store",
            )
        elif wrist_near_case:
            alert_image = log_incident_if_needed(
                frame=store_display_frame,
                subject_tag="WRIST_NEAR",
                risk_score=max(risk_score, 40),
                risk_level="MEDIUM",
                reasons=reasons,
                mask_label=mask_label,
                mask_confidence=mask_confidence,
                face_covering_detected=face_covering_detected,
                people_near_case_count=people_near_case_count,
                wrist_near_case=wrist_near_case,
                loitering_seconds=loitering_seconds,
                cooldown_frames=wrist_incident_cooldown_frames,
                cooldown_limit=WRIST_INCIDENT_COOLDOWN,
                alert_zone="store",
            )
        elif not _staff_mode_active and flagged_count > 0:
            alert_image = log_flag_incidents_for_people(
                entrance_display_frame,
                store_display_frame,
                detections,
                mask_results,
                face_by_person,
                risk_score,
                risk_level,
                people_near_case_count,
                wrist_near_case,
                loitering_seconds,
            )

        latest_status = {
            "running": True,
            "mode": MODE,
            "videoProfile": VIDEO_PROFILE if MODE == "video" else None,
            "currentPersonId": active_subject_tag,
            "currentSubjectTag": active_subject_tag,
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
            "caseMotionLevel": case_motion_level,
            "caseMotionScore": int(case_motion_score),
            "caseInteraction": bool(case_interaction),
            "flaggedCount": int(flagged_count),
            "alarmLevel": alarm_level,
            "alarmActive": bool(alarm_active),
            "loiteringSeconds": int(loitering_seconds),
            "riskScore": int(risk_score),
            "riskLevel": risk_level,
            "reasons": reasons,
            "entranceReasons": entrance_reasons,
            "storeReasons": store_reasons,
            "lastAlertImage": alert_image,
            "flaggedPeople": flagged_people,
            "dismissedIds": sorted(_dismissed_pids),
            "staffIds": sorted(_staff_pids),
            "staffModeActive": bool(_staff_mode_active),
            "autoStaffAll": bool(_staff_mode_active),
            "customerCount": int(crowd_metrics["customerCount"]),
            "staffCount": int(crowd_metrics["staffCount"]),
            "crowdLevel": crowd_metrics["crowdLevel"],
            "crowdActive": bool(crowd_metrics["crowdActive"]),
            "crowdSeconds": int(crowd_metrics["crowdSeconds"]),
            "flaggedCustomerCount": int(crowd_metrics["flaggedCustomerCount"]),
            "trackedPersonIds": crowd_metrics["trackedPersonIds"],
            "playbackSpeed": playback_speed,
            "sceneCutResetEnabled": scene_cut_reset,
            "sceneCutCount": _scene_cut_count,
            "flagPolicy": {
                "stickyMode": FLAG_STICKY_MODE,
                "holdSeconds": FLAG_HOLD_SEC,
                "minConf": FLAG_MIN_CONF,
                "incidentDelaySec": MIN_TRACK_SEC_BEFORE_INCIDENT,
            },
        }

        update_latest_frames(entrance_display_frame, store_display_frame)

        # Per-stage timing — printed every 30 frames so the console stays readable.
        if _face_frame_counter % 30 == 0:
            _t_end = time.time()
            total_ms   = (_t_end   - _t0)       * 1000
            person_ms  = (_t_person - _t0)       * 1000
            face_ms    = (_t_face   - _t_person) * 1000
            pose_ms    = (_t_pose   - _t_face)   * 1000
            other_ms   = (_t_end   - _t_pose)    * 1000
            fps        = 1000.0 / max(total_ms, 1)
            print(
                f"FPS:{fps:5.1f} | "
                f"person:{person_ms:5.0f}ms | "
                f"face:{face_ms:5.0f}ms (async) | "
                f"motion:{pose_ms:5.0f}ms | "
                f"other:{other_ms:5.0f}ms | "
                f"total:{total_ms:5.0f}ms"
            )

        # Webcam: pace to ~30 FPS so the loop does not spin at 100+ on light frames.
        if IS_WEBCAM:
            _spent = time.time() - _loop_start
            _wait = (1.0 / WEBCAM_TARGET_FPS) - _spent
            if _wait > 0:
                time.sleep(_wait)
        elif IS_RTSP:
            _spent = time.time() - _loop_start
            _wait = (1.0 / video_target_fps) - _spent
            if _wait > 0:
                time.sleep(_wait)
        else:
            elapsed = time.time() - _loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            elif playback_speed >= 1.0 and elapsed > frame_interval * 2:
                skip = min(int(elapsed / frame_interval) - 1, 8)
                for _ in range(skip):
                    if not cap.grab():
                        break

    cap.release()
    latest_status["running"] = False


# ============================================================
# Public controls used by FastAPI
# ============================================================

def build_flagged_people(detections, person_results, face_by_person, frame_height):
    people = []
    now = time.monotonic()

    for detection in detections:
        if not detection.get("registered"):
            continue
        pid = detection.get("person_id")
        if not pid or _skip_mask_pipeline(pid):
            continue

        flagged, label, conf = person_can_be_flagged(
            pid, person_results, face_by_person, detection["box"], frame_height
        )
        if not flagged:
            continue

        live, _, _ = person_flag_status(person_results, pid)
        hold_active = _flag_hold_until.get(pid, 0) > now and not live
        if live:
            flag_state = "confirmed"
            note = "Live mask detection (confidence {:.0f}%)".format(conf * 100)
        elif hold_active:
            flag_state = "hold"
            note = "Rechecking — brief hold, not a permanent flag"
        elif pid in _sticky_flagged_pids:
            flag_state = "sticky"
            note = "High-confidence lock — clears if unmasked sustained or staff dismisses"
        else:
            flag_state = "confirmed"
            note = "Active flag"

        people.append(
            {
                "personId": pid,
                "maskStatus": label,
                "confidence": conf,
                "flagState": flag_state,
                "note": note,
            }
        )

    return people


def _clear_flag_state_for_person(person_id):
    """Remove all flag/mask state for dismiss or staff marking."""
    _purge_face_queue_for_person(person_id)
    _clear_mask_votes_for_person(person_id)
    with _face_result_lock:
        _face_results_per_person.pop(person_id, None)
    _flag_incident_logged_pids.discard(person_id)
    _sticky_flagged_pids.discard(person_id)
    _sticky_flag_snapshot.pop(person_id, None)
    _flag_hold_until.pop(person_id, None)
    _sticky_unmask_votes.pop(person_id, None)
    _flag_qualify_since.pop(person_id, None)
    _face_key_last_classified.pop(person_id, None)


def mark_person_as_staff(person_id):
    """Exclude this P-ID from mask detection, flags, and customer crowd counts."""
    person_id = str(person_id or "").strip()
    if not person_id:
        return {"marked": False, "error": "person_id is required"}

    _staff_pids.add(person_id)
    _dismissed_pids.discard(person_id)
    _clear_flag_state_for_person(person_id)

    return {
        "marked": True,
        "personId": person_id,
        "staffForSession": True,
    }


def unmark_person_as_staff(person_id):
    person_id = str(person_id or "").strip()
    if not person_id:
        return {"unmarked": False, "error": "person_id is required"}

    _staff_pids.discard(person_id)
    return {"unmarked": True, "personId": person_id}


def _clear_all_customer_monitoring_state():
    """Drop mask/flag state when entering staff mode (restock / closed floor)."""
    global _flag_incident_logged_pids
    global _sticky_flagged_pids
    global _sticky_flag_snapshot
    global _flag_hold_until
    global _sticky_unmask_votes
    global _flag_qualify_since
    global _crowd_active_since

    _purge_all_face_queues()
    with _face_result_lock:
        _face_results_per_person.clear()
    _face_key_last_classified.clear()
    _flag_incident_logged_pids = set()
    _sticky_flagged_pids = set()
    _sticky_flag_snapshot = {}
    _flag_hold_until = {}
    _sticky_unmask_votes = {}
    _flag_qualify_since = {}
    _crowd_active_since = None


def _purge_all_face_queues():
    """Best-effort drain of pending mask crops."""
    for queue in (_face_input_queue,):
        try:
            while True:
                queue.get_nowait()
        except queue.Empty:
            pass
    if IS_WEBCAM:
        with _webcam_crop_lock:
            global _webcam_latest_crop
            _webcam_latest_crop = (None, None, True, False)
        _webcam_mask_event.set()


def set_staff_mode(active: bool):
    global _staff_mode_active
    _staff_mode_active = bool(active)
    if _staff_mode_active:
        _clear_all_customer_monitoring_state()
    return {"staffModeActive": _staff_mode_active, "autoStaffAll": _staff_mode_active}


def dismiss_person_flag(person_id):
    """Staff dismiss: treat this P-ID as unmasked for the rest of the engine session."""
    person_id = str(person_id or "").strip()
    if not person_id:
        return {"dismissed": False, "error": "person_id is required"}

    _dismissed_pids.add(person_id)
    _staff_pids.discard(person_id)
    _clear_flag_state_for_person(person_id)
    with _face_result_lock:
        _face_results_per_person[person_id] = (False, "Unmasked", 1.0)

    return {
        "dismissed": True,
        "personId": person_id,
        "dismissedForSession": True,
    }


# Backwards-compatible alias for the API layer.
def clear_person_flag(person_id):
    result = dismiss_person_flag(person_id)
    if result.get("dismissed"):
        return {
            "cleared": True,
            "personId": result["personId"],
            "clearedForSession": True,
        }
    return {"cleared": False, "error": result.get("error", "dismiss failed")}


def _clear_status_feed():
    """Drop stale flag data when the engine stops."""
    latest_status["flaggedPeople"] = []
    latest_status["dismissedIds"] = []
    latest_status["staffIds"] = []
    latest_status["staffModeActive"] = False
    latest_status["flaggedCount"] = 0
    latest_status["faceCoveringDetected"] = False
    latest_status["alarmLevel"] = "NONE"
    latest_status["alarmActive"] = False


def start_engine():
    global engine_running
    global _face_yolo_result
    global _last_stream_jpeg_at
    global _flag_incident_logged_pids
    global _crowd_incident_logged
    global _sticky_flagged_pids
    global _sticky_flag_snapshot
    global _dismissed_pids
    global _staff_pids
    global _staff_mode_active
    global _crowd_active_since
    global _flag_hold_until
    global _sticky_unmask_votes
    global _mask_vote_masked
    global _mask_vote_unmasked
    global flag_incident_cooldown_frames
    global wrist_incident_cooldown_frames
    global previous_gray_case_roi
    global latest_status
    if engine_running:
        return

    print(f"VaultVision: {describe_source()}")

    reset_person_identity_session(reset_tracker=False)
    _reset_yolo_person_tracker()
    _clear_status_feed()
    _last_stream_jpeg_at = 0.0
    _flag_incident_logged_pids = set()
    _crowd_incident_logged = False
    _sticky_flagged_pids = set()
    _sticky_flag_snapshot = {}
    _dismissed_pids = set()
    _staff_pids = set()
    _staff_mode_active = False
    _crowd_active_since = None
    _flag_hold_until = {}
    _sticky_unmask_votes = {}
    _mask_vote_masked = {}
    _mask_vote_unmasked = {}
    flag_incident_cooldown_frames = 0
    wrist_incident_cooldown_frames = 0
    previous_gray_case_roi = None
    _face_yolo_result = []

    # Warm up tf.function + Keras (matches classify_mask preprocessing).
    _warm = (np.random.rand(1, MASK_IMG_SIZE[0], MASK_IMG_SIZE[1], 3) * 255).astype(np.float32)
    _mask_model_predict_batch(_warm)
    del _warm

    face_yolo_thread = threading.Thread(target=_face_yolo_worker, daemon=True)
    face_yolo_thread.start()

    if IS_WEBCAM:
        mask_thread = threading.Thread(target=_webcam_mask_worker, daemon=True)
    else:
        mask_thread = threading.Thread(target=_face_detection_worker, daemon=True)
    mask_thread.start()

    thread = threading.Thread(target=run_vision_loop, daemon=True)
    thread.start()


def stop_engine():
    global engine_running, latest_status
    engine_running = False
    latest_status["running"] = False
    _clear_status_feed()


def get_latest_status():
    return latest_status
