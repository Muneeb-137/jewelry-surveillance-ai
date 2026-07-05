"""

JewelGuard — webcam vs video source & pipeline settings

========================================================



HOW TO SWITCH (pick one):



  1. Double-click a starter script (easiest):

       start_webcam.bat          → live webcam

       start_video.bat           → sample video, NEAR profile (store / close)

       start_video_far.bat       → sample video, FAR profile (distant faces)



  2. Set env var before starting:

       set JEWELGUARD_MODE=webcam

       set JEWELGUARD_MODE=video

       set JEWELGUARD_MODE=rtsp

       set JEWELGUARD_RTSP_URL=rtsp://user:pass@192.168.1.50/stream1

       set JEWELGUARD_VIDEO_PROFILE=near   (default)

       set JEWELGUARD_VIDEO_PROFILE=far

       set JEWELGUARD_VIDEO_PATH=C:\\path\\to\\file.mp4

       set JEWELGUARD_RUNTIME=demo          (MP4 review — slow + angle-cut reset)

       set JEWELGUARD_RUNTIME=live          (CCTV/webcam — real-time, static IDs)

       set JEWELGUARD_VIDEO_PLAYBACK_SPEED=0.35   (demo only; 1.0=real-time)

       set JEWELGUARD_VIDEO_SCENE_CUT_RESET=1     (demo only)



  3. Edit DEFAULT_MODE below (when not using the .bat scripts).



Webcam and video use separate settings — edit WEBCAM or VIDEO_* only.
Do not tune webcam via VIDEO_NEAR / VIDEO_FAR (those are for file playback only).

"""



from __future__ import annotations



import os

from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent



# ── Default mode when no env var / .bat script is used ───────────────────────

DEFAULT_MODE = "webcam"  # "webcam" | "video"



# ── Default video profile when mode=video ────────────────────────────────────

DEFAULT_VIDEO_PROFILE = "near"  # "near" | "far"



# ── Default video file (used when mode=video) ────────────────────────────────

DEFAULT_VIDEO_FILE = (

    PROJECT_ROOT / "data" / "sample_videos" / "13.06.2026_16.32.32_REC.mp4"

)



# =============================================================================
# WEBCAM — live camera (tuned & locked — do not copy video settings here)
#
# Pipeline (vision_engine.py):
#   - Sync Keras on first frame + instant mask-on while facing camera
#   - Async worker with yaw-aware frontal gate + mouth keypoint check
#   - Scarf/balaclava: extended crop; Unmasked only if frontal + mouth visible
#   - Side turn: holds Masked (yaw detection); needs 2 votes to clear Masked
#
# Start: start_webcam.bat  or  set JEWELGUARD_MODE=webcam
# =============================================================================

WEBCAM = {

    "FACE_DET_CONF": 0.35,

    "FACE_DET_IMGSZ": 640,          # must match OpenVINO face export size

    "MIN_FACE_BOX_SIZE": 28,

    "FACE_CLASSIFY_MIN_CONF": 0.35,

    "MASKED_THRESHOLD": 0.22,

    "UNMASKED_THRESHOLD": 0.68,

    "FACE_DET_MAX_DET": 15,

    "PERSON_TRACK_INTERVAL": 3,

    "MASK_CONFIRM": 2,              # frontal: 2 votes before Masked

    "MASK_CONFIRM_TILT": 3,         # side angle: 3 readings

    "UNMASK_CLEAR": 1,              # Unmasked while facing camera

    "UNMASK_CLEAR_FROM_MASKED": 2,  # votes needed to drop Masked → Unmasked

    "FLAG_MIN_PERSON_HEIGHT_RATIO": 0.12,  # ignore small background detections

    "FLAG_REQUIRE_FACE_LINK": True,        # must have face linked to person box

    "FLAG_MIN_CONF": 0.82,

    "FLAG_SUSTAIN_SEC": 0.0,

    "MIN_TRACK_SEC_BEFORE_FLAG": 0.0,

    "MIN_TRACK_SEC_BEFORE_INCIDENT": 0.8,

    "FLAG_STICKY_MODE": "off",

    "FLAG_HOLD_SEC": 2.0,

    "FLAG_STICKY_MIN_CONF": 0.90,

    "STICKY_UNMASK_CLEAR": 4,

    "AUTO_TRUST_UNMASKED": True,

    "UNMASK_VERIFY_MIN_CONF": 0.68,

    "UNMASK_VERIFY_SUSTAIN_SEC": 0.5,

    "UNMASKED_LATCH_ENABLED": True,

    "MASKED_OVERRIDE_SUSTAIN_SEC": 0.8,

    "ENTRANCE_GATED_IDS": False,

    "TARGET_FPS": 30,

    "BLANK_RECONNECT_AFTER": 15,

    "INPUT_QUEUE_SIZE": 2,

}



# =============================================================================

# VIDEO NEAR — store / close faces (original conservative thresholds)

# =============================================================================

VIDEO_NEAR = {

    "FACE_DET_CONF": 0.32,

    "FACE_DET_IMGSZ": 640,

    "MIN_FACE_BOX_SIZE": 20,

    "FACE_CLASSIFY_MIN_CONF": 0.30,

    "MASKED_THRESHOLD": 0.18,

    "UNMASKED_THRESHOLD": 0.65,

    "FACE_DET_MAX_DET": 25,

    "PERSON_TRACK_INTERVAL": 1,

    "PERSON_CONF": 0.28,

    "PERSON_MAX_DET": 25,

    "FACE_REQUEUE_INTERVAL": 0.20,

    "MASK_CONFIRM": 4,

    "UNMASK_CLEAR": 2,

    "FLAG_MIN_CONF": 0.82,

    "FLAG_MIN_PERSON_HEIGHT_RATIO": 0.08,  # skip tiny background people

    "FLAG_REQUIRE_FACE_LINK": True,        # face must link to person box (no head-crop flag)

    "FLAG_SUSTAIN_SEC": 0.8,

    "MIN_TRACK_SEC_BEFORE_FLAG": 0.0,

    "MIN_TRACK_SEC_BEFORE_INCIDENT": 2.0,  # sustained flag before SQL log

    "FLAG_STICKY_MODE": "off",

    "FLAG_HOLD_SEC": 2.5,

    "FLAG_STICKY_MIN_CONF": 0.90,

    "STICKY_UNMASK_CLEAR": 5,

    "AUTO_TRUST_UNMASKED": True,

    "UNMASK_VERIFY_MIN_CONF": 0.65,

    "UNMASK_VERIFY_SUSTAIN_SEC": 1.2,

    "UNMASKED_LATCH_ENABLED": True,

    "MASKED_OVERRIDE_SUSTAIN_SEC": 1.5,

    "MASK_RESULT_TTL": 5.0,

    "ENTRANCE_GATED_IDS": False,

    "INPUT_QUEUE_SIZE": 20,

    "TARGET_FPS": 25,

    "STREAM_FPS": 15,

}



# =============================================================================

# VIDEO FAR — distant / small faces (lower thresholds + head-crop fallback)
# Consistency: MASK_CONFIRM + FLAG_SUSTAIN_SEC + MIN_TRACK_SEC_BEFORE_FLAG
# prevent brief false FLAGGED alerts.

# =============================================================================

VIDEO_FAR = {

    "FACE_DET_CONF": 0.20,

    "FACE_DET_IMGSZ": 640,

    "MIN_FACE_BOX_SIZE": 16,

    "FACE_CLASSIFY_MIN_CONF": 0.20,

    "MASKED_THRESHOLD": 0.10,

    "UNMASKED_THRESHOLD": 0.55,

    "FACE_DET_MAX_DET": 25,

    "PERSON_TRACK_INTERVAL": 1,

    "PERSON_CONF": 0.22,

    "PERSON_MAX_DET": 25,

    "FACE_REQUEUE_INTERVAL": 0.15,

    "MASK_CONFIRM": 4,

    "UNMASK_CLEAR": 4,

    "FLAG_MIN_CONF": 0.82,

    "FLAG_MIN_PERSON_HEIGHT_RATIO": 0.0,

    "FLAG_REQUIRE_FACE_LINK": False,

    "FLAG_SUSTAIN_SEC": 1.0,

    "MIN_TRACK_SEC_BEFORE_FLAG": 0.8,

    "MIN_TRACK_SEC_BEFORE_INCIDENT": 2.0,

    "UNMASKED_LATCH_ENABLED": True,

    "MASKED_OVERRIDE_SUSTAIN_SEC": 1.2,

    "MASK_RESULT_TTL": 6.0,

    "ENTRANCE_GATED_IDS": False,

    "INPUT_QUEUE_SIZE": 20,

    "TARGET_FPS": 25,

    "STREAM_FPS": 15,

}



# Back-compat alias

VIDEO = VIDEO_NEAR





def resolve_mode() -> str:

    raw = os.environ.get("JEWELGUARD_MODE", DEFAULT_MODE).strip().lower()

    if raw in ("webcam", "cam", "0", "live"):

        return "webcam"

    if raw in ("video", "file", "mp4"):

        return "video"

    if raw in ("rtsp", "stream", "ipcam", "ip"):

        return "rtsp"

    return DEFAULT_MODE if DEFAULT_MODE in ("webcam", "video", "rtsp") else "webcam"





def resolve_video_profile() -> str:

    raw = os.environ.get("JEWELGUARD_VIDEO_PROFILE", DEFAULT_VIDEO_PROFILE).strip().lower()

    if raw in ("far", "distance", "distant", "small"):

        return "far"

    return "near"





def resolve_video_source():

    mode = resolve_mode()

    if mode == "webcam":

        return 0

    if mode == "rtsp":

        url = os.environ.get("JEWELGUARD_RTSP_URL", "").strip()

        if not url:

            raise ValueError("Set JEWELGUARD_RTSP_URL for rtsp mode (e.g. rtsp://user:pass@camera/stream)")

        return url

    env_path = os.environ.get("JEWELGUARD_VIDEO_PATH", "").strip()

    if env_path:

        return str(Path(env_path))

    return str(DEFAULT_VIDEO_FILE)





MODE = resolve_mode()

VIDEO_PROFILE = resolve_video_profile()

VIDEO_SOURCE = resolve_video_source()

IS_WEBCAM = MODE == "webcam"

IS_RTSP = MODE == "rtsp"

IS_VIDEO_FAR = MODE == "video" and VIDEO_PROFILE == "far"

ACTIVE = WEBCAM if IS_WEBCAM else (VIDEO_FAR if IS_VIDEO_FAR else VIDEO_NEAR)





def describe() -> str:

    if IS_WEBCAM:

        return "mode=WEBCAM (live camera)"

    if IS_RTSP:

        return f"mode=RTSP profile={VIDEO_PROFILE} url={VIDEO_SOURCE}"

    return f"mode=VIDEO profile={VIDEO_PROFILE} file={VIDEO_SOURCE}"

