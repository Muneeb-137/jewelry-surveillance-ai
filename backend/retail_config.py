"""Production-oriented settings via environment variables."""

from __future__ import annotations

import os

# Optional API key for staff actions (start/stop/dismiss/ack). Empty = disabled.
API_KEY = os.environ.get("JEWELGUARD_API_KEY", "").strip()

# Display branding (dashboard header, API title)
PRODUCT_NAME = os.environ.get("JEWELGUARD_PRODUCT_NAME", "VaultVision").strip()
PRODUCT_TAGLINE = os.environ.get(
    "JEWELGUARD_PRODUCT_TAGLINE", "Retail Surveillance System"
).strip()

# Optional location label (e.g. "Downtown Branch") — shown in status bar when set
STORE_NAME = os.environ.get("JEWELGUARD_STORE_NAME", "").strip()

# POST JSON to this URL when incidents are created (Slack, Teams, custom SIEM, etc.)
WEBHOOK_URL = os.environ.get("JEWELGUARD_WEBHOOK_URL", "").strip()

# Only webhook-notify incidents at or above this risk score
NOTIFY_MIN_RISK = int(os.environ.get("JEWELGUARD_NOTIFY_MIN_RISK", "40"))

# Delete incidents older than this many days on startup (0 = keep forever)
RETENTION_DAYS = int(os.environ.get("JEWELGUARD_RETENTION_DAYS", "90"))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(value, maximum))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _resolve_mode_for_runtime() -> str:
    raw = os.environ.get("JEWELGUARD_MODE", "webcam").strip().lower()
    if raw in ("video", "file", "mp4"):
        return "video"
    if raw in ("rtsp", "stream", "ipcam", "ip"):
        return "rtsp"
    return "webcam"


def resolve_runtime_profile() -> str:
    """demo = sample MP4 review (slow playback, angle-cut reset). live = CCTV/webcam."""
    raw = os.environ.get("JEWELGUARD_RUNTIME", "").strip().lower()
    if raw in ("demo", "file", "review", "training"):
        return "demo"
    if raw in ("live", "cctv", "production", "store"):
        return "live"
    return "demo" if _resolve_mode_for_runtime() == "video" else "live"


RUNTIME_PROFILE = resolve_runtime_profile()
RUNTIME_IS_DEMO = RUNTIME_PROFILE == "demo"
RUNTIME_IS_LIVE = RUNTIME_PROFILE == "live"
RUNTIME_LABEL = "Demo" if RUNTIME_IS_DEMO else "Live"


_default_playback = 0.35 if RUNTIME_IS_DEMO else 1.0
_default_scene_cut = RUNTIME_IS_DEMO

# Video file only: 1.0 = real-time, 0.25 ≈ 4× slower (demo default 0.35)
VIDEO_PLAYBACK_SPEED = _env_float(
    "JEWELGUARD_VIDEO_PLAYBACK_SPEED", _default_playback, minimum=0.05, maximum=4.0
)

# Demo MP4 only: reset P-001… on hard angle cuts. Live CCTV keeps IDs for the session.
VIDEO_SCENE_CUT_RESET = _env_bool("JEWELGUARD_VIDEO_SCENE_CUT_RESET", _default_scene_cut)

# Histogram correlation below this → treat frame as a new camera angle
SCENE_CUT_CORREL_THRESHOLD = _env_float(
    "JEWELGUARD_SCENE_CUT_THRESHOLD", 0.62, minimum=0.2, maximum=0.95
)

# Mean pixel change on a downscaled frame — used with histogram for angle cuts
SCENE_CUT_MAD_THRESHOLD = _env_float(
    "JEWELGUARD_SCENE_CUT_MAD", 32.0, minimum=5.0, maximum=100.0
)

# Strong pixel jump alone also counts as a hard cut (typical CCTV angle switch)
SCENE_CUT_MAD_STRONG = _env_float(
    "JEWELGUARD_SCENE_CUT_MAD_STRONG", 58.0, minimum=20.0, maximum=120.0
)

# Minimum seconds between scene-cut resets (avoids false triggers on motion blur)
SCENE_CUT_MIN_INTERVAL_SEC = _env_float(
    "JEWELGUARD_SCENE_CUT_MIN_SEC", 0.8, minimum=0.2, maximum=10.0
)
