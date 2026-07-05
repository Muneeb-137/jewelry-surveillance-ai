"""Headless mask-pipeline test for a single MP4 (env must be set before imports)."""
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
VIDEO = Path(
    os.environ.get(
        "JEWELGUARD_VIDEO_PATH",
        ROOT / "data" / "sample_videos" / "19.06.2026_17.56.23_REC.mp4",
    )
)

os.environ["JEWELGUARD_MODE"] = "video"
os.environ["JEWELGUARD_VIDEO_PATH"] = str(VIDEO)
os.environ["JEWELGUARD_VIDEO_PROFILE"] = os.environ.get(
    "JEWELGUARD_VIDEO_PROFILE", "near"
)
os.environ["JEWELGUARD_RUNTIME"] = os.environ.get("JEWELGUARD_RUNTIME", "demo")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from backend.database import get_connection, init_db
from backend.vision_engine import get_latest_status, start_engine, stop_engine
import backend.vision_engine as ve

init_db()

cap = cv2.VideoCapture(str(VIDEO))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 14.0
video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
cap.release()
video_sec = video_frames / max(video_fps, 1)

runtime = os.environ["JEWELGUARD_RUNTIME"]
profile = os.environ["JEWELGUARD_VIDEO_PROFILE"]
playback = 0.35 if runtime == "demo" else 1.0
wall_budget = (video_sec / max(playback, 0.05)) * 1.25 + 30

conn = get_connection()
last_id_before = conn.execute("SELECT COALESCE(MAX(id), 0) FROM incidents").fetchone()[0]
conn.close()

print(f"Testing: {VIDEO}")
print(f"Profile: {profile} | runtime: {runtime} | video: {video_sec:.1f}s | budget: {wall_budget:.0f}s wall")
start_engine()

person_mask_history = {}
flag_events = []
st = {}
t0 = time.time()

while time.time() - t0 < wall_budget:
    st = get_latest_status()
    if st.get("error"):
        print("ERROR:", st["error"])
        break
    if not st.get("running") and time.time() - t0 > 10:
        print("Engine stopped.")
        break

    elapsed = round(time.time() - t0, 1)
    with ve._face_result_lock:
        raw_results = {
            pid: (res[1], round(res[2], 2))
            for pid, res in ve._face_results_per_person.items()
        }

    for pid, (label, conf) in raw_results.items():
        hist = person_mask_history.setdefault(pid, [])
        if not hist or hist[-1][1:] != (label, conf):
            hist.append((elapsed, label, conf))

    flagged = st.get("flaggedPeople") or []
    if flagged:
        flag_events.append(
            {
                "t": elapsed,
                "flagged": [
                    {
                        "personId": p["personId"],
                        "maskStatus": p["maskStatus"],
                        "confidence": round(p["confidence"], 2),
                        "flagState": p.get("flagState"),
                    }
                    for p in flagged
                ],
            }
        )

    time.sleep(0.25)

stop_engine()
elapsed_total = round(time.time() - t0, 1)

conn = get_connection()
rows = conn.execute(
    """
    SELECT person_id, mask_status, mask_confidence, risk_score, risk_description, timestamp
    FROM incidents
    WHERE id > ?
    ORDER BY id
    """,
    (last_id_before,),
).fetchall()
conn.close()

def _mask_bursts(hist):
    bursts = []
    current = None
    for t, label, conf in hist:
        if label == "Masked":
            if current is None:
                current = {"start": t, "end": t, "peak_conf": conf}
            else:
                current["end"] = t
                current["peak_conf"] = max(current["peak_conf"], conf)
        elif current is not None:
            bursts.append(current)
            current = None
    if current is not None:
        bursts.append(current)
    return bursts

masked_pids = {
    pid for pid, hist in person_mask_history.items() if any(e[1] == "Masked" for e in hist)
}
unmasked_only = {
    pid for pid, hist in person_mask_history.items() if hist and all(e[1] != "Masked" for e in hist)
}

summary = {
    "video": str(VIDEO),
    "profile": profile,
    "runtime": runtime,
    "video_sec": round(video_sec, 1),
    "elapsed_wall_sec": elapsed_total,
    "tracked_ids": st.get("trackedPersonIds") or [],
    "persons_seen": len(person_mask_history),
    "ever_masked_pids": sorted(masked_pids),
    "never_masked_pids": sorted(unmasked_only),
    "mask_bursts_by_pid": {
        pid: _mask_bursts(hist) for pid, hist in person_mask_history.items() if any(e[1] == "Masked" for e in hist)
    },
    "incidents": [dict(r) for r in rows],
    "flag_event_count": len(flag_events),
    "final": {
        "flaggedCount": st.get("flaggedCount"),
        "faceCoveringDetected": st.get("faceCoveringDetected"),
        "sceneCutCount": st.get("sceneCutCount"),
        "playbackSpeed": st.get("playbackSpeed"),
    },
}

slug = VIDEO.stem
out_path = ROOT / "data" / f"test_mask_{slug}.json"
out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"\nWrote {out_path}")
