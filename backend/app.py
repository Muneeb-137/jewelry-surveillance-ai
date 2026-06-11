from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from pathlib import Path

from backend.vision_engine import (
    start_engine,
    stop_engine,
    get_latest_status,
    get_latest_frame_bytes
)

from backend.database import (
    init_db,
    get_incidents,
    get_incident_by_id
)

app = FastAPI(title="JewelGuard AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.get("/")
def home():
    return {"message": "JewelGuard AI backend is running"}


@app.post("/start")
def start():
    try:
        start_engine()
        return {"message": "Vision engine started"}
    except Exception as e:
        print("START ENGINE ERROR:", str(e))
        return {
            "message": "Failed to start vision engine",
            "error": str(e)
        }


@app.post("/stop")
def stop():
    stop_engine()
    return {"message": "Vision engine stopped"}


@app.get("/status")
def status():
    return get_latest_status()


@app.get("/incidents")
def incidents():
    return get_incidents(limit=100)


@app.get("/incidents/{incident_id}")
def incident_detail(incident_id: int):
    incident = get_incident_by_id(incident_id)

    if incident is None:
        return {"error": "Incident not found"}

    return incident


@app.get("/screenshot")
def screenshot(path: str):
    screenshot_path = Path(path)

    if not screenshot_path.exists():
        return {"error": "Screenshot not found"}

    return FileResponse(str(screenshot_path))


@app.get("/frame")
def frame():
    frame_bytes = get_latest_frame_bytes()

    if frame_bytes is None:
        return {"error": "No frame available yet. Start the engine first."}

    return Response(
        content=frame_bytes,
        media_type="image/jpeg"
    )


def generate_frame_stream():
    while True:
        frame_bytes = get_latest_frame_bytes()

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame_bytes +
                b"\r\n"
            )

        time.sleep(0.05)


@app.get("/frame_stream")
def frame_stream():
    return StreamingResponse(
        generate_frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )