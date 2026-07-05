import asyncio
import csv
import io
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from backend.auth import require_staff_api_key
from backend.database import (
    get_incident_by_id,
    get_incidents,
    get_staff_actions,
    init_db,
    log_staff_action,
    resolve_screenshot_path,
    update_incident_status,
)
from backend.retail_config import API_KEY, PRODUCT_NAME, PRODUCT_TAGLINE, RUNTIME_LABEL, RUNTIME_PROFILE, STORE_NAME, WEBHOOK_URL
from backend.source_config import MODE, VIDEO_PROFILE, describe as describe_source
from backend.vision_engine import (
    clear_person_flag,
    get_latest_frame_bytes,
    get_latest_status,
    mark_person_as_staff,
    set_staff_mode,
    start_engine,
    stop_engine,
    unmark_person_as_staff,
)

app = FastAPI(title=f"{PRODUCT_NAME} Backend")

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
    return {
        "message": f"{PRODUCT_NAME} backend is running",
        "productName": PRODUCT_NAME,
        "productTagline": PRODUCT_TAGLINE,
        "store": STORE_NAME or None,
    }


@app.get("/health")
def health():
    status = get_latest_status()
    return {
        "ok": True,
        "productName": PRODUCT_NAME,
        "productTagline": PRODUCT_TAGLINE,
        "store": STORE_NAME or None,
        "mode": MODE,
        "videoProfile": VIDEO_PROFILE if MODE in ("video", "rtsp") else None,
        "runtimeProfile": RUNTIME_PROFILE,
        "runtimeLabel": RUNTIME_LABEL,
        "source": describe_source(),
        "engineRunning": bool(status.get("running")),
        "webhookConfigured": bool(WEBHOOK_URL),
        "apiKeyRequired": bool(API_KEY),
    }


@app.post("/start")
def start(_: None = Depends(require_staff_api_key)):
    try:
        start_engine()
        return {"message": "Vision engine started"}
    except Exception as e:
        print("START ENGINE ERROR:", str(e))
        return {"message": "Failed to start vision engine", "error": str(e)}


@app.post("/stop")
def stop(_: None = Depends(require_staff_api_key)):
    stop_engine()
    return {"message": "Vision engine stopped"}


@app.post("/flags/clear")
def flags_clear(payload: dict = Body(default={}), _: None = Depends(require_staff_api_key)):
    person_id = payload.get("person_id") or payload.get("personId")
    staff_name = payload.get("staff_name") or payload.get("staffName") or "staff"
    result = clear_person_flag(person_id)
    if result.get("cleared"):
        log_staff_action(
            "dismiss_flag",
            target_id=result.get("personId"),
            staff_name=staff_name,
            detail="Staff dismissed false-positive mask flag",
        )
    return result


@app.post("/staff/mark")
def staff_mark(payload: dict = Body(default={}), _: None = Depends(require_staff_api_key)):
    person_id = payload.get("person_id") or payload.get("personId")
    staff_name = payload.get("staff_name") or payload.get("staffName") or "staff"
    result = mark_person_as_staff(person_id)
    if result.get("marked"):
        log_staff_action(
            "mark_staff",
            target_id=result.get("personId"),
            staff_name=staff_name,
            detail="Marked as working staff — excluded from mask and crowd",
        )
    return result


@app.post("/staff/unmark")
def staff_unmark(payload: dict = Body(default={}), _: None = Depends(require_staff_api_key)):
    person_id = payload.get("person_id") or payload.get("personId")
    staff_name = payload.get("staff_name") or payload.get("staffName") or "staff"
    result = unmark_person_as_staff(person_id)
    if result.get("unmarked"):
        log_staff_action(
            "unmark_staff",
            target_id=result.get("personId"),
            staff_name=staff_name,
            detail="Removed staff marking",
        )
    return result


@app.post("/staff-mode/start")
def staff_mode_start(payload: dict = Body(default={}), _: None = Depends(require_staff_api_key)):
    staff_name = payload.get("staff_name") or payload.get("staffName") or "staff"
    result = set_staff_mode(True)
    log_staff_action("staff_mode_start", staff_name=staff_name, detail="Staff mode enabled")
    return {"message": "Staff mode enabled", **result}


@app.post("/staff-mode/stop")
def staff_mode_stop(payload: dict = Body(default={}), _: None = Depends(require_staff_api_key)):
    staff_name = payload.get("staff_name") or payload.get("staffName") or "staff"
    result = set_staff_mode(False)
    log_staff_action("staff_mode_stop", staff_name=staff_name, detail="Staff mode disabled")
    return {"message": "Staff mode disabled", **result}


@app.get("/status")
def status():
    data = get_latest_status()
    data["productName"] = PRODUCT_NAME
    data["productTagline"] = PRODUCT_TAGLINE
    data["storeName"] = STORE_NAME or None
    return data


@app.get("/incidents")
def incidents(status_filter: str | None = Query(default=None, alias="status")):
    return get_incidents(limit=200, status=status_filter)


@app.get("/incidents/export")
def incidents_export():
    rows = get_incidents(limit=5000)
    if not rows:
        return Response(content="", media_type="text/csv")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=jewelguard_incidents.csv"},
    )


@app.get("/incidents/{incident_id}")
def incident_detail(incident_id: int):
    incident = get_incident_by_id(incident_id)
    if incident is None:
        return {"error": "Incident not found"}
    return incident


@app.post("/incidents/{incident_id}/acknowledge")
def acknowledge_incident(
    incident_id: int,
    payload: dict = Body(default={}),
    _: None = Depends(require_staff_api_key),
):
    staff_name = payload.get("staff_name") or payload.get("staffName") or "staff"
    staff_note = payload.get("staff_note") or payload.get("staffNote")
    if not update_incident_status(incident_id, "acknowledged", staff_note=staff_note):
        raise HTTPException(status_code=404, detail="Incident not found")
    log_staff_action(
        "acknowledge_incident",
        target_id=str(incident_id),
        staff_name=staff_name,
        detail=staff_note or "",
    )
    return get_incident_by_id(incident_id)


@app.post("/incidents/{incident_id}/resolve")
def resolve_incident(
    incident_id: int,
    payload: dict = Body(default={}),
    _: None = Depends(require_staff_api_key),
):
    staff_name = payload.get("staff_name") or payload.get("staffName") or "staff"
    staff_note = payload.get("staff_note") or payload.get("staffNote")
    if not update_incident_status(incident_id, "resolved", staff_note=staff_note):
        raise HTTPException(status_code=404, detail="Incident not found")
    log_staff_action(
        "resolve_incident",
        target_id=str(incident_id),
        staff_name=staff_name,
        detail=staff_note or "",
    )
    return get_incident_by_id(incident_id)


@app.get("/audit")
def audit_log(limit: int = Query(default=50, ge=1, le=500)):
    return get_staff_actions(limit=limit)


@app.get("/screenshot")
def screenshot(path: str):
    screenshot_path = resolve_screenshot_path(path)
    if screenshot_path is None:
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return FileResponse(str(screenshot_path))


@app.get("/frame")
def frame(view: str = "store"):
    frame_bytes = get_latest_frame_bytes(view=view)
    if frame_bytes is None:
        return Response(
            content=b"",
            status_code=503,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


async def generate_frame_stream(view: str = "store"):
    while True:
        frame_bytes = get_latest_frame_bytes(view=view)
        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        await asyncio.sleep(0.066)


@app.get("/frame_stream")
def frame_stream(view: str = "store"):
    return StreamingResponse(
        generate_frame_stream(view=view),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
