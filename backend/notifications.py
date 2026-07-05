"""Outbound alerts for real-world ops integrations."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from backend.retail_config import NOTIFY_MIN_RISK, PRODUCT_NAME, STORE_NAME, WEBHOOK_URL


def _post_webhook(payload: dict) -> None:
    if not WEBHOOK_URL:
        return
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": f"{PRODUCT_NAME.replace(' ', '')}/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            pass
    except urllib.error.URLError as exc:
        print(f"Webhook delivery failed: {exc}")


def notify_incident_created(incident: dict) -> None:
    risk = int(incident.get("risk_score") or 0)
    if risk < NOTIFY_MIN_RISK:
        return
    payload = {
        "event": "incident.created",
        "store": STORE_NAME,
        "incident": incident,
    }
    threading.Thread(target=_post_webhook, args=(payload,), daemon=True).start()
