"""Smoke test: wait for flags, dismiss one P-ID, verify it stays unmasked."""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
POLL_SEC = 1.5
WAIT_FLAG_SEC = 90
POST_DISMISS_SEC = 20


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return json.loads(r.read().decode())


def post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, method="POST", data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def wait_running():
    for _ in range(30):
        try:
            s = get("/status")
            if s.get("running"):
                return s
        except urllib.error.URLError:
            pass
        time.sleep(1)
    raise RuntimeError("Engine did not reach running state")


def snapshot(label, s):
    flagged = s.get("flaggedPeople") or []
    dismissed = s.get("dismissedIds") or s.get("staffClearedIds") or []
    print(
        f"[{label}] flaggedCount={s.get('flaggedCount', 0)} "
        f"faceCovering={s.get('faceCoveringDetected')} "
        f"dismissed={dismissed}"
    )
    for p in flagged:
        print(
            f"  - {p.get('personId')}: {p.get('maskStatus')} "
            f"{round((p.get('confidence') or 0) * 100)}% ({p.get('flagState')})"
        )
    if not flagged:
        print("  (no flagged people in panel list)")
    stale = s.get("trackedPeople")
    if stale:
        print(f"  WARNING: old API field trackedPeople still present ({len(stale)} entries)")
    return flagged, dismissed


def main():
    print("=== JewelGuard dismiss test ===\n")

    try:
        post("/stop")
        time.sleep(1)
    except Exception:
        pass

    print("Starting engine...")
    print(post("/start"))
    time.sleep(3)
    s = wait_running()
    snapshot("after start", s)

    target = None
    deadline = time.time() + WAIT_FLAG_SEC
    print(f"\nWaiting up to {WAIT_FLAG_SEC}s for a flagged person...")
    while time.time() < deadline:
        s = get("/status")
        flagged, _ = snapshot("poll", s)
        if flagged:
            target = flagged[0]["personId"]
            break
        time.sleep(POLL_SEC)

    if not target:
        print("\nFAIL: No flagged person appeared — cannot test dismiss.")
        sys.exit(1)

    other_ids = {p["personId"] for p in flagged if p["personId"] != target}
    print(f"\nDismissing {target} (others flagged: {sorted(other_ids) or 'none'})...")
    result = post("/flags/clear", {"person_id": target})
    print("Dismiss API response:", result)
    if not result.get("cleared") and not result.get("dismissed"):
        print("FAIL: dismiss API did not confirm success")
        sys.exit(1)

    time.sleep(1)
    s = get("/status")
    snapshot("immediately after dismiss", s)

    dismissed = s.get("dismissedIds") or s.get("staffClearedIds") or []
    if target not in dismissed:
        print(f"FAIL: {target} not in dismissedIds")
        sys.exit(1)

    flagged_ids = {p["personId"] for p in (s.get("flaggedPeople") or [])}
    if target in flagged_ids:
        print(f"FAIL: {target} still in flaggedPeople after dismiss")
        sys.exit(1)

    print(f"\nMonitoring {POST_DISMISS_SEC}s to ensure {target} does not re-flag...")
    reflag = False
    end = time.time() + POST_DISMISS_SEC
    while time.time() < end:
        s = get("/status")
        flagged, dismissed = snapshot("post-dismiss", s)
        flagged_ids = {p["personId"] for p in flagged}
        if target in flagged_ids:
            reflag = True
            print(f"FAIL: {target} re-appeared in flaggedPeople")
            break
        if target not in (s.get("dismissedIds") or s.get("staffClearedIds") or []):
            print(f"FAIL: {target} dropped from dismissedIds")
            sys.exit(1)
        time.sleep(POLL_SEC)

    if reflag:
        sys.exit(1)

    print("\n=== PASS ===")
    print(f"{target} dismissed, stayed out of flaggedPeople for {POST_DISMISS_SEC}s.")
    if other_ids:
        still_flagged = {p["personId"] for p in (get("/status").get("flaggedPeople") or [])}
        overlap = other_ids & still_flagged
        if overlap:
            print(f"Other flagged P-IDs still active (expected): {sorted(overlap)}")
        else:
            print("Note: other previously flagged P-IDs no longer flagged (may have left frame).")


if __name__ == "__main__":
    main()
