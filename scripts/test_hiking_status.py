"""Poll /status while hiking video runs — quick auto-trust smoke test."""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read().decode())


def post(path):
    req = urllib.request.Request(BASE + path, method="POST", data=b"{}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


try:
    post("/stop")
    time.sleep(1)
except Exception:
    pass

print("Starting engine...")
print(post("/start"))

samples = []
for i in range(45):
    time.sleep(2)
    s = get("/status")
    tracked = s.get("trackedPeople") or []
    sample = {
        "t": i * 2,
        "flagged": s.get("flaggedCount", 0),
        "verified": s.get("verifiedUnmaskedIds", []),
        "cleared": s.get("staffClearedIds", []),
        "total": s.get("totalPeople", 0),
        "people": [
            {
                "id": p.get("personId"),
                "flagged": p.get("flagged"),
                "trusted": p.get("trusted"),
                "reason": p.get("trustReason"),
                "mask": p.get("maskStatus"),
                "conf": p.get("confidence"),
            }
            for p in tracked
        ],
    }
    samples.append(sample)
    parts = []
    for p in sample["people"]:
        if p["flagged"]:
            tag = "FLAG"
        elif p["trusted"]:
            tag = "VER"
        else:
            tag = "ok"
        parts.append(f"{p['id']}:{tag} {p['mask']} {p['conf']:.2f}")
    line = (
        f"t={sample['t']:2d}s flagged={sample['flagged']} "
        f"verified={sample['verified']} total={sample['total']}"
    )
    if parts:
        line += " | " + ", ".join(parts)
    print(line)

max_flagged = max(s["flagged"] for s in samples)
ever_verified = sorted({pid for s in samples for pid in s["verified"]})
ever_flagged_ids = sorted(
    {p["id"] for s in samples for p in s["people"] if p["flagged"]}
)
final = get("/status")
print("--- SUMMARY ---")
print("Max flagged count:", max_flagged)
print("Ever verified:", ever_verified)
print("Ever flagged P-IDs:", ever_flagged_ids)
print("Final verified:", final.get("verifiedUnmaskedIds"))
print("Final flagged:", final.get("flaggedCount"))
print("Mode:", final.get("mode"), final.get("videoProfile"))
