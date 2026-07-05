"""Clear all incidents and screenshot files for a fresh test run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.database import DB_PATH, SCREENSHOT_DIR, clear_all_incidents, init_db

if __name__ == "__main__":
    init_db()
    result = clear_all_incidents(delete_screenshots=True, clear_staff_actions=False)
    print(f"Database: {DB_PATH}")
    print(f"Screenshots folder: {SCREENSHOT_DIR}")
    print(f"Deleted {result['incidentsDeleted']} incidents")
    print(f"Deleted {result['screenshotsDeleted']} screenshot files")
    print("\nRestart the vision engine (Stop then Start on dashboard) so in-memory")
    print("incident cooldowns reset and new flags can log again.")
