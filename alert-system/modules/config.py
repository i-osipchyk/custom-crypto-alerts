import asyncio
from pathlib import Path
from datetime import datetime, timezone

# ===== Market config =====
KLINE_INTERVAL = "15m"
EMA_PERIOD = 20
DAILY_VOLUME_THRESHOLD = 3_000_000
RELATIVE_VOLUME_THRESHOLD = 4
UP_PCT_THRESHOLD_DAY = 10
UP_PCT_THRESHOLD_NOW = 7

# ===== REST / WS =====
REST_CONCURRENCY = 10
WS_RECONNECT_DELAY = 5

# ===== Time utils =====
def get_today_start_ms():
    now = datetime.now(timezone.utc)
    today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return int(today.timestamp() * 1000)

TODAY_START_MS = get_today_start_ms()

ALERT_COOLDOWN_MS = 60 * 3 * 1000


# ===== PATHS =====
BASE_PATH = "alert-system"

JSON_ALERTS_PATH = f'{BASE_PATH}/alerts/alerts_{datetime.now(timezone.utc)}.json'.replace(" ", "_").replace(":", "-")
JSON_ALERTS_FILE = Path(JSON_ALERTS_PATH)
JSON_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
JSON_ALERTS_FILE_LOCK = asyncio.Lock()
