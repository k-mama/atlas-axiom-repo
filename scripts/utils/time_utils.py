from __future__ import annotations
from datetime import datetime, timezone


def utc_now_iso(timespec: str = "seconds") -> str:
    return datetime.now(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")
