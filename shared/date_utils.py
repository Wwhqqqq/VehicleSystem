from __future__ import annotations

import re
from datetime import date, timedelta


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_date(raw_value: str | None) -> str:
    today = date.today()
    if not raw_value:
        return today.isoformat()
    value = raw_value.strip()
    if DATE_PATTERN.match(value):
        return value
    if "明天" in value:
        return (today + timedelta(days=1)).isoformat()
    if "后天" in value:
        return (today + timedelta(days=2)).isoformat()
    if "今天" in value:
        return today.isoformat()
    return today.isoformat()
