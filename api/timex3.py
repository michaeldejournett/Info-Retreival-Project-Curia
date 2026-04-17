"""Resolve TIMEX3 value strings into concrete (date_from, date_to) / (time_from, time_to) pairs.

Handles the value patterns emitted by the temporal_tagger_roberta2roberta model.
Reference for TIMEX3 value grammar: https://www.timeml.org/publications/timeMLdocs/timeml_1.2.1.html
"""
from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, time, timedelta
from typing import Optional, Tuple

# Season → (start_month, end_month). Standard northern-hemisphere convention.
_SEASONS = {
    "SP": (3, 5),
    "SU": (6, 8),
    "FA": (9, 11),
    "WI": (12, 2),
}

# Time-of-day tags → (start HH:MM, end HH:MM).
_TOD = {
    "MO": (time(6, 0), time(12, 0)),   # morning
    "AF": (time(12, 0), time(17, 0)),  # afternoon
    "EV": (time(17, 0), time(21, 0)),  # evening
    "NI": (time(21, 0), time(23, 59)), # night
    "DT": (time(12, 0), time(18, 0)),  # daytime (approx)
}

_RE_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_RE_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_RE_YEAR = re.compile(r"^(\d{4})$")
_RE_WEEK = re.compile(r"^(\d{4})-W(\d{2})(-WE)?$")
_RE_SEASON = re.compile(r"^(\d{4})-(SP|SU|FA|WI)$")
_RE_DURATION = re.compile(r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?$")
_RE_DATETIME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):?(\d{2})?")
_RE_TIME_ONLY = re.compile(r"^T(\d{2}):?(\d{2})?$")


def _month_end(y: int, m: int) -> date:
    return date(y, m, monthrange(y, m)[1])


def _iso_week_to_date(y: int, w: int, dow: int = 1) -> date:
    """Monday of ISO week (dow=1) or other weekday."""
    return date.fromisocalendar(y, w, dow)


def resolve_date_value(value: str, ref: date) -> Optional[Tuple[date, date]]:
    """Convert a TIMEX3 value to an inclusive (start, end) date range.
    Returns None for values that don't represent a date."""
    if not value or value in ("null", "NULL"):
        return None

    v = value.strip()

    if v in ("PRESENT_REF",):
        return (ref, ref)
    if v == "FUTURE_REF":
        return (ref, ref + timedelta(days=30))
    if v == "PAST_REF":
        return (ref - timedelta(days=30), ref)

    m = _RE_DATETIME.match(v)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dd = date(y, mo, d)
            return (dd, dd)
        except ValueError:
            return None

    m = _RE_DATE.match(v)
    if m:
        try:
            dd = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return (dd, dd)
        except ValueError:
            return None

    m = _RE_WEEK.match(v)
    if m:
        y, w = int(m.group(1)), int(m.group(2))
        try:
            mon = _iso_week_to_date(y, w, 1)
        except ValueError:
            return None
        if m.group(3):  # -WE suffix → Saturday..Sunday
            return (mon + timedelta(days=5), mon + timedelta(days=6))
        return (mon, mon + timedelta(days=6))

    m = _RE_MONTH.match(v)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        try:
            return (date(y, mo, 1), _month_end(y, mo))
        except ValueError:
            return None

    m = _RE_YEAR.match(v)
    if m:
        y = int(m.group(1))
        return (date(y, 1, 1), date(y, 12, 31))

    m = _RE_SEASON.match(v)
    if m:
        y, s = int(m.group(1)), m.group(2)
        start_mo, end_mo = _SEASONS[s]
        if s == "WI":  # winter spans year boundary (Dec..Feb next year)
            return (date(y, 12, 1), _month_end(y + 1, 2))
        return (date(y, start_mo, 1), _month_end(y, end_mo))

    m = _RE_DURATION.match(v)
    if m:
        yrs = int(m.group(1) or 0)
        mos = int(m.group(2) or 0)
        wks = int(m.group(3) or 0)
        dys = int(m.group(4) or 0)
        days = yrs * 365 + mos * 30 + wks * 7 + dys
        if days <= 0:
            return None
        return (ref, ref + timedelta(days=days))

    return None


def resolve_time_value(value: str) -> Optional[Tuple[Optional[time], Optional[time]]]:
    """Convert a TIMEX3 TIME-type value (or time-of-day tag) to an (start, end) time range."""
    if not value or value in ("null", "NULL"):
        return None

    v = value.strip()

    # Strip leading "T" if present for TOD lookup
    tod_key = v[1:] if v.startswith("T") and len(v) <= 3 else v
    if tod_key in _TOD:
        start, end = _TOD[tod_key]
        return (start, end)

    m = _RE_DATETIME.match(v)
    if m:
        h = int(m.group(4))
        mm = int(m.group(5)) if m.group(5) else 0
        if 0 <= h < 24:
            start = time(h, mm)
            end = time(h + 1, mm) if h < 23 else time(23, 59)
            return (start, end)

    m = _RE_TIME_ONLY.match(v)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2)) if m.group(2) else 0
        if 0 <= h < 24:
            start = time(h, mm)
            end = time(h + 1, mm) if h < 23 else time(23, 59)
            return (start, end)

    return None
