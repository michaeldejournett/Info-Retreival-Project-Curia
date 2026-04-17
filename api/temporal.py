"""Stage 1: temporal extraction via fine-tuned tagger + TIMEX3→ISO resolver."""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, time
from typing import List, Optional, Tuple

from local_models import generate_temporal, temporal_loaded
from timex3 import resolve_date_value, resolve_time_value

logger = logging.getLogger(__name__)

# The temporal tagger emits spans like:
#   <TIMEX3 tid="t1" type="DATE" value="2026-04-24">two weeks</TIMEX3>
# We extract all of them and pick the broadest date range + first time range.
_RE_TIMEX = re.compile(
    r'<TIMEX3[^>]*\btype="([A-Z]+)"[^>]*\bvalue="([^"]+)"[^>]*>',
    re.IGNORECASE,
)


def _parse_tags(tagged_text: str) -> List[Tuple[str, str]]:
    return [(m.group(1).upper(), m.group(2)) for m in _RE_TIMEX.finditer(tagged_text)]


def _merge_date_ranges(
    ranges: List[Tuple[date, date]],
) -> Optional[Tuple[date, date]]:
    if not ranges:
        return None
    start = min(r[0] for r in ranges)
    end = max(r[1] for r in ranges)
    return (start, end)


def extract_temporal(
    query: str,
) -> Tuple[Optional[Tuple[date, date]], Optional[Tuple[Optional[time], Optional[time]]]]:
    """Return (date_range, time_range). Either may be None."""
    if not temporal_loaded():
        return None, None

    today = datetime.now().date()
    raw = generate_temporal(query, dct=today.isoformat())
    if not raw:
        return None, None

    tags = _parse_tags(raw)
    if not tags:
        logger.info("Temporal tagger emitted no TIMEX3 tags: %s", raw[:200])
        return None, None

    date_ranges: List[Tuple[date, date]] = []
    time_range: Optional[Tuple[Optional[time], Optional[time]]] = None

    for ttype, tvalue in tags:
        if ttype in ("DATE", "DURATION", "SET"):
            rng = resolve_date_value(tvalue, ref=today)
            if rng:
                date_ranges.append(rng)
            # DURATION/TIME values can also carry time-of-day info
            if time_range is None:
                tr = resolve_time_value(tvalue)
                if tr:
                    time_range = tr
        elif ttype == "TIME":
            # TIMEX3 TIME can be date+time; try both
            rng = resolve_date_value(tvalue, ref=today)
            if rng:
                date_ranges.append(rng)
            if time_range is None:
                tr = resolve_time_value(tvalue)
                if tr:
                    time_range = tr

    return _merge_date_ranges(date_ranges), time_range
