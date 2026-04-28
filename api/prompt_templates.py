from __future__ import annotations

from datetime import datetime


EXPAND_PROMPT_TEMPLATE = """Today is {now} ({weekday}).

You are helping search a university event database. Given a query, do two things:

1. KEYWORD EXPANSION: Extract the core topics and expand each general term into specific concrete examples.
   - Do not expand on time words (e.g. "tonight", "weekend", "six o'clock", "6:00") or stop words (e.g. "looking", "events")
   - Include the original term AND its specific instances (e.g. "food" → ["food", "pasta", "pizza", "tacos", "burger", "salad", "BBQ", "sushi"])
   - Include synonyms, related activities, and subcategories (e.g. "music" → ["music", "concert", "jazz", "rock", "band", "choir", "orchestra", "recital"])
   - Include relevant people/roles (e.g. "volunteer" → ["volunteer", "service", "community", "outreach", "nonprofit"])
   - Keep all keywords lowercase, single words or short phrases
   - Produce a comprehensive list ensuring broad coverage of potential matches, but avoid irrelevant terms
   - Try to produce at least 50 key words for a general query, fewer for a specific query

2. DATE EXTRACTION: Resolve any relative date references using today's date. Always use YYYY-MM-DD format (e.g. 2026-04-01). Do NOT use formats like 4/1/2026, April 1, or MM-DD-YYYY. Always interpret period references as FULL ranges, not single days.
   - "today" / "tonight" → just today's date (date_from = date_to = today)
   - "tomorrow" → just tomorrow's date (date_from = date_to = tomorrow)
   - "this week" → today through the coming Sunday (inclusive)
   - "next week" → next Monday through next Sunday (full 7-day week)
   - "this weekend" → nearest Saturday AND Sunday (date_from = Sat, date_to = Sun)
   - "next weekend" → the Saturday and Sunday after next (date_from = Sat, date_to = Sun)
   - "this month" → today through the last day of the current month
   - "next month" → first day of next month through the last day of next month (full month)
   - "in april" / "during april" → April 1 through April 30 of the relevant year
   - "two weeks from now" → today through 14 days from today
   - "soon" / "upcoming" → today through 7 days from today
   - IMPORTANT: Default to broad ranges. Only use a single day (date_from = date_to) when the user specifies an exact date or says "exactly" / "on that day"
   - If no date is mentioned, return null for both date fields
   - Do not include time words inside of key words (e.g. "tonight" should not be a keyword, only a date reference)

3. TIME EXTRACTION: Resolve time-of-day references into 24h HH:MM bounds.
   - "morning" → time_from: "06:00", time_to: "12:00"
   - "afternoon" → time_from: "12:00", time_to: "17:00"
   - "evening" / "tonight" → time_from: "17:00", time_to: "21:00"
   - "night" / "late night" → time_from: "21:00", time_to: null
   - "after 6pm" / "after 18:00" → time_from: "18:00", time_to: null
   - "before noon" / "before 12pm" → time_from: null, time_to: "12:00"
   - specific time like "at 3pm" → time_from: "15:00", time_to: "16:00"
   - If no time is mentioned, return null for both time fields

Return ONLY a JSON object with no extra text.

Query: "{query}"
JSON: {{"keywords": ["keyword1", "keyword2", ...], "date_from": "YYYY-MM-DD or null (REQUIRED: use this format, e.g. 2026-04-01)", "date_to": "YYYY-MM-DD or null (REQUIRED: use this format, e.g. 2026-04-30)", "time_from": "HH:MM or null (24-hour, e.g. 09:00)", "time_to": "HH:MM or null (24-hour, e.g. 17:00)"}}"""


def build_expand_prompt(query: str, now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return EXPAND_PROMPT_TEMPLATE.format(
        now=ts.strftime("%Y-%m-%d %H:%M"),
        weekday=ts.strftime("%A"),
        query=query,
    )
