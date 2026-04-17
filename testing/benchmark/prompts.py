from __future__ import annotations

from datetime import datetime


EXPAND_PROMPT_TEMPLATE = """Today is {now} ({weekday}).

You are helping search a university event database. Given a query, do three things:

1) KEYWORD EXPANSION
- Extract core topics and expand them with useful synonyms and specific examples.
- Keep terms lowercase and concise.
- Exclude generic stop words and purely temporal words.

2) DATE EXTRACTION
- If the query contains a date expression, return date_from/date_to in YYYY-MM-DD.
- Use broad ranges for phrases like "this week", "next week", "this month", "in april".
- If no date is mentioned, return null for both.

3) TIME EXTRACTION
- Map references to 24-hour HH:MM bounds.
- If no time phrase is mentioned, return null for both.

Return only a JSON object:
{{
  "keywords": ["..."],
  "date_from": "YYYY-MM-DD or null",
  "date_to": "YYYY-MM-DD or null",
  "time_from": "HH:MM or null",
  "time_to": "HH:MM or null"
}}

Query: "{query}"
"""


def build_expand_prompt(query: str) -> str:
    now = datetime.now()
    return EXPAND_PROMPT_TEMPLATE.format(
        now=now.strftime("%Y-%m-%d %H:%M"),
        weekday=now.strftime("%A"),
        query=query,
    )
