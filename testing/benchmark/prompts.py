from __future__ import annotations

from datetime import datetime

from api.prompt_templates import build_expand_prompt as build_system_expand_prompt

# Compact prompt for small decoder-only models (<2B params).
# Uses a concrete filled-in example so the model doesn't pattern-match on nulls,
# then ends with "JSON:" to prime the model to generate the object directly.
COMPACT_PROMPT_TEMPLATE = """Today is {now} ({weekday}).

Extract date/time and keywords from the query. Return ONLY a JSON object with fields in this exact order: date_from, date_to, time_from, time_to, keywords.

- date_from / date_to: copy any YYYY-MM-DD date exactly; "today"/"tonight"={today}; "this weekend"={example_sat}; null if no date
- time_from / time_to: 24h HH:MM
  AFTER HH:MM  → time_from=HH:MM, time_to=null
  BEFORE HH:MM → time_from=null,  time_to=HH:MM
  "morning"="06:00", "afternoon"="12:00", "evening"/"tonight"="17:00"; null if no time
- keywords: 5-10 lowercase synonyms and related terms

Example A (before → time_to):
Query: "open mic on {example_sat} before 20:00"
JSON: {{"date_from": "{example_sat}", "date_to": "{example_sat}", "time_from": null, "time_to": "20:00", "keywords": ["open mic", "performance", "music", "acoustic", "singer"]}}

Example B (after → time_from):
Query: "jazz concerts on {example_sat} after 17:00"
JSON: {{"date_from": "{example_sat}", "date_to": "{example_sat}", "time_from": "17:00", "time_to": null, "keywords": ["jazz", "concert", "music", "band", "live"]}}

Example C (no time):
Query: "volunteer opportunities near campus"
JSON: {{"date_from": null, "date_to": null, "time_from": null, "time_to": null, "keywords": ["volunteer", "service", "community", "outreach", "campus"]}}

Query: "{query}"
JSON:"""

# Fill-in prompt for encoder-decoder models (flan-t5, LaMini).
# Uses labeled key: value format matching flan-t5's training distribution;
# the labeled fallback parser extracts fields even when JSON is malformed.
SEQ2SEQ_PROMPT_TEMPLATE = (
    'Extract event search fields from the query. '
    'Query: "{query}" '
    'Answer with: keywords: <terms>, date_from: <YYYY-MM-DD or null>, date_to: <YYYY-MM-DD or null>, '
    'time_from: <HH:MM or null>, time_to: <HH:MM or null>'
)


def build_expand_prompt(query: str) -> str:
    return build_system_expand_prompt(query=query)


def build_compact_prompt(query: str, now: datetime | None = None) -> str:
    from datetime import timedelta
    ts = now or datetime.now()
    days_to_sat = (5 - ts.weekday()) % 7 or 7
    example_sat = (ts + timedelta(days=days_to_sat)).strftime("%Y-%m-%d")
    today = ts.strftime("%Y-%m-%d")
    return COMPACT_PROMPT_TEMPLATE.format(
        now=ts.strftime("%Y-%m-%d %H:%M"),
        weekday=ts.strftime("%A"),
        today=today,
        example_sat=example_sat,
        query=query,
    )


def build_seq2seq_prompt(query: str, now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return SEQ2SEQ_PROMPT_TEMPLATE.format(
        now=ts.strftime("%Y-%m-%d %H:%M"),
        query=query,
    )
