from __future__ import annotations

from datetime import datetime

from api.prompt_templates import build_expand_prompt as build_system_expand_prompt

_TEMPORAL_PROMPT_PREFIX = """\
You extract date and time ranges from event search queries.
Given today's date and a query, first write ONE line starting with 'Reasoning: ' explaining the date math, then output ONE JSON object and nothing else.
Schema: {"date_from":"YYYY-MM-DD"|null,"date_to":"YYYY-MM-DD"|null,"time_from":"HH:MM"|null,"time_to":"HH:MM"|null}
Use null when the query does not imply that field. Weekend=Sat+Sun. Tonight=today,17:00-21:00. Morning=06:00-12:00. Afternoon=12:00-17:00. Evening=17:00-21:00.

Today: 2026-04-17 (Friday)
Query: sports games tonight
Reasoning: Tonight means today 2026-04-17, evening 17:00-21:00.
{"date_from":"2026-04-17","date_to":"2026-04-17","time_from":"17:00","time_to":"21:00"}

Today: 2026-04-13 (Monday)
Query: this weekend concerts
Reasoning: Next weekend from Monday 2026-04-13 is Sat 2026-04-18 and Sun 2026-04-19.
{"date_from":"2026-04-18","date_to":"2026-04-19","time_from":null,"time_to":null}

Today: 2026-04-17 (Friday)
Query: art exhibits next month
Reasoning: Next month after April 2026 is May 2026, full month.
{"date_from":"2026-05-01","date_to":"2026-05-31","time_from":null,"time_to":null}

Today: 2026-04-17 (Friday)
Query: events two weeks from now
Reasoning: Two weeks after 2026-04-17 is 2026-05-01.
{"date_from":"2026-05-01","date_to":"2026-05-01","time_from":null,"time_to":null}

"""

_KEYWORDS_PROMPT_PREFIX = """\
You expand search queries into related keywords for a university event database.
Respond with ONE JSON array of 20-30 lowercase strings and nothing else.
Include the original topic, synonyms, specific examples, subcategories, and related roles.
Do not include time words (tonight, morning, weekend) or generic stop words.

Query: pizza party
["food","pizza","pasta","italian","spaghetti","lasagna","calzone","dining","buffet","snacks","potluck","dinner","lunch","meal","restaurant","cuisine","eating","social","gathering","party","celebration","refreshments","free food","catering"]

Query: data science talks
["data science","machine learning","ai","analytics","statistics","python","r","visualization","deep learning","neural network","nlp","big data","research","seminar","lecture","workshop","stem","technology","algorithms","modeling","prediction","conference","presentation","speaker","career","internship"]

Query: weekend concerts
["music","concert","band","jazz","rock","pop","indie","folk","blues","classical","orchestra","choir","live music","performance","show","gig","recital","singer","musician","acoustic","electronic","hip hop","r&b","open mic","venue","entertainment"]

"""

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


def build_temporal_prompt(query: str, now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return f"{_TEMPORAL_PROMPT_PREFIX}Today: {ts.strftime('%Y-%m-%d')} ({ts.strftime('%A')})\nQuery: {query}"


def build_keywords_prompt(query: str) -> str:
    return f"{_KEYWORDS_PROMPT_PREFIX}Query: {query}"


def build_seq2seq_prompt(query: str, now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return SEQ2SEQ_PROMPT_TEMPLATE.format(
        now=ts.strftime("%Y-%m-%d %H:%M"),
        query=query,
    )
