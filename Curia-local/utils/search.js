/**
 * Client-side event search for Curia on GitHub Pages.
 *
 * Two-tier approach:
 * 1. Static keyword generalization map (instant, same as backend keywords.js)
 * 2. LLM keyword expansion via @huggingface/transformers (LaMini-Flan-T5-248M)
 *    — same model used by the Character Creator
 *
 * Also handles date/time parsing from natural language queries.
 */

// ── Static keyword generalization map (ported from backend/keywords.js) ──────
const GENERALIZATIONS = [
  {
    terms: [
      'pizza',
      'burger',
      'taco',
      'sushi',
      'pasta',
      'barbecue',
      'bbq',
      'sandwich',
      'buffet',
      'potluck',
    ],
    add: ['food', 'dining'],
  },
  { terms: ['coffee', 'cafe', 'espresso', 'latte'], add: ['food', 'beverage', 'cafe'] },
  {
    terms: ['beer', 'brewery', 'brew', 'craft beer', 'ale', 'lager'],
    add: ['food', 'beverage', 'alcohol'],
  },
  {
    terms: ['wine', 'winery', 'tasting', 'vineyard', 'sommelier'],
    add: ['food', 'beverage', 'alcohol'],
  },
  {
    terms: ['bake', 'baking', 'cook', 'cooking', 'chef', 'culinary', 'recipe'],
    add: ['food', 'cooking'],
  },
  {
    terms: ['dinner', 'lunch', 'breakfast', 'brunch', 'meal', 'feast', 'banquet'],
    add: ['food', 'dining'],
  },
  {
    terms: ['biology', 'botany', 'zoology', 'microbiology', 'ecology'],
    add: ['science', 'stem', 'biology'],
  },
  {
    terms: ['chemistry', 'biochemistry', 'organic chemistry', 'chemical'],
    add: ['science', 'stem', 'chemistry'],
  },
  {
    terms: ['physics', 'quantum', 'thermodynamics', 'mechanics'],
    add: ['science', 'stem', 'physics'],
  },
  {
    terms: ['astronomy', 'astrophysics', 'telescope', 'planet', 'space', 'nasa', 'cosmos'],
    add: ['science', 'stem', 'space'],
  },
  {
    terms: ['geology', 'geoscience', 'earth science', 'mineralogy'],
    add: ['science', 'stem', 'geology'],
  },
  {
    terms: ['neuroscience', 'psychology', 'cognition', 'brain'],
    add: ['science', 'stem', 'health'],
  },
  {
    terms: ['genetics', 'dna', 'genome', 'gene', 'molecular'],
    add: ['science', 'stem', 'biology'],
  },
  {
    terms: ['statistics', 'probability', 'calculus', 'algebra', 'math', 'mathematics'],
    add: ['science', 'stem', 'math'],
  },
  {
    terms: [
      'python',
      'javascript',
      'typescript',
      'java',
      'rust',
      'golang',
      'c++',
      'swift',
      'kotlin',
    ],
    add: ['technology', 'coding', 'programming'],
  },
  {
    terms: ['machine learning', 'deep learning', 'neural network', 'nlp'],
    add: ['technology', 'ai', 'stem'],
  },
  {
    terms: ['artificial intelligence', 'ai ', ' ai,', 'llm', 'gpt', 'chatbot'],
    add: ['technology', 'ai', 'stem'],
  },
  {
    terms: ['robotics', 'robot', 'drone', 'automation'],
    add: ['technology', 'engineering', 'stem'],
  },
  {
    terms: ['cybersecurity', 'hacking', 'ctf', 'security', 'pentest'],
    add: ['technology', 'security'],
  },
  {
    terms: ['data science', 'data analytics', 'big data', 'visualization', 'tableau', 'pandas'],
    add: ['technology', 'stem', 'data'],
  },
  {
    terms: ['hackathon', 'hack', 'coding challenge', 'competition'],
    add: ['technology', 'coding'],
  },
  {
    terms: ['web development', 'frontend', 'backend', 'full stack', 'react', 'vue', 'angular'],
    add: ['technology', 'coding', 'web'],
  },
  {
    terms: ['cloud', 'aws', 'azure', 'gcp', 'devops', 'kubernetes', 'docker'],
    add: ['technology', 'cloud', 'engineering'],
  },
  { terms: ['app', 'mobile', 'ios', 'android'], add: ['technology', 'mobile'] },
  {
    terms: ['electrical', 'circuit', 'electronics', 'semiconductor'],
    add: ['engineering', 'stem'],
  },
  { terms: ['mechanical', 'cad', 'solidworks', '3d printing'], add: ['engineering', 'stem'] },
  { terms: ['civil engineering', 'structural', 'construction'], add: ['engineering', 'stem'] },
  { terms: ['basketball', 'nba', 'dribble', 'dunk', 'hoop'], add: ['sports', 'athletics'] },
  { terms: ['soccer', 'futbol', 'penalty', 'goal kick'], add: ['sports', 'athletics'] },
  { terms: ['american football', 'nfl', 'touchdown', 'husker'], add: ['sports', 'athletics'] },
  { terms: ['baseball', 'softball', 'pitcher', 'homerun'], add: ['sports', 'athletics'] },
  { terms: ['volleyball', 'spike', 'serve'], add: ['sports', 'athletics'] },
  { terms: ['swimming', 'swim', 'lap pool', 'aquatic'], add: ['sports', 'fitness', 'athletics'] },
  {
    terms: ['running', 'marathon', '5k', '10k', 'cross country', 'track'],
    add: ['sports', 'fitness', 'athletics'],
  },
  { terms: ['cycling', 'bike', 'bicycle', 'triathlon'], add: ['sports', 'fitness'] },
  { terms: ['tennis', 'racket', 'court'], add: ['sports', 'athletics'] },
  { terms: ['golf', 'putt', 'fairway', 'tee'], add: ['sports', 'athletics'] },
  {
    terms: ['wrestling', 'boxing', 'martial arts', 'judo', 'mma', 'kickboxing'],
    add: ['sports', 'athletics'],
  },
  { terms: ['yoga', 'pilates', 'stretch'], add: ['fitness', 'wellness', 'health'] },
  { terms: ['gym', 'weightlifting', 'strength training', 'crossfit'], add: ['fitness', 'health'] },
  { terms: ['rock climbing', 'bouldering', 'climbing'], add: ['sports', 'fitness', 'outdoor'] },
  {
    terms: ['hiking', 'trail', 'backpacking', 'camping'],
    add: ['sports', 'fitness', 'outdoor', 'nature'],
  },
  {
    terms: ['painting', 'watercolor', 'acrylic', 'oil paint', 'canvas'],
    add: ['art', 'visual arts', 'creative'],
  },
  {
    terms: ['drawing', 'illustration', 'sketch', 'comic'],
    add: ['art', 'visual arts', 'creative'],
  },
  { terms: ['sculpture', 'ceramics', 'pottery', 'clay'], add: ['art', 'visual arts', 'creative'] },
  { terms: ['photography', 'photo', 'camera', 'portrait'], add: ['art', 'creative'] },
  {
    terms: ['film', 'cinema', 'movie', 'screening', 'documentary'],
    add: ['art', 'entertainment', 'film'],
  },
  {
    terms: ['theater', 'theatre', 'play', 'musical', 'broadway', 'improv', 'drama'],
    add: ['art', 'performance', 'entertainment'],
  },
  {
    terms: ['dance', 'ballet', 'hip hop dance', 'salsa', 'ballroom'],
    add: ['art', 'performance', 'dance'],
  },
  {
    terms: ['concert', 'live music', 'gig', 'band', 'show'],
    add: ['music', 'performance', 'entertainment'],
  },
  { terms: ['jazz', 'bebop', 'blues'], add: ['music', 'performance', 'art'] },
  {
    terms: ['opera', 'symphony', 'orchestra', 'classical music', 'choir', 'choral'],
    add: ['music', 'performance', 'art'],
  },
  {
    terms: ['rap', 'hip hop', 'r&b', 'soul music'],
    add: ['music', 'performance', 'entertainment'],
  },
  { terms: ['comedy', 'stand up', 'open mic', 'improv'], add: ['entertainment', 'comedy'] },
  {
    terms: ['gaming', 'esports', 'video game', 'tabletop', 'board game'],
    add: ['entertainment', 'gaming'],
  },
  {
    terms: ['poetry', 'spoken word', 'literary', 'writing'],
    add: ['art', 'creative', 'literature'],
  },
  { terms: ['book club', 'reading', 'author'], add: ['education', 'literature'] },
  {
    terms: ['lecture', 'seminar', 'colloquium', 'talk'],
    add: ['academic', 'education', 'learning'],
  },
  {
    terms: ['workshop', 'training', 'tutorial', 'bootcamp'],
    add: ['education', 'learning', 'skills'],
  },
  { terms: ['conference', 'symposium', 'summit'], add: ['academic', 'professional', 'networking'] },
  {
    terms: ['research', 'study', 'experiment', 'lab', 'thesis', 'dissertation'],
    add: ['academic', 'stem', 'research'],
  },
  { terms: ['internship', 'intern', 'co-op'], add: ['career', 'professional', 'job'] },
  {
    terms: ['networking event', 'career fair', 'job fair', 'recruiter', 'employer'],
    add: ['career', 'professional', 'networking'],
  },
  { terms: ['resume', 'cv', 'job search', 'interview'], add: ['career', 'professional'] },
  {
    terms: ['startup', 'entrepreneur', 'pitch', 'venture', 'founder'],
    add: ['career', 'business', 'entrepreneurship'],
  },
  {
    terms: ['business', 'finance', 'accounting', 'economics', 'marketing'],
    add: ['business', 'professional'],
  },
  { terms: ['leadership', 'management', 'executive'], add: ['professional', 'career'] },
  {
    terms: ['meditation', 'mindfulness', 'breathing', 'guided'],
    add: ['wellness', 'mindfulness', 'health'],
  },
  {
    terms: ['mental health', 'anxiety', 'stress', 'depression', 'therapy', 'counseling'],
    add: ['wellness', 'health', 'mental health'],
  },
  {
    terms: ['nutrition', 'diet', 'healthy eating', 'vegan', 'vegetarian'],
    add: ['health', 'food', 'wellness'],
  },
  { terms: ['first aid', 'cpr', 'medical', 'nursing', 'healthcare'], add: ['health', 'medical'] },
  {
    terms: ['volunteer', 'volunteering', 'community service', 'giving back'],
    add: ['community', 'service', 'volunteer'],
  },
  {
    terms: ['charity', 'nonprofit', 'donation', 'fundraiser', 'fundraising'],
    add: ['community', 'nonprofit'],
  },
  {
    terms: ['sustainability', 'environment', 'climate', 'green', 'eco'],
    add: ['environment', 'community', 'sustainability'],
  },
  {
    terms: ['diversity', 'equity', 'inclusion', 'dei', 'multicultural'],
    add: ['community', 'social', 'diversity'],
  },
  {
    terms: ['garden', 'gardening', 'planting', 'nature', 'park'],
    add: ['community', 'outdoor', 'nature'],
  },
  {
    terms: ['religion', 'faith', 'spiritual', 'church', 'mosque', 'temple', 'prayer'],
    add: ['community', 'spiritual'],
  },
  {
    terms: ['international', 'culture', 'cultural', 'heritage', 'global'],
    add: ['community', 'culture', 'international'],
  },
  { terms: ['greek life', 'fraternity', 'sorority'], add: ['community', 'social', 'student life'] },
  {
    terms: ['student org', 'club', 'student government', 'association'],
    add: ['community', 'student life'],
  },
];

const STOP_WORDS = new Set([
  'a',
  'an',
  'the',
  'and',
  'or',
  'but',
  'in',
  'on',
  'at',
  'to',
  'for',
  'of',
  'with',
  'by',
  'is',
  'are',
  'was',
  'were',
  'be',
  'been',
  'have',
  'has',
  'had',
  'do',
  'does',
  'did',
  'will',
  'would',
  'could',
  'should',
  'may',
  'might',
  'shall',
  'can',
  'need',
  'i',
  'me',
  'my',
  'we',
  'our',
  'you',
  'your',
  'it',
  'its',
  'they',
  'them',
  'their',
  'this',
  'that',
  'these',
  'those',
  'what',
  'which',
  'who',
  'when',
  'where',
  'how',
  'find',
  'show',
  'me',
  'events',
  'event',
  'something',
  'looking',
  'for',
  'want',
  'unl',
  'nebraska',
  'lincoln',
  'campus',
  // Common LLM noise words for this model
  'date',
  'time',
  'next',
  'month',
  'year',
  'day',
  'week',
  'location',
  'venue',
  'type',
  'style',
  'language',
  'description',
  'trend',
  'availability',
  'taste',
  'price',
  'menu',
  'dish',
  'specialty',
  'recipe',
  'cuisine',
  'restaurant',
  'other',
  'more',
  'related',
  'keyword',
  'keywords',
  'synonym',
  'synonyms',
  'topic',
  'topics',
  'about',
  'like',
  'such',
  'also',
  'new',
  'any',
  'all',
  'get',
  'use',
  'used',
]);

// ── Static keyword expansion ──────────────────────────────────────────────────
export function expandStaticTerms(query) {
  const lower = query.toLowerCase();
  const expanded = new Set();

  // Add base terms (non-stop words)
  for (const word of lower.split(/\s+/)) {
    if (word.length > 2 && !STOP_WORDS.has(word)) expanded.add(word);
  }

  // Run through generalizations
  for (const { terms, add } of GENERALIZATIONS) {
    if (terms.some((t) => lower.includes(t))) {
      for (const kw of add) expanded.add(kw);
    }
  }

  return [...expanded];
}

// ── LLM keyword expansion (lazy-loaded, same pattern as Character Creator) ────
let llmPipeline = null;
let llmLoading = false;

async function getLlmPipeline() {
  if (llmPipeline) return llmPipeline;
  if (llmLoading) {
    // Wait for existing load
    while (llmLoading) await new Promise((r) => setTimeout(r, 200));
    return llmPipeline;
  }

  llmLoading = true;
  try {
    const { pipeline, env } = await import('@huggingface/transformers');
    env.backends.onnx.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/';
    llmPipeline = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-248M', {
      quantized: true,
    });
    return llmPipeline;
  } finally {
    llmLoading = false;
  }
}

// Regex that matches common date/time phrases in a query
const DATE_PHRASE_RE =
  /\b(today|tomorrow|this\s+week(?:end)?|next\s+week|this\s+month|next\s+month|in\s+\d+\s+(?:day|week|month)s?|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?)\b/gi;

function extractDatePhrase(query) {
  const m = query.match(DATE_PHRASE_RE);
  return m ? m[0].toLowerCase().trim() : null;
}

function stripDatePhrase(query) {
  return query.replace(DATE_PHRASE_RE, '').replace(/\s+/g, ' ').trim();
}

async function expandWithLlm(query) {
  try {
    const pipe = await getLlmPipeline();
    const topic = stripDatePhrase(query) || query;
    const prompt = `List comma-separated search synonyms for events about: ${topic}`;
    const [result] = await pipe(prompt, {
      max_new_tokens: 60,
      repetition_penalty: 1.5,
      no_repeat_ngram_size: 3,
    });
    const text = result.generated_text ?? '';
    return text
      .split(/[,;|\n]+/)
      .map((w) =>
        w
          .trim()
          .replace(/^\d+[.)]\s*/, '')
          .toLowerCase()
      )
      .flatMap((w) => w.split(/\s+/))
      .filter((w) => w.length > 2 && !/^\d/.test(w) && !STOP_WORDS.has(w));
  } catch {
    return [];
  }
}

async function parseDateWithLlm(query) {
  const phrase = extractDatePhrase(query);
  if (!phrase) return null;
  try {
    const pipe = await getLlmPipeline();
    const today = new Date().toISOString().slice(0, 10);
    const prompt = `Today is ${today}. Convert ${phrase} to a date range in YYYY-MM-DD to YYYY-MM-DD format.`;
    const [result] = await pipe(prompt, {
      max_new_tokens: 25,
      repetition_penalty: 1.3,
      no_repeat_ngram_size: 3,
    });
    const text = (result.generated_text ?? '').trim();
    console.log('[parseDateWithLlm] phrase:', phrase, '| raw output:', text);
    const match = text.match(/(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})/);
    if (match) {
      console.log('[parseDateWithLlm] range match:', match[1], '→', match[2]);
      return { start: match[1], end: match[2] };
    }
    const single = text.match(/(\d{4}-\d{2}-\d{2})/);
    if (single) {
      console.log('[parseDateWithLlm] single date match:', single[1]);
      return { start: single[1], end: single[1] };
    }
    console.log('[parseDateWithLlm] no date found in output');
    return null;
  } catch {
    return null;
  }
}

// ── Date/time parsing from natural language ───────────────────────────────────
export function parseDateRange(query) {
  const lower = query.toLowerCase();
  const now = new Date();
  const dow = now.getDay(); // 0=Sun

  const ymd = (d) => new Date(d).toISOString().slice(0, 10);
  const shift = (n) => {
    const d = new Date(now);
    d.setDate(d.getDate() + n);
    return d;
  };

  // Helpers that return {start,end} given a calendar unit + signed offset
  // (offset: 0=current, 1=next, -1=last)
  const unitRange = {
    day: (offset) => {
      const d = ymd(shift(offset));
      return { start: d, end: d };
    },
    week: (offset) => {
      // Week = Mon–Sun; find Monday of target week
      const daysToMon = dow === 0 ? -6 : -(dow - 1);
      const mon = shift(daysToMon + offset * 7);
      return { start: ymd(mon), end: ymd(shift(daysToMon + offset * 7 + 6)) };
    },
    weekend: (offset) => {
      // Saturday of target week (positive = upcoming, negative = past)
      const daysToSat = (6 - dow + 7) % 7 || 7;
      const sat = shift(daysToSat + offset * 7);
      return { start: ymd(sat), end: ymd(shift(daysToSat + offset * 7 + 1)) };
    },
    month: (offset) => {
      const raw = now.getMonth() + offset;
      const y = now.getFullYear() + Math.floor(raw / 12);
      const m = ((raw % 12) + 12) % 12;
      return { start: ymd(new Date(y, m, 1)), end: ymd(new Date(y, m + 1, 0)) };
    },
    year: (offset) => {
      const y = now.getFullYear() + offset;
      return { start: `${y}-01-01`, end: `${y}-12-31` };
    },
  };

  // ── Aliases ───────────────────────────────────────────────────────────────
  if (/\b(today|tonight)\b/.test(lower)) return unitRange.day(0);
  if (/\btomorrow\b/.test(lower)) return unitRange.day(1);
  if (/\byesterday\b/.test(lower)) return unitRange.day(-1);

  // ── (this|current|last|next) (day|week|weekend|month|year) ───────────────
  const modUnitRe = /\b(this|current|last|next)\s+(day|week|weekend|month|year)\b/;
  const mu = lower.match(modUnitRe);
  if (mu) {
    const offset = { this: 0, current: 0, last: -1, next: 1 }[mu[1]];
    return unitRange[mu[2]](offset);
  }

  // ── "in X days/weeks/months" ──────────────────────────────────────────────
  const inRe = /\bin\s+(\d+)\s+(day|week|month)s?\b/;
  const inM = lower.match(inRe);
  if (inM) {
    const n = parseInt(inM[1], 10);
    if (inM[2] === 'day') return unitRange.day(n);
    if (inM[2] === 'week') return unitRange.week(Math.round(n));
    if (inM[2] === 'month') return unitRange.month(n);
  }

  // ── "next <weekday>" ──────────────────────────────────────────────────────
  const weekdays = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  const ndRe = /\bnext\s+(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b/;
  const nd = lower.match(ndRe);
  if (nd) {
    const target = weekdays.indexOf(nd[1]);
    const diff = (target - dow + 7) % 7 || 7;
    const d = ymd(shift(diff));
    return { start: d, end: d };
  }

  // ── Named months ("march events", "events in april") ─────────────────────
  const months = [
    'january',
    'february',
    'march',
    'april',
    'may',
    'june',
    'july',
    'august',
    'september',
    'october',
    'november',
    'december',
  ];
  for (let i = 0; i < months.length; i++) {
    if (lower.includes(months[i])) return unitRange.month(i - now.getMonth());
  }

  return null;
}

export function parseTimeRange(query) {
  const lower = query.toLowerCase();
  if (/\bmorning\b/.test(lower)) return { start: '06:00', end: '12:00' };
  if (/\bnoon\b/.test(lower)) return { start: '11:00', end: '13:00' };
  if (/\bafternoon\b/.test(lower)) return { start: '12:00', end: '17:00' };
  if (/\bevening\b/.test(lower)) return { start: '17:00', end: '21:00' };
  if (/\bnight\b/.test(lower)) return { start: '19:00', end: '23:59' };
  return null;
}

// ── Scoring (ported from api/search.py) ──────────────────────────────────────
const FIELD_WEIGHTS = { name: 4, group: 3, description: 2, location: 1 };

function scoreEvent(event, terms) {
  if (!terms.length) return 0;
  const fields = {
    name: (event.name ?? '').toLowerCase(),
    group: (event.group ?? '').toLowerCase(),
    description: (event.description ?? '').toLowerCase(),
    location: (event.venue ?? '').toLowerCase(),
  };

  let score = 0;
  for (const term of terms) {
    for (const [field, weight] of Object.entries(FIELD_WEIGHTS)) {
      if (fields[field].includes(term)) score += weight;
    }
  }
  return score;
}

// ── Public search API ─────────────────────────────────────────────────────────

/**
 * Fast keyword-only search (no LLM). Returns ranked results instantly.
 */
export function searchKeyword(events, query) {
  if (!query.trim()) return { results: events, terms: [], llmUsed: false };

  const terms = expandStaticTerms(query);
  const dateRange = parseDateRange(query);
  const timeRange = parseTimeRange(query);

  const scored = events
    .map((ev) => ({ ev, score: scoreEvent(ev, terms) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .map(({ ev }) => ev);

  return {
    results: scored.length ? scored : [],
    terms,
    llmUsed: false,
    date_range: dateRange,
    time_range: timeRange,
    count: scored.length,
  };
}

/**
 * AI-assisted search with LLM keyword expansion.
 * Falls back to static keyword search if LLM fails.
 */
export async function searchAi(events, query) {
  if (!query.trim()) return { results: events, terms: [], llmUsed: false };

  const staticTerms = expandStaticTerms(query);
  const timeRange = parseTimeRange(query);

  const llmTerms = await expandWithLlm(query).catch(() => []);

  const dateRange = parseDateRange(query);
  const llmUsed = llmTerms.length > 0;

  const allTerms = [...new Set([...staticTerms, ...llmTerms])];

  const scored = events
    .map((ev) => ({ ev, score: scoreEvent(ev, allTerms) }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .map(({ ev }) => ev);

  return {
    results: scored.length ? scored : [],
    terms: allTerms,
    llmUsed,
    date_range: dateRange,
    time_range: timeRange,
    count: scored.length,
  };
}
