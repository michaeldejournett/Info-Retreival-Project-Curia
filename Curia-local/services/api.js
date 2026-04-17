/**
 * Static data layer for Curia on GitHub Pages.
 * Loads pre-scraped UNL events from /curia/events.json and transforms
 * them from the scraper format into the frontend event format.
 */

// Keyword → category mapping for inference
const CATEGORY_SIGNALS = [
  {
    keywords: [
      'technology',
      'coding',
      'programming',
      'software',
      'computer',
      'hackathon',
      'ai ',
      'artificial intelligence',
      'machine learning',
      'data science',
      'engineering',
      'stem',
      'robot',
      'cyber',
      'web',
      'cloud',
      'mobile',
      'app',
    ],
    category: 'technology',
  },
  {
    keywords: [
      'music',
      'concert',
      'jazz',
      'band',
      'orchestra',
      'choir',
      'symphony',
      'opera',
      'recital',
      'performance',
      'live music',
      'rap',
      'hip hop',
      'r&b',
    ],
    category: 'music',
  },
  {
    keywords: [
      'sport',
      'athletic',
      'game',
      'basketball',
      'football',
      'soccer',
      'volleyball',
      'baseball',
      'softball',
      'swim',
      'running',
      'track',
      'golf',
      'tennis',
      'wrestling',
      'husker',
      'rec center',
      'fitness',
    ],
    category: 'sports',
  },
  {
    keywords: [
      'food',
      'dinner',
      'lunch',
      'breakfast',
      'brunch',
      'meal',
      'pizza',
      'cook',
      'culinary',
      'beverage',
      'beer',
      'wine',
      'dining',
      'taste',
      'baking',
      'potluck',
    ],
    category: 'food',
  },
  {
    keywords: [
      'art',
      'gallery',
      'exhibit',
      'museum',
      'theater',
      'theatre',
      'film',
      'movie',
      'cinema',
      'dance',
      'painting',
      'sculpture',
      'photography',
      'comedy',
      'poetry',
      'literary',
      'drama',
      'ross movie',
      'visual arts',
    ],
    category: 'arts',
  },
  {
    keywords: [
      'community',
      'volunteer',
      'service',
      'charity',
      'nonprofit',
      'diversity',
      'inclusion',
      'culture',
      'cultural',
      'multicultural',
      'international',
      'greek',
      'student org',
      'club',
      'sustainability',
      'environment',
      'garden',
    ],
    category: 'community',
  },
  {
    keywords: [
      'health',
      'wellness',
      'yoga',
      'meditation',
      'mental health',
      'fitness',
      'nutrition',
      'medical',
      'nursing',
      'healthcare',
      'mindfulness',
      'gym',
    ],
    category: 'health',
  },
  {
    keywords: [
      'lecture',
      'seminar',
      'workshop',
      'conference',
      'academic',
      'research',
      'education',
      'career',
      'networking',
      'internship',
      'job fair',
      'resume',
      'startup',
      'business',
      'leadership',
      'colloquium',
      'symposium',
      'dissertation',
    ],
    category: 'education',
  },
];

function inferCategory(event) {
  const text = `${event.title} ${event.description ?? ''} ${event.group ?? ''}`.toLowerCase();

  // Ross Movie events are always arts
  if (event.group === 'Ross Movie') return 'arts';

  let bestCategory = null;
  let bestScore = 0;

  for (const { keywords, category } of CATEGORY_SIGNALS) {
    const score = keywords.reduce((sum, kw) => sum + (text.includes(kw) ? 1 : 0), 0);
    if (score > bestScore) {
      bestScore = score;
      bestCategory = category;
    }
  }

  return bestCategory ?? 'community';
}

function parseIsoDate(iso) {
  if (!iso) return { date: null, time: null };
  // e.g. "2026-03-01T12:00:00-06:00"
  const [datePart, timePart] = iso.split('T');
  const time = timePart ? timePart.slice(0, 5) : null; // "HH:MM"
  return { date: datePart, time };
}

let cachedEvents = null;

export async function loadEvents() {
  if (cachedEvents) return cachedEvents;

  const res = await fetch('/curia/events.json');
  if (!res.ok) throw new Error(`Failed to load events: ${res.status}`);
  const data = await res.json();

  // Scraper format → frontend format
  cachedEvents = data.events.map((ev, idx) => {
    const { date: startDate, time: startTime } = parseIsoDate(ev.start);
    const { date: endDate, time: endTime } = parseIsoDate(ev.end);
    const category = inferCategory(ev);

    // Venue is the full location string from scraper; extract the meaningful name
    // Format: "Building Name-Room Name Room:Room Number" or just "Building Name"
    const venueFull = ev.location ?? '';
    const venue = venueFull.split(' Room:')[0].split('-').slice(0, 2).join(' - ') || venueFull;

    return {
      id: idx,
      name: ev.title,
      description: ev.description ?? '',
      date: startDate,
      time: startTime,
      endDate: endDate,
      endTime: endTime,
      venue: venue || 'UNL Campus',
      location: 'Lincoln, NE',
      category,
      tags: [],
      imageUrl: ev.image_url ?? null,
      url: ev.url ?? null,
      price: null,
      groupCount: 0,
      source: ev.source,
      audience: ev.audience ?? [],
    };
  });

  return cachedEvents;
}
