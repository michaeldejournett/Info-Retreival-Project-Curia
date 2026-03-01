import { Router } from 'express'
import db from '../db.js'

const router = Router()

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8080'

const listEvents = db.prepare(`
  SELECT e.*,
    (SELECT COUNT(*) FROM groups g WHERE g.event_id = e.id) AS groupCount
  FROM events e
  ORDER BY e.date ASC
`)

const getEvent = db.prepare(`
  SELECT e.*,
    (SELECT COUNT(*) FROM groups g WHERE g.event_id = e.id) AS groupCount
  FROM events e
  WHERE e.id = ?
`)

const getEventByUrl = db.prepare(`
  SELECT e.*,
    (SELECT COUNT(*) FROM groups g WHERE g.event_id = e.id) AS groupCount
  FROM events e
  WHERE e.url = ?
`)

const FIELD_WEIGHTS = { name: 4, venue: 2, category: 2, description: 1, tags: 1 }

router.get('/', (_req, res) => {
  try {
    const rows = listEvents.all()
    res.json(rows.map(formatEvent))
  } catch (err) {
    console.error('GET /api/events', err)
    res.status(500).json({ error: 'Failed to fetch events' })
  }
})

router.get('/search', async (req, res) => {
  try {
    const { q, no_llm } = req.query
    if (!q || !q.trim()) return res.json({ terms: [], llmUsed: false, results: [] })

    const rawTerms = q.toLowerCase().match(/[a-z0-9]+/g)?.filter(w => w.length > 1) || []
    let terms = rawTerms
    let llmUsed = false
    let date_range = null
    let time_range = null

    // no_llm from client overrides env default; env default is false when API key is set
    const noLlm = no_llm === 'true' ? 'true' : 'false'

    let scored = []
    let fastapiError = null
    try {
      const resp = await fetch(
        `${FASTAPI_URL}/search?${new URLSearchParams({ q, top: '100', no_llm: noLlm })}`,
        { signal: AbortSignal.timeout(15000) }
      )
      if (resp.ok) {
        const data = await resp.json()
        if (data.terms?.length) {
          terms = data.terms
          if (noLlm === 'false') llmUsed = true
        }
        // Only apply date/time filters in AI mode — non-AI search is pure keyword matching
        if (noLlm === 'false') {
          if (data.date_range) date_range = data.date_range
          if (data.time_range) time_range = data.time_range
        }
        // Use FastAPI's results when available (warm cache / post-scrape pool)
        if (data.results?.length) {
          scored = data.results.map((r) => {
            const row = r.url ? getEventByUrl.get(r.url) : null
            if (row) return { ...formatEvent(row), score: r.score ?? 0 }
            return fastApiToEvent(r)
          })
        }
      } else {
        fastapiError = `HTTP ${resp.status}`
        console.warn('FastAPI search returned non-ok:', resp.status)
      }
    } catch (err) {
      fastapiError = err?.message || String(err)
      console.warn('FastAPI search unavailable — using raw terms:', fastapiError)
    }

    if (scored.length === 0) {
      const allRows = listEvents.all()
      const pureFilter = terms.length === 0 && (date_range || time_range)

    for (const row of allRows) {
      const ev = formatEvent(row)

      // Apply date filter from AI (inclusive range, multi-day overlap)
      if (date_range) {
        const evStart = toYmd(ev.date)
        const evEnd = toYmd(ev.endDate || ev.date)
        if (!evStart) continue
        if (evEnd < date_range.start) continue   // event ends before range
        if (evStart > date_range.end) continue   // event starts after range
      }

      // Apply time filter from AI (ev.time is "HH:MM" 24h)
      if (time_range) {
        const t = ev.time || ''
        if (time_range.start && t < time_range.start) continue
        if (time_range.end && t > time_range.end) continue
      }

      // Pure date/time query — include all events that passed the filter
      if (pureFilter) {
        scored.push({ ...ev, score: 0 })
        continue
      }

      let score = 0
      const fields = {
        name: (ev.name || '').toLowerCase(),
        description: (ev.description || '').toLowerCase(),
        venue: (ev.venue || '').toLowerCase(),
        category: (ev.category || '').toLowerCase(),
        tags: (ev.tags || []).join(' ').toLowerCase(),
      }

      for (let i = 0; i < terms.length; i++) {
        const posWeight = terms.length - i  // earlier terms worth more
        for (const [field, text] of Object.entries(fields)) {
          if (text.includes(terms[i])) score += (FIELD_WEIGHTS[field] || 1) * posWeight
        }
      }

      if (score > 0) scored.push({ ...ev, score })
    }
    }

    scored.sort((a, b) => b.score - a.score)

    res.json({ terms, llmUsed, count: scored.length, results: scored, date_range, time_range, _fastapiUrl: FASTAPI_URL, _fastapiError: fastapiError })
  } catch (err) {
    console.error('GET /api/events/search', err)
    res.status(500).json({ error: 'Search failed' })
  }
})

router.get('/:id', (req, res) => {
  try {
    const row = getEvent.get(req.params.id)
    if (!row) return res.status(404).json({ error: 'Event not found' })
    res.json(formatEvent(row))
  } catch (err) {
    console.error('GET /api/events/:id', err)
    res.status(500).json({ error: 'Failed to fetch event' })
  }
})

function toYmd(v) {
  if (!v) return ''
  const s = String(v)
  return s.length >= 10 ? s.slice(0, 10) : s
}

function fastApiToEvent(r) {
  const start = r.start || ''
  const date = start.length >= 10 ? start.slice(0, 10) : ''
  const time = start.length >= 16 ? start.slice(11, 16) : ''
  let id = 0
  for (let i = 0; i < (r.url || '').length; i++) id = ((id << 5) - id) + r.url.charCodeAt(i)
  return {
    id: id < 0 ? id : -id,
    name: r.title || 'Event',
    description: '',
    date,
    time,
    endDate: date,
    endTime: time,
    location: r.location || '',
    venue: r.location || '',
    price: 0,
    category: r.group || 'community',
    tags: [],
    groupCount: 0,
    url: r.url || null,
    imageUrl: r.image_url || null,
    score: r.score,
  }
}

function safeParseTags(raw) {
  try { return JSON.parse(raw || '[]') } catch { return [] }
}

function formatEvent(row) {
  return {
    id: row.id,
    name: row.name,
    description: row.description,
    date: row.date,
    time: row.time,
    endDate: row.end_date,
    endTime: row.end_time,
    location: row.location,
    venue: row.venue,
    price: row.price,
    category: row.category,
    tags: safeParseTags(row.tags),
    groupCount: row.groupCount,
    url: row.url || null,
    imageUrl: row.image_url || null,
  }
}

export default router
