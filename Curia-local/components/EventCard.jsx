import { useEffect, useRef, useState } from 'react';
import { getCategoryMeta } from '../data/events';

function formatDate(dateStr) {
  if (!dateStr) return '';
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    timeZone: 'America/Chicago',
  });
}

function formatTime(timeStr) {
  if (!timeStr) return '';
  const [h, m] = timeStr.split(':').map(Number);
  const ampm = h >= 12 ? 'PM' : 'AM';
  const hour = h % 12 || 12;
  return `${hour}:${String(m).padStart(2, '0')} ${ampm} CT`;
}

export default function EventCard({ event, groupCount, onClick }) {
  const cat = getCategoryMeta(event.category);
  const nameRef = useRef(null);
  const [truncated, setTruncated] = useState(false);

  useEffect(() => {
    const el = nameRef.current;
    if (el) setTruncated(el.scrollHeight > el.clientHeight);
  }, [event.name]);

  return (
    <article
      className="event-card"
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
      title={truncated ? event.name : undefined}
    >
      {/* Category image or color strip */}
      {event.imageUrl ? (
        <img className="event-card-image" src={event.imageUrl} alt={event.name} />
      ) : (
        <div className="event-card-banner" style={{ background: cat.color }} />
      )}

      <div className="event-card-body">
        <div className="event-card-category">
          <span style={{ color: cat.color }}>{cat.label}</span>
        </div>

        <h2 className="event-card-name" ref={nameRef}>
          {event.name}
        </h2>

        <div className="event-card-meta">
          <div className="event-card-meta-item">
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
              <line x1="16" y1="2" x2="16" y2="6" />
              <line x1="8" y1="2" x2="8" y2="6" />
              <line x1="3" y1="10" x2="21" y2="10" />
            </svg>
            <span>
              {formatDate(event.date)} · {formatTime(event.time)}
            </span>
          </div>

          <div className="event-card-meta-item">
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
              <circle cx="12" cy="10" r="3" />
            </svg>
            <span>{event.venue}</span>
          </div>
        </div>

        <div className="event-card-footer">
          <span className="event-groups-badge">
            <svg
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
            {groupCount} {groupCount === 1 ? 'group' : 'groups'}
          </span>
        </div>
      </div>
    </article>
  );
}
