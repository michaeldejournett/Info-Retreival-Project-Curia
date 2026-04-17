/**
 * Generates and triggers download of an .ics (iCalendar) file for an event.
 */

function pad(n) {
  return String(n).padStart(2, '0');
}

function toIcsDate(dateStr, timeStr) {
  // dateStr: "2026-03-20", timeStr: "09:00"
  const [year, month, day] = dateStr.split('-');
  const [hour, minute] = timeStr.split(':');
  return `${year}${pad(month)}${pad(day)}T${pad(hour)}${pad(minute)}00`;
}

function escapeIcs(str) {
  return String(str)
    .replace(/\\/g, '\\\\')
    .replace(/;/g, '\\;')
    .replace(/,/g, '\\,')
    .replace(/\n/g, '\\n');
}

function foldLine(line) {
  // ICS spec: lines > 75 chars should be folded
  const MAX = 75;
  if (line.length <= MAX) return line;
  let result = '';
  while (line.length > MAX) {
    result += line.slice(0, MAX) + '\r\n ';
    line = line.slice(MAX);
  }
  return result + line;
}

export function generateIcs(event) {
  const now = new Date();
  const stamp =
    `${now.getUTCFullYear()}${pad(now.getUTCMonth() + 1)}${pad(now.getUTCDate())}` +
    `T${pad(now.getUTCHours())}${pad(now.getUTCMinutes())}${pad(now.getUTCSeconds())}Z`;

  const dtStart = toIcsDate(event.date || '', event.time || '00:00');
  const dtEnd = toIcsDate(
    event.endDate || event.date || '',
    event.endTime || event.time || '00:00'
  );

  const lines = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//Curia//Event Finder//EN',
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH',
    'BEGIN:VEVENT',
    foldLine(`UID:event-${event.id}-${stamp}@curia.events`),
    `DTSTAMP:${stamp}`,
    `DTSTART:${dtStart}`,
    `DTEND:${dtEnd}`,
    foldLine(`SUMMARY:${escapeIcs(event.name)}`),
    foldLine(`DESCRIPTION:${escapeIcs(event.description)}`),
    foldLine(`LOCATION:${escapeIcs((event.venue || '') + ', ' + (event.location || ''))}`),
    event.price === 0 ? 'X-COST:Free' : foldLine(`X-COST:$${event.price}`),
    'END:VEVENT',
    'END:VCALENDAR',
  ];

  const content = lines.join('\r\n');
  const blob = new Blob([content], { type: 'text/calendar;charset=utf-8' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = `${event.name.replace(/[^a-z0-9]/gi, '_')}.ics`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
