import { useState, useEffect } from 'react';

const MONTHS = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];
const DAYS = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];

function toStr(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function formatDisplay(str) {
  if (!str) return null;
  const [y, m, d] = str.split('-').map(Number);
  return `${MONTHS[m - 1].slice(0, 3)} ${d}, ${y}`;
}

export default function DateRangePicker({ dateFrom, dateTo, onChange }) {
  const today = new Date();
  const [view, setView] = useState(() => {
    if (dateFrom) {
      const [y, mo] = dateFrom.split('-').map(Number);
      return new Date(y, mo - 1, 1);
    }
    return new Date(today.getFullYear(), today.getMonth(), 1);
  });

  // Navigate calendar when an external change sets dateFrom (e.g. AI search)
  useEffect(() => {
    if (!dateFrom) return;
    const [y, mo] = dateFrom.split('-').map(Number);
    setView((prev) => {
      if (prev.getFullYear() === y && prev.getMonth() === mo - 1) return prev;
      return new Date(y, mo - 1, 1);
    });
  }, [dateFrom]);

  const year = view.getFullYear();
  const month = view.getMonth();

  // Build calendar grid
  const firstDow = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const daysInPrev = new Date(year, month, 0).getDate();

  const cells = [];
  for (let i = firstDow - 1; i >= 0; i--)
    cells.push({ ds: toStr(new Date(year, month - 1, daysInPrev - i)), other: true });
  for (let d = 1; d <= daysInMonth; d++)
    cells.push({ ds: toStr(new Date(year, month, d)), other: false });
  let next = 1;
  while (cells.length % 7 !== 0)
    cells.push({ ds: toStr(new Date(year, month + 1, next++)), other: true });

  const handleClick = (ds) => {
    if (!dateFrom || (dateFrom && dateTo)) {
      // First click (or reset): set start
      onChange({ dateFrom: ds, dateTo: '' });
    } else {
      // Second click: set end, auto-sort so from is always earlier
      const from = ds < dateFrom ? ds : dateFrom;
      const to = ds < dateFrom ? dateFrom : ds;
      onChange({ dateFrom: from, dateTo: to });
    }
  };

  const todayStr = toStr(today);

  return (
    <div className="drp">
      {/* Header */}
      <div className="drp-header">
        <button className="drp-nav" onClick={() => setView(new Date(year, month - 1, 1))}>
          ‹
        </button>
        <span className="drp-title">
          {MONTHS[month]} {year}
        </span>
        <button className="drp-nav" onClick={() => setView(new Date(year, month + 1, 1))}>
          ›
        </button>
      </div>

      {/* Day names */}
      <div className="drp-grid">
        {DAYS.map((d) => (
          <div key={d} className="drp-weekday">
            {d}
          </div>
        ))}

        {cells.map(({ ds, other }) => {
          const isStart = ds === dateFrom;
          const isEnd = ds === dateTo;
          const inRange = dateFrom && dateTo && ds > dateFrom && ds < dateTo;
          const isToday = ds === todayStr;

          let cls = 'drp-day';
          if (other) cls += ' drp-other';
          if (isStart) cls += ' drp-start';
          if (isEnd) cls += ' drp-end';
          if (inRange) cls += ' drp-range';
          if (isToday && !isStart && !isEnd) cls += ' drp-today';

          return (
            <button key={ds + (other ? 'o' : '')} className={cls} onClick={() => handleClick(ds)}>
              {Number(ds.slice(8))}
            </button>
          );
        })}
      </div>

      {/* Selected range display */}
      {(dateFrom || dateTo) && (
        <div className="drp-footer">
          <span className="drp-range-text">
            {dateFrom ? formatDisplay(dateFrom) : '…'}
            {' → '}
            {dateTo ? formatDisplay(dateTo) : <em style={{ opacity: 0.5 }}>pick end</em>}
          </span>
          <button className="drp-clear" onClick={() => onChange({ dateFrom: '', dateTo: '' })}>
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
