import { useState } from 'react';
import DateRangePicker from './DateRangePicker';

const PAGE_SIZE_OPTIONS = [10, 20, 50];

export default function SearchFilters({
  filters,
  onChange,
  onReset,
  categories,
  pageSize,
  onPageSizeChange,
  page,
  totalPages,
  totalCount,
  onPageChange,
}) {
  const [open, setOpen] = useState(() => window.innerWidth > 900);

  const set = (key, value) => onChange({ ...filters, [key]: value });

  const toggleCategory = (id) => {
    const current = filters.category;
    set('category', current.includes(id) ? current.filter((c) => c !== id) : [...current, id]);
  };

  const hasActiveFilters = filters.category.length || filters.dateFrom || filters.dateTo;

  const activeCount = [
    filters.category.length ? 'cat' : '',
    filters.dateFrom || filters.dateTo ? 'date' : '',
  ].filter(Boolean).length;

  return (
    <>
      <aside className={`filters-sidebar${open ? '' : ' collapsed'}`}>
        {/* Mobile-only header — always visible, acts as toggle */}
        <button className="filter-mobile-header" onClick={() => setOpen((o) => !o)}>
          <span>
            Filters{' '}
            {activeCount > 0 && <span style={{ color: 'var(--primary)' }}>({activeCount})</span>}
          </span>
          <span>{open ? '✕' : '▼'}</span>
        </button>

        {/* Filter content */}
        <div className="filter-content">
          <div className="filters-title">
            <span>Filters</span>
            {hasActiveFilters && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={onReset}
                style={{ padding: '2px 8px' }}
              >
                Clear all
              </button>
            )}
          </div>

          {/* Category — multi-select */}
          <div className="filter-group">
            <span className="filter-label">Category</span>
            <div className="category-pills">
              {categories.map((cat) => (
                <button
                  key={cat.id}
                  className={`category-pill ${filters.category.includes(cat.id) ? 'active' : ''}`}
                  onClick={() => toggleCategory(cat.id)}
                >
                  {cat.label}
                </button>
              ))}
            </div>
          </div>

          {/* Date range */}
          <div className="filter-group">
            <span className="filter-label">Date Range</span>
            <DateRangePicker
              dateFrom={filters.dateFrom}
              dateTo={filters.dateTo}
              onChange={({ dateFrom, dateTo }) => onChange({ ...filters, dateFrom, dateTo })}
            />
          </div>

          {/* Page size + pagination */}
          <div className="filter-group" style={{ marginBottom: 0 }}>
            <div className="page-size-selector" style={{ marginBottom: 10 }}>
              <label
                htmlFor="sidebar-page-size"
                style={{
                  fontWeight: 600,
                  fontSize: '0.78rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: 'var(--text-muted)',
                }}
              >
                Show
              </label>
              <select
                id="sidebar-page-size"
                value={pageSize}
                onChange={(e) => onPageSizeChange(Number(e.target.value))}
              >
                {PAGE_SIZE_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
              <span>per page</span>
            </div>

            {totalCount > 0 && (
              <>
                <div className="sidebar-range">
                  Events {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, totalCount)} of{' '}
                  {totalCount.toLocaleString()}
                </div>
                <div className="sidebar-pagination">
                  <button
                    className="btn btn-outline"
                    disabled={page === 1}
                    onClick={() => onPageChange((p) => p - 1)}
                  >
                    ‹ Prev
                  </button>
                  <span className="sidebar-page-info">
                    {page} / {totalPages}
                  </span>
                  <button
                    className="btn btn-outline"
                    disabled={page === totalPages}
                    onClick={() => onPageChange((p) => p + 1)}
                  >
                    Next ›
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile backdrop — tap to close */}
      {open && <div className="filter-backdrop" onClick={() => setOpen(false)} />}
    </>
  );
}
