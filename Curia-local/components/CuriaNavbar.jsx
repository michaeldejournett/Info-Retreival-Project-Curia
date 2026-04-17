import { useLayoutEffect, useRef, useState } from 'react';
import { getUser } from '../utils/groups.js';

export default function CuriaNavbar({
  searchQuery,
  onSearchChange,
  onSearchSubmit,
  onLogoClick,
  aiSearch,
  onAiToggle,
}) {
  const [showTagline, setShowTagline] = useState(false);
  const navbarRef = useRef(null);
  const searchRef = useRef(null);
  const searchInputRef = useRef(null);
  const actionsRef = useRef(null);
  const taglineRef = useRef(null);
  const user = getUser();

  useLayoutEffect(() => {
    const measure = () => {
      const navbarEl = navbarRef.current;
      const searchEl = searchRef.current;
      const actionsEl = actionsRef.current;
      const taglineEl = taglineRef.current;
      if (!navbarEl || !searchEl || !actionsEl || !taglineEl) {
        setShowTagline(false);
        return;
      }

      const navbarRect = navbarEl.getBoundingClientRect();
      const searchRect = (searchInputRef.current || searchEl).getBoundingClientRect();
      const actionsRect = actionsEl.getBoundingClientRect();
      const centerX = navbarRect.left + navbarRect.width / 2;
      const leftFree = centerX - searchRect.right;
      const rightFree = actionsRect.left - centerX;
      const halfSpace = Math.min(leftFree, rightFree) - 8;
      const halfTagline = taglineEl.offsetWidth / 2;
      setShowTagline(halfSpace >= halfTagline);
    };

    const raf = requestAnimationFrame(measure);
    window.addEventListener('resize', measure);
    let ro;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(measure);
      [navbarRef, searchRef, actionsRef, taglineRef, searchInputRef].forEach(
        (r) => r.current && ro.observe(r.current)
      );
    }
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', measure);
      ro?.disconnect();
    };
  }, [searchQuery]);

  return (
    <nav className="navbar" ref={navbarRef}>
      <button className="navbar-brand" onClick={onLogoClick}>
        <img src="/curia/logo.png" alt="Curia" style={{ height: '60px' }} />
      </button>

      <div className="navbar-search" ref={searchRef}>
        <span className="navbar-search-icon">
          <svg
            width="15"
            height="15"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </span>
        <input
          ref={searchInputRef}
          type="search"
          placeholder={aiSearch ? 'AI search — press Enter…' : 'Search events…'}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value, aiSearch)}
          onKeyDown={(e) => e.key === 'Enter' && onSearchSubmit?.()}
          aria-label="Search events"
        />
        <button
          className={`search-mode-toggle ${aiSearch ? 'active' : ''}`}
          onClick={() => onAiToggle?.(!aiSearch)}
          title={aiSearch ? 'Switch to keyword search' : 'Switch to AI search'}
        >
          ✨ AI
        </button>
      </div>

      <span
        ref={taglineRef}
        className={`navbar-tagline ${showTagline ? '' : 'navbar-tagline-hidden'}`}
      >
        We already know
        <br />
        where you're going.
      </span>

      <div className="navbar-actions" ref={actionsRef}>
        <span className="navbar-guest-badge" title="Groups are saved locally in your browser">
          👤 {user.name}
        </span>
      </div>
    </nav>
  );
}
