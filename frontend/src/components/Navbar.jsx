import { useLayoutEffect, useRef, useState } from 'react'
import logo from '../assets/logo.png'
import { api } from '../api'
import NotificationBell from './NotificationBell'
import MyGroupsMenu from './MyGroupsMenu'

export default function Navbar({ searchQuery, onSearchChange, onSearchSubmit, onLogoClick, user, onUserChange, onNavigateToEvent, aiSearch, onAiToggle }) {
  const [showTagline, setShowTagline] = useState(false)
  const navbarRef = useRef(null)
  const searchRef = useRef(null)
  const searchInputRef = useRef(null)
  const actionsRef = useRef(null)
  const taglineRef = useRef(null)

  useLayoutEffect(() => {
    const measure = () => {
      const navbarEl = navbarRef.current
      const searchEl = searchRef.current
      const actionsEl = actionsRef.current
      const taglineEl = taglineRef.current
      if (!navbarEl || !searchEl || !actionsEl || !taglineEl) {
        setShowTagline(false)
        return
      }

      const navbarRect = navbarEl.getBoundingClientRect()
      const searchRect = (searchInputRef.current || searchEl).getBoundingClientRect()
      const actionsRect = actionsEl.getBoundingClientRect()
      const centerX = navbarRect.left + navbarRect.width / 2
      const leftFree = centerX - searchRect.right
      const rightFree = actionsRect.left - centerX
      const halfSpace = Math.min(leftFree, rightFree) - 8
      const halfTagline = taglineEl.offsetWidth / 2

      setShowTagline(halfSpace >= halfTagline)
    }

    const raf = requestAnimationFrame(measure)
    window.addEventListener('resize', measure)
    let ro
    if (window.ResizeObserver) {
      const navbarEl = navbarRef.current
      const searchEl = searchRef.current
      const actionsEl = actionsRef.current
      const taglineEl = taglineRef.current
      ro = new ResizeObserver(measure)
      if (navbarEl) ro.observe(navbarEl)
      if (searchEl) ro.observe(searchEl)
      if (actionsEl) ro.observe(actionsEl)
      if (taglineEl) ro.observe(taglineEl)
      if (searchInputRef.current) ro.observe(searchInputRef.current)
    }
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', measure)
      ro?.disconnect()
    }
  }, [user, searchQuery])

  const handleLogout = async () => {
    await api.logout()
    window.location.reload()
  }

  return (
    <nav className="navbar" ref={navbarRef}>
      <button
        className="navbar-brand"
        onClick={onLogoClick}
        style={{ background: 'none', border: 'none', cursor: 'pointer' }}
      >
        <img src={logo} alt="Curia" style={{ height: '80px' }} />
      </button>

      <div className="navbar-search" ref={searchRef}>
        <span className="navbar-search-icon">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
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
        We already know<br />where you're going.
      </span>

      <div className="navbar-actions" ref={actionsRef}>
        {user && <MyGroupsMenu onNavigateToEvent={onNavigateToEvent} />}
        {user && <NotificationBell onNavigateToEvent={onNavigateToEvent} />}
        {user ? (
          <>
            <div className="navbar-divider" />
            <div className="navbar-user">
              {user.picture && (
                <img src={user.picture} alt={user.name} className="navbar-avatar" referrerPolicy="no-referrer" />
              )}
              <span className="navbar-username">{user.name}</span>
              <button className="navbar-signout-btn" onClick={handleLogout} title="Sign out">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
                </svg>
                <span className="navbar-signout-text">Sign out</span>
              </button>
            </div>
          </>
        ) : (
          <a className="btn btn-primary btn-sm" href="/api/auth/google">
            Sign in with Google
          </a>
        )}
      </div>
    </nav>
  )
}
