import { useState, useEffect } from 'react';
import { getCategoryMeta } from '../data/events.js';
import { generateIcs } from '../utils/icsGenerator.js';
import GroupModal from './GroupModal.jsx';
import GroupMessageBoard from './GroupMessageBoard.jsx';
import {
  getGroups,
  createGroup,
  joinGroup,
  leaveGroup,
  deleteGroup,
  isMember,
  isCreator,
  getUser,
} from '../utils/groups.js';

function formatDate(dateStr) {
  if (!dateStr) return 'TBD';
  const d = new Date(dateStr + 'T12:00:00');
  return d.toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
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

export default function EventDetail({ event, onBack }) {
  const cat = getCategoryMeta(event.category);
  const user = getUser();
  const [groups, setGroups] = useState([]);
  const [modal, setModal] = useState(null); // { mode: 'create' | 'join', group? }
  const [modalError, setModalError] = useState('');
  const [expandedChat, setExpandedChat] = useState({});

  const refresh = () => setGroups(getGroups(event.id));

  useEffect(() => {
    refresh();
  }, [event.id]);

  const handleDownloadCalendar = () => {
    const icsContent = generateIcs(event);
    const blob = new Blob([icsContent], { type: 'text/calendar' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${event.name.replace(/\s+/g, '_')}.ics`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleModalConfirm = ({ groupName, description, capacity, meetupDetails, vibeTags }) => {
    try {
      if (modal.mode === 'create') {
        createGroup(event.id, {
          name: groupName,
          description,
          capacity: capacity || 8,
          vibes: vibeTags ?? [],
          meetupDetails,
        });
      } else {
        joinGroup(event.id, modal.group.id);
      }
      refresh();
      setModal(null);
      setModalError('');
    } catch (err) {
      setModalError(err.message);
    }
  };

  const handleLeave = (groupId) => {
    leaveGroup(event.id, groupId);
    refresh();
  };

  const handleDelete = (groupId) => {
    if (!confirm('Delete this group?')) return;
    deleteGroup(event.id, groupId);
    refresh();
  };

  const toggleChat = (groupId) => {
    setExpandedChat((prev) => ({ ...prev, [groupId]: !prev[groupId] }));
  };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <button className="detail-back" onClick={onBack}>
        ← Back to events
      </button>

      <div className="detail-hero">
        {event.imageUrl ? (
          <img className="detail-hero-image" src={event.imageUrl} alt={event.name} />
        ) : (
          <div className="detail-hero-banner" style={{ height: 16, background: cat.color }} />
        )}

        <div className="detail-hero-body">
          <div className="detail-category-badge" style={{ background: cat.bg, color: cat.color }}>
            {cat.label}
          </div>

          <h1 className="detail-name">{event.name}</h1>

          <div className="detail-meta-grid">
            <div className="detail-meta-item">
              <span className="detail-meta-icon">📅</span>
              <div>
                <div className="detail-meta-label">Date</div>
                <div className="detail-meta-value">{formatDate(event.date)}</div>
              </div>
            </div>

            {event.time && (
              <div className="detail-meta-item">
                <span className="detail-meta-icon">🕐</span>
                <div>
                  <div className="detail-meta-label">Time</div>
                  <div className="detail-meta-value">
                    {formatTime(event.time)}
                    {event.endTime ? ` – ${formatTime(event.endTime)}` : ''}
                  </div>
                </div>
              </div>
            )}

            <div className="detail-meta-item">
              <span className="detail-meta-icon">📍</span>
              <div>
                <div className="detail-meta-label">Location</div>
                <div className="detail-meta-value">{event.venue || 'UNL Campus'}</div>
              </div>
            </div>

            {event.audience?.length > 0 && (
              <div className="detail-meta-item">
                <span className="detail-meta-icon">👥</span>
                <div>
                  <div className="detail-meta-label">Audience</div>
                  <div className="detail-meta-value" style={{ fontSize: '0.85rem' }}>
                    {event.audience.join(', ')}
                  </div>
                </div>
              </div>
            )}
          </div>

          {event.description && <p className="detail-description">{event.description}</p>}

          <div className="detail-actions">
            <button className="btn btn-outline" onClick={handleDownloadCalendar}>
              📅 Add to Calendar
            </button>
            {event.url && (
              <a
                href={event.url}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-ghost"
              >
                🔗 View on UNL Events
              </a>
            )}
          </div>
        </div>
      </div>

      {/* Groups Section */}
      <div className="groups-section">
        <div className="groups-section-header">
          <h2 className="groups-section-title">
            Looking for a Group
            {groups.length > 0 && (
              <span
                style={{
                  fontWeight: 400,
                  fontSize: '0.9rem',
                  color: 'var(--text-muted)',
                  marginLeft: 8,
                }}
              >
                ({groups.length})
              </span>
            )}
          </h2>
          <button
            className="btn btn-primary btn-sm"
            onClick={() => {
              setModal({ mode: 'create' });
              setModalError('');
            }}
          >
            + Create Group
          </button>
        </div>

        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 16 }}>
          Groups are saved locally in your browser. You are signed in as{' '}
          <strong>{user.name}</strong>.
        </p>

        {groups.length === 0 ? (
          <div className="groups-empty">
            <p>No groups yet — be the first to create one!</p>
          </div>
        ) : (
          <div className="groups-list">
            {groups.map((group) => {
              const member = isMember(group);
              const creator = isCreator(group);
              const full = group.members.length >= group.capacity;
              const chatOpen = expandedChat[group.id];

              return (
                <div
                  key={group.id}
                  className={`group-card ${full && !member ? 'group-card-full' : ''}`}
                >
                  <div className="group-card-header">
                    <div className="group-card-title-row">
                      <span className="group-card-name">{group.name}</span>
                      <span className={`group-status-badge ${full ? 'full' : 'open'}`}>
                        {full ? 'Full' : 'Open'}
                      </span>
                      {member && <span className="badge joined-badge">Joined</span>}
                    </div>
                    <div className="group-card-actions">
                      {creator ? (
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={() => handleDelete(group.id)}
                        >
                          Delete
                        </button>
                      ) : member ? (
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() => handleLeave(group.id)}
                        >
                          Leave
                        </button>
                      ) : (
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => {
                            setModal({ mode: 'join', group });
                            setModalError('');
                          }}
                          disabled={full}
                        >
                          Join
                        </button>
                      )}
                    </div>
                  </div>

                  <p className="group-card-meta">Organized by {group.creatorName}</p>

                  {/* Capacity bar */}
                  <div className="capacity-bar-wrapper">
                    <div className="capacity-bar">
                      <div
                        className={`capacity-bar-fill ${full ? 'full' : ''}`}
                        style={{
                          width: `${Math.min((group.members.length / group.capacity) * 100, 100)}%`,
                        }}
                      />
                    </div>
                    <span className="capacity-label">
                      {group.members.length}/{group.capacity}
                    </span>
                  </div>

                  {group.description && (
                    <p className="group-card-description">{group.description}</p>
                  )}

                  {group.meetupDetails && (
                    <div className="group-meetup">
                      <span className="group-meetup-icon">📌</span>
                      {group.meetupDetails}
                    </div>
                  )}

                  {group.vibes?.length > 0 && (
                    <div className="group-vibes">
                      {group.vibes.map((v) => (
                        <span key={v} className="vibe-tag">
                          {v}
                        </span>
                      ))}
                    </div>
                  )}

                  <div className="group-members">
                    {group.members.map((m) => (
                      <span key={m.id} className="member-chip">
                        {m.name}
                      </span>
                    ))}
                  </div>

                  <div className="group-toolbar">
                    <button className="btn btn-ghost btn-sm" onClick={() => toggleChat(group.id)}>
                      💬 {chatOpen ? 'Hide Chat' : 'Show Chat'}
                    </button>
                  </div>

                  {chatOpen && <GroupMessageBoard eventId={event.id} group={group} />}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {modal && (
        <GroupModal
          mode={modal.mode}
          group={modal.group}
          eventName={event.name}
          userName={user.name}
          error={modalError}
          onConfirm={handleModalConfirm}
          onClose={() => {
            setModal(null);
            setModalError('');
          }}
        />
      )}
    </div>
  );
}
