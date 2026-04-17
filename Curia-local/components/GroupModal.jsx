import { useState } from 'react';

const VIBE_CATEGORIES = {
  Energy: ['chill', 'hype', 'lowkey', 'wild', 'cozy', 'intense'],
  Crowd: [
    'first-timers welcome',
    'regulars only',
    'strangers welcome',
    'small group',
    'big group',
    'solo-friendly',
  ],
  Social: ['pregame', 'afterparty', 'dinner first', 'grab drinks', 'carpool', 'tailgate'],
  Focus: [
    'study group',
    'networking',
    'just vibes',
    'deep talks',
    'creative collab',
    'competitive',
  ],
  Logistics: [
    '21+',
    'all ages',
    'free entry',
    'bringing food',
    'outdoor',
    'BYOB',
    'sober-friendly',
  ],
  Misc: [
    'matching outfits',
    'dog-friendly',
    'photos',
    'late night',
    'morning people',
    'budget-friendly',
    'VIP',
  ],
};

export default function GroupModal({
  mode,
  group,
  eventName,
  userName,
  error,
  onConfirm,
  onClose,
}) {
  const isCreate = mode === 'create';

  const [groupName, setGroupName] = useState('');
  const [description, setDescription] = useState('');
  const [capacity, setCapacity] = useState('');
  const [meetupDetails, setMeetupDetails] = useState('');
  const [vibeTags, setVibeTags] = useState([]);

  const toggleVibe = (tag) => {
    setVibeTags((prev) => (prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isCreate && !groupName.trim()) return;
    onConfirm({
      groupName: isCreate ? groupName.trim() : group.name,
      description: isCreate ? description.trim() : '',
      capacity: isCreate ? parseInt(capacity) || 0 : undefined,
      meetupDetails: isCreate ? meetupDetails.trim() : undefined,
      vibeTags: isCreate ? vibeTags : undefined,
    });
  };

  return (
    <div className="modal-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true">
        <h2 className="modal-title">{isCreate ? 'Create a Group' : `Join "${group?.name}"`}</h2>
        <p className="modal-subtitle">
          {isCreate
            ? `Organize a crew to attend "${eventName}"`
            : `Let the group know you're coming to "${eventName}"`}
        </p>

        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: 16 }}>
          Joining as <strong>{userName}</strong>
        </p>

        <form onSubmit={handleSubmit}>
          {isCreate && (
            <>
              <div className="form-group">
                <label className="form-label" htmlFor="group-name">
                  Group Name *
                </label>
                <input
                  id="group-name"
                  className="form-input"
                  type="text"
                  placeholder="e.g. Raikes Table, Friday Squad…"
                  value={groupName}
                  onChange={(e) => setGroupName(e.target.value)}
                  required
                  autoFocus
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="group-desc">
                  Description (optional)
                </label>
                <textarea
                  id="group-desc"
                  className="form-input"
                  placeholder="Tell others what this group is about…"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={2}
                />
              </div>

              <div className="form-row">
                <div className="form-group form-group-half">
                  <label className="form-label" htmlFor="capacity">
                    Max Group Size
                  </label>
                  <input
                    id="capacity"
                    className="form-input"
                    type="number"
                    min="2"
                    max="50"
                    placeholder="No limit"
                    value={capacity}
                    onChange={(e) => setCapacity(e.target.value)}
                  />
                </div>
                <div className="form-group form-group-half">
                  <label className="form-label" htmlFor="meetup">
                    Meetup Details
                  </label>
                  <input
                    id="meetup"
                    className="form-input"
                    type="text"
                    placeholder="e.g. Fountain at 6:45pm"
                    value={meetupDetails}
                    onChange={(e) => setMeetupDetails(e.target.value)}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Vibe Tags</label>
                <div className="vibe-picker-categorized">
                  {Object.entries(VIBE_CATEGORIES).map(([category, tags]) => (
                    <div key={category} className="vibe-category">
                      <span className="vibe-category-label">{category}</span>
                      <div className="vibe-picker">
                        {tags.map((tag) => (
                          <button
                            key={tag}
                            type="button"
                            className={`vibe-chip ${vibeTags.includes(tag) ? 'active' : ''}`}
                            onClick={() => toggleVibe(tag)}
                          >
                            {tag}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {error && (
            <p style={{ color: 'var(--danger, #e53e3e)', fontSize: '0.88rem', marginBottom: 12 }}>
              {error}
            </p>
          )}

          <div className="modal-footer">
            <button type="button" className="btn btn-ghost" onClick={onClose}>
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isCreate && !groupName.trim()}
            >
              {isCreate ? 'Create Group' : 'Join Group'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
