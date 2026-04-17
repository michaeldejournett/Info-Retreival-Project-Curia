/**
 * LocalStorage-based groups store for Curia on GitHub Pages.
 * Replaces the backend SQLite/Express groups API.
 *
 * Groups are stored per-event in localStorage under the key "curia_groups".
 * Messages are stored under "curia_messages_<groupId>".
 * User identity is a random guest ID stored in "curia_user".
 */

// ── Guest user identity ───────────────────────────────────────────────────────
export function getUser() {
  let user = null;
  try {
    user = JSON.parse(localStorage.getItem('curia_user'));
  } catch {}
  if (!user) {
    const id = Math.random().toString(36).slice(2, 8).toUpperCase();
    user = { id, name: `Guest_${id}` };
    localStorage.setItem('curia_user', JSON.stringify(user));
  }
  return user;
}

// ── Seed data — a few example groups to give the page life ───────────────────
const SEED_GROUPS = [
  {
    eventId: null, // filled in on first use
    name: "Let's go together!",
    description: 'Anyone want to meet up before and walk over together?',
    capacity: 5,
    vibes: ['casual', 'friendly'],
    meetupDetails: 'Meet at the Union front entrance at 15 min before start',
    creatorId: 'SEED01',
    creatorName: 'Husker Fan',
    members: [
      { id: 'SEED01', name: 'Husker Fan' },
      { id: 'SEED02', name: 'StudyBuddy' },
    ],
    messages: [
      {
        authorId: 'SEED01',
        authorName: 'Husker Fan',
        body: 'Hey everyone, excited for this!',
        ts: Date.now() - 3600000,
      },
      {
        authorId: 'SEED02',
        authorName: 'StudyBuddy',
        body: 'Me too! See you there 👋',
        ts: Date.now() - 1800000,
      },
    ],
  },
];

// ── Storage helpers ───────────────────────────────────────────────────────────
function loadAll() {
  try {
    return JSON.parse(localStorage.getItem('curia_groups')) ?? {};
  } catch {
    return {};
  }
}

function saveAll(data) {
  localStorage.setItem('curia_groups', JSON.stringify(data));
}

function nextId() {
  return `g_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Returns { [eventId]: count } for every event that has stored groups. No seeding. */
export function getAllGroupCounts() {
  const all = loadAll();
  const counts = {};
  for (const [eventId, groups] of Object.entries(all)) {
    counts[eventId] = Array.isArray(groups) ? groups.length : 0;
  }
  return counts;
}

export function getGroups(eventId) {
  const all = loadAll();
  const groups = all[eventId] ?? [];

  // Seed one example group the first time this event is viewed
  if (groups.length === 0) {
    const seed = {
      ...SEED_GROUPS[0],
      id: nextId(),
      eventId,
      ts: Date.now() - 7200000,
    };
    all[eventId] = [seed];
    saveAll(all);
    return [seed];
  }

  return groups;
}

export function createGroup(eventId, { name, description, capacity, vibes, meetupDetails }) {
  const user = getUser();
  const all = loadAll();
  const groups = all[eventId] ?? [];

  const group = {
    id: nextId(),
    eventId,
    name,
    description,
    capacity: Number(capacity) || 8,
    vibes: vibes ?? [],
    meetupDetails: meetupDetails ?? '',
    creatorId: user.id,
    creatorName: user.name,
    members: [{ id: user.id, name: user.name }],
    messages: [],
    ts: Date.now(),
  };

  all[eventId] = [group, ...groups];
  saveAll(all);
  return group;
}

export function joinGroup(eventId, groupId) {
  const user = getUser();
  const all = loadAll();
  const groups = all[eventId] ?? [];
  const idx = groups.findIndex((g) => g.id === groupId);
  if (idx === -1) return null;

  const group = groups[idx];
  if (group.members.some((m) => m.id === user.id)) return group;
  if (group.members.length >= group.capacity) throw new Error('Group is full');

  group.members = [...group.members, { id: user.id, name: user.name }];
  saveAll(all);
  return group;
}

export function leaveGroup(eventId, groupId) {
  const user = getUser();
  const all = loadAll();
  const groups = all[eventId] ?? [];
  const idx = groups.findIndex((g) => g.id === groupId);
  if (idx === -1) return;

  const group = groups[idx];
  group.members = group.members.filter((m) => m.id !== user.id);

  // Delete group if creator leaves and it's empty
  if (group.creatorId === user.id && group.members.length === 0) {
    all[eventId] = groups.filter((g) => g.id !== groupId);
  }

  saveAll(all);
}

export function deleteGroup(eventId, groupId) {
  const all = loadAll();
  all[eventId] = (all[eventId] ?? []).filter((g) => g.id !== groupId);
  saveAll(all);
}

export function getMessages(eventId, groupId) {
  const all = loadAll();
  const group = (all[eventId] ?? []).find((g) => g.id === groupId);
  return group?.messages ?? [];
}

export function postMessage(eventId, groupId, body) {
  const user = getUser();
  const all = loadAll();
  const groups = all[eventId] ?? [];
  const group = groups.find((g) => g.id === groupId);
  if (!group) return null;

  const msg = { authorId: user.id, authorName: user.name, body, ts: Date.now() };
  group.messages = [...(group.messages ?? []), msg];
  saveAll(all);
  return msg;
}

export function isMember(group) {
  const user = getUser();
  return group.members.some((m) => m.id === user.id);
}

export function isCreator(group) {
  const user = getUser();
  return group.creatorId === user.id;
}
