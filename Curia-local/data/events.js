export const CATEGORIES = [
  { id: 'technology', label: 'Technology', color: '#6366f1', bg: '#eef2ff' },
  { id: 'music', label: 'Music', color: '#ec4899', bg: '#fdf2f8' },
  { id: 'sports', label: 'Sports', color: '#10b981', bg: '#d1fae5' },
  { id: 'food', label: 'Food & Drink', color: '#f59e0b', bg: '#fffbeb' },
  { id: 'arts', label: 'Arts & Culture', color: '#8b5cf6', bg: '#ede9fe' },
  { id: 'community', label: 'Community', color: '#0ea5e9', bg: '#e0f2fe' },
  { id: 'health', label: 'Health & Wellness', color: '#14b8a6', bg: '#ccfbf1' },
  { id: 'education', label: 'Education', color: '#f97316', bg: '#fff7ed' },
];

export const getCategoryMeta = (id) =>
  CATEGORIES.find((c) => c.id === id) ?? { label: id ?? 'Other', color: '#6366f1', bg: '#eef2ff' };
