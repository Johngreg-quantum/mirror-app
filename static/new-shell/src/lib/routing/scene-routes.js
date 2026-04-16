import { createAppHref } from './navigation.js';

const SCENE_ENTRY_LABELS = {
  home: 'Scene browser',
  levels: 'Levels',
  daily: 'Daily',
  leaderboard: 'Leaderboard',
  challenge: 'Challenge',
};

const SCENE_ENTRY_BACK_PATHS = {
  home: '/',
  levels: '/levels',
  daily: '/daily',
  leaderboard: '/',
  challenge: '/challenge/sample-challenge',
};

function encodePathPart(value) {
  return encodeURIComponent(String(value || '')).replace(/%2F/g, '/');
}

export function scenePath(sceneId, query = {}) {
  const params = new URLSearchParams();

  Object.entries(query).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      params.set(key, value);
    }
  });

  const suffix = params.toString() ? `?${params.toString()}` : '';
  return `/scene/${encodePathPart(sceneId)}${suffix}`;
}

export function sceneHref(sceneId, query = {}) {
  return createAppHref(scenePath(sceneId, query));
}

export function getSceneEntryLabel(source) {
  return SCENE_ENTRY_LABELS[source] || 'Scene browser';
}

export function getSceneBackPath(query = {}) {
  if (query.from === 'challenge' && query.challengeId) {
    return `/challenge/${encodePathPart(query.challengeId)}`;
  }

  return SCENE_ENTRY_BACK_PATHS[query.from] || '/';
}

export function getSceneBackHref(query = {}) {
  return createAppHref(getSceneBackPath(query));
}
