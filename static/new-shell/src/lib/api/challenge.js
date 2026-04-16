import { getJson } from './http.js';

function encodeChallengeId(challengeId) {
  return encodeURIComponent(String(challengeId || '').trim());
}

export function fetchChallengeEntry(challengeId) {
  return getJson(`/api/challenge/${encodeChallengeId(challengeId)}`);
}
