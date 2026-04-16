import { getJson } from './http.js';

export function fetchSceneConfig() {
  return getJson('/api/scene-config');
}

export function fetchCurrentSession() {
  return getJson('/api/auth/me', { auth: true });
}

export function fetchDailyChallenge() {
  return getJson('/api/daily');
}

export function fetchLeaderboard() {
  return getJson('/api/leaderboard');
}

export function fetchProgress() {
  return getJson('/api/progress', { auth: true });
}

export function fetchProfile() {
  return getJson('/api/profile', { auth: true });
}

export function fetchHistory() {
  return getJson('/api/history', { auth: true });
}

export async function loadPublicHomeData() {
  const sceneConfig = await fetchSceneConfig();
  const daily = await fetchDailyChallenge();
  const leaderboard = await fetchLeaderboard();

  return { sceneConfig, daily, leaderboard };
}

export async function loadPersonalReadData() {
  const progress = await fetchProgress();
  const profile = await fetchProfile();
  const history = await fetchHistory();

  return { progress, profile, history };
}
