import { fetchCurrentSession } from '../lib/api/read-data.js';
import { getReadOnlyAuthToken } from '../lib/api/http.js';
import {
  loginWithLegacyAuth,
  logoutFromLegacyAuth,
  registerWithLegacyAuth,
} from '../lib/api/auth.js';
import { adaptSessionUser } from '../lib/adapters/session-adapter.js';

const SESSION_REFRESH_DEDUPE_MS = 750;

function createSnapshot(overrides = {}) {
  return {
    status: 'unknown',
    user: null,
    error: null,
    hasToken: false,
    source: 'session-read-through',
    ...overrides,
  };
}

export function createSessionStore() {
  let snapshot = createSnapshot();
  let inFlightRefresh = null;
  let inFlightToken = null;
  let lastRefreshToken = null;
  let lastRefreshAt = 0;
  const subscribers = new Set();

  function setSnapshot(nextSnapshot) {
    snapshot = nextSnapshot;
    subscribers.forEach((subscriber) => subscriber(snapshot));
  }

  async function runSessionCheck(token) {
    setSnapshot(createSnapshot({
      status: snapshot.status === 'authenticated' ? 'authenticated' : 'loading',
      user: snapshot.user,
      hasToken: true,
      source: 'browser-session',
    }));

    try {
      const rawUser = await fetchCurrentSession();

      if (getReadOnlyAuthToken() !== token) {
        return snapshot;
      }

      setSnapshot(createSnapshot({
        status: 'authenticated',
        user: adaptSessionUser(rawUser),
        hasToken: true,
        source: '/api/auth/me',
      }));
    } catch (error) {
      if (getReadOnlyAuthToken() !== token) {
        return snapshot;
      }

      const authRequired = Boolean(error?.authRequired);

      if (authRequired) {
        logoutFromLegacyAuth();
      }

      setSnapshot(createSnapshot({
        status: authRequired ? 'unauthenticated' : 'error',
        error: authRequired ? null : error,
        hasToken: !authRequired,
        source: '/api/auth/me',
      }));
    }

    lastRefreshToken = token;
    lastRefreshAt = Date.now();
    return snapshot;
  }

  async function refreshSession({ force = false } = {}) {
    const token = getReadOnlyAuthToken();

    if (!token) {
      inFlightRefresh = null;
      inFlightToken = null;
      lastRefreshToken = null;
      setSnapshot(createSnapshot({
        status: 'unauthenticated',
        hasToken: false,
        source: 'browser-session',
      }));
      return snapshot;
    }

    if (inFlightRefresh && token === inFlightToken) {
      return inFlightRefresh;
    }

    const now = Date.now();

    if (
      !force
      && snapshot.status !== 'unknown'
      && snapshot.status !== 'loading'
      && token === lastRefreshToken
      && now - lastRefreshAt < SESSION_REFRESH_DEDUPE_MS
    ) {
      return snapshot;
    }

    inFlightToken = token;
    inFlightRefresh = runSessionCheck(token);
    const currentRefresh = inFlightRefresh;

    try {
      return await currentRefresh;
    } finally {
      if (inFlightRefresh === currentRefresh) {
        inFlightRefresh = null;
        inFlightToken = null;
      }
    }
  }

  async function loginWithLegacy(credentials) {
    await loginWithLegacyAuth(credentials);
    return refreshSession({ force: true });
  }

  async function registerWithLegacy(credentials) {
    await registerWithLegacyAuth(credentials);
    return refreshSession({ force: true });
  }

  async function logoutWithLegacy() {
    logoutFromLegacyAuth();
    return refreshSession();
  }

  return {
    getSnapshot() {
      return snapshot;
    },
    load: refreshSession,
    refreshSession,
    loginWithLegacy,
    registerWithLegacy,
    logoutWithLegacy,
    subscribe(subscriber) {
      subscribers.add(subscriber);
      return () => subscribers.delete(subscriber);
    },
  };
}
