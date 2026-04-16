export function createInitialReadCacheState() {
  return {
    postScore: null,
    inFlightPostScoreRefresh: null,
  };
}

export function createInitialChallengeState() {
  return {
    entriesById: {},
    resultsById: {},
  };
}

export function ensureReadCacheState(appState) {
  if (!appState || typeof appState !== 'object') {
    return createInitialReadCacheState();
  }

  if (!appState.readCache || typeof appState.readCache !== 'object') {
    appState.readCache = createInitialReadCacheState();
    return appState.readCache;
  }

  if (!('postScore' in appState.readCache)) {
    appState.readCache.postScore = null;
  }

  if (!('inFlightPostScoreRefresh' in appState.readCache)) {
    appState.readCache.inFlightPostScoreRefresh = null;
  }

  return appState.readCache;
}

export function resetPostScoreState(appState) {
  const readCache = ensureReadCacheState(appState);
  readCache.postScore = null;
  readCache.inFlightPostScoreRefresh = null;
  return readCache;
}

export function ensureChallengeState(appState) {
  if (!appState || typeof appState !== 'object') {
    return createInitialChallengeState();
  }

  if (!appState.challenge || typeof appState.challenge !== 'object') {
    appState.challenge = createInitialChallengeState();
    return appState.challenge;
  }

  if (!appState.challenge.entriesById || typeof appState.challenge.entriesById !== 'object') {
    appState.challenge.entriesById = {};
  }

  if (!appState.challenge.resultsById || typeof appState.challenge.resultsById !== 'object') {
    appState.challenge.resultsById = {};
  }

  return appState.challenge;
}

export function storeChallengeEntry(appState, challengeId, challengeEntry) {
  if (!challengeId || !challengeEntry) {
    return null;
  }

  const challengeState = ensureChallengeState(appState);
  challengeState.entriesById[challengeId] = {
    challengeId,
    raw: challengeEntry,
    updatedAt: Date.now(),
  };
  return challengeState.entriesById[challengeId];
}

export function getStoredChallengeEntry(appState, challengeId) {
  if (!challengeId) {
    return null;
  }

  return ensureChallengeState(appState).entriesById[challengeId] || null;
}

export function storeChallengeResult(appState, {
  challengeId,
  challengeEntry = null,
  analyzeResult = null,
} = {}) {
  if (!challengeId || !analyzeResult) {
    return null;
  }

  const challengeState = ensureChallengeState(appState);

  if (challengeEntry) {
    storeChallengeEntry(appState, challengeId, challengeEntry);
  }

  challengeState.resultsById[challengeId] = {
    challengeId,
    challengeEntry: challengeEntry || challengeState.entriesById[challengeId]?.raw || null,
    analyzeResult,
    updatedAt: Date.now(),
  };

  return challengeState.resultsById[challengeId];
}

export function getStoredChallengeResult(appState, challengeId) {
  if (!challengeId) {
    return null;
  }

  return ensureChallengeState(appState).resultsById[challengeId] || null;
}

export function clearChallengeResults(appState) {
  const challengeState = ensureChallengeState(appState);
  challengeState.resultsById = {};
  return challengeState;
}

export function createInitialAppState() {
  return {
    session: {
      status: 'unknown',
      user: null,
      error: null,
      hasToken: false,
      source: 'session-read-through',
    },
    scenes: {
      items: [],
      activeSceneId: null,
    },
    progress: {
      status: 'stubbed',
      level: null,
    },
    daily: {
      status: 'stubbed',
      challengeId: null,
    },
    challenge: createInitialChallengeState(),
    readCache: createInitialReadCacheState(),
  };
}
