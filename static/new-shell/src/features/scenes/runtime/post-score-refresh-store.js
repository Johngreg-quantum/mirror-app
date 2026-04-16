import { refreshPostScoreReadCache } from '../../../lib/api/post-score-refresh.js';

function createSnapshot(overrides = {}) {
  return {
    status: 'idle',
    bundle: null,
    error: null,
    errors: {},
    refreshedAt: null,
    ...overrides,
  };
}

function buildRefreshError(errors = {}) {
  const failedKeys = Object.keys(errors);

  if (!failedKeys.length) {
    return null;
  }

  const firstError = errors[failedKeys[0]];
  const detail = firstError?.message ? ` ${firstError.message}` : '';

  return new Error(`Some post-score surfaces could not refresh.${detail}`);
}

export function createPostScoreRefreshStore({
  analyzeStore,
  appState,
  sessionStatus = 'unknown',
} = {}) {
  let snapshot = createSnapshot();
  let disposed = false;
  let lastResult = null;
  let requestVersion = 0;
  const subscribers = new Set();

  function publish(nextSnapshot) {
    if (disposed) {
      return;
    }

    snapshot = nextSnapshot;
    subscribers.forEach((subscriber) => subscriber(snapshot));
  }

  async function refreshForResult(result) {
    const activeVersion = requestVersion + 1;
    requestVersion = activeVersion;

    publish(createSnapshot({
      status: 'refreshing',
      bundle: snapshot.bundle,
      refreshedAt: snapshot.refreshedAt,
    }));

    const bundle = await refreshPostScoreReadCache({ appState, sessionStatus });

    if (disposed || requestVersion !== activeVersion) {
      return snapshot;
    }

    const error = buildRefreshError(bundle.errors);

    publish(createSnapshot({
      status: error ? 'degraded' : 'success',
      bundle,
      error,
      errors: bundle.errors || {},
      refreshedAt: bundle.refreshedAt || Date.now(),
    }));

    return snapshot;
  }

  const unsubscribeAnalyze = analyzeStore?.subscribe?.((analyzeSnapshot) => {
    if (analyzeSnapshot.status === 'success' && analyzeSnapshot.result && analyzeSnapshot.result !== lastResult) {
      lastResult = analyzeSnapshot.result;
      refreshForResult(analyzeSnapshot.result).catch((error) => {
        if (disposed) {
          return;
        }

        publish(createSnapshot({
          status: 'degraded',
          bundle: snapshot.bundle,
          error,
          errors: {},
          refreshedAt: snapshot.refreshedAt,
        }));
      });
    }
  });

  return {
    getSnapshot() {
      return snapshot;
    },
    cleanup() {
      disposed = true;
      requestVersion += 1;
      unsubscribeAnalyze?.();
      subscribers.clear();
    },
    subscribe(subscriber) {
      subscribers.add(subscriber);
      subscriber(snapshot);
      return () => subscribers.delete(subscriber);
    },
  };
}
