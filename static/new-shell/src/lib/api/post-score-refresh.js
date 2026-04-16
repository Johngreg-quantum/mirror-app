import {
  fetchDailyChallenge,
  fetchHistory,
  fetchLeaderboard,
  fetchProfile,
  fetchProgress,
  fetchSceneConfig,
} from './read-data.js';
import { ensureReadCacheState } from '../../state/app-state.js';

const POST_SCORE_CACHE_TTL_MS = 30 * 1000;

export function getFreshPostScoreReadCache(appState) {
  const bundle = appState?.readCache?.postScore;

  if (!bundle?.refreshedAt) {
    return null;
  }

  if (Date.now() - bundle.refreshedAt > POST_SCORE_CACHE_TTL_MS) {
    return null;
  }

  return bundle;
}

export async function refreshPostScoreReadCache({ appState, sessionStatus = 'unknown' } = {}) {
  const readCache = ensureReadCacheState(appState);

  if (readCache.inFlightPostScoreRefresh) {
    return readCache.inFlightPostScoreRefresh;
  }

  const currentRefresh = (async () => {
    const previousBundle = readCache.postScore || {};
    const requests = [
      ['sceneConfig', fetchSceneConfig()],
      ['daily', fetchDailyChallenge()],
      ['leaderboard', fetchLeaderboard()],
    ];

    if (sessionStatus === 'authenticated') {
      requests.push(
        ['progress', fetchProgress()],
        ['profile', fetchProfile()],
        ['history', fetchHistory()],
      );
    }

    const settled = await Promise.allSettled(requests.map(([, request]) => request));
    const bundle = {
      ...previousBundle,
      refreshedAt: Date.now(),
      errors: {
        ...(previousBundle.errors || {}),
      },
    };

    settled.forEach((result, index) => {
      const [key] = requests[index];

      if (result.status === 'fulfilled') {
        bundle[key] = result.value;
        delete bundle.errors[key];
        return;
      }

      bundle.errors[key] = result.reason;
    });

    readCache.postScore = bundle;
    return bundle;
  })();

  readCache.inFlightPostScoreRefresh = currentRefresh;

  try {
    return await currentRefresh;
  } finally {
    if (readCache.inFlightPostScoreRefresh === currentRefresh) {
      readCache.inFlightPostScoreRefresh = null;
    }
  }
}
