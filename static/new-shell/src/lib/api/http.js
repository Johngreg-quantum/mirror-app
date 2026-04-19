import { logApiFailure } from '../observability.js';

export class ApiError extends Error {
  constructor(message, {
    status = 0,
    authRequired = false,
    rateLimited = false,
    retryAfterMs = null,
    attempts = 1,
  } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.authRequired = authRequired;
    this.rateLimited = rateLimited;
    this.retryAfterMs = retryAfterMs;
    this.attempts = attempts;
  }
}

const GET_RETRY_POLICY = {
  maxRetries: 1,
  baseDelayMs: 450,
  maxDelayMs: 1600,
  jitterMs: 250,
};

export const LEGACY_AUTH_TOKEN_KEY = 'mirror_token';

export function getLegacyAuthToken() {
  try {
    return window.localStorage.getItem(LEGACY_AUTH_TOKEN_KEY);
  } catch {
    return null;
  }
}

export function getReadOnlyAuthToken() {
  return getLegacyAuthToken();
}

export function storeLegacyAuthToken(token) {
  try {
    window.localStorage.setItem(LEGACY_AUTH_TOKEN_KEY, token);
  } catch {
    throw new ApiError('Could not store the auth session in this browser.', {
      status: 0,
      authRequired: true,
    });
  }
}

export function clearLegacyAuthToken() {
  try {
    window.localStorage.removeItem(LEGACY_AUTH_TOKEN_KEY);
  } catch {
    // If storage is unavailable, the next session refresh will still settle
    // from the browser's current state.
  }
}

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function parseRetryAfterMs(response) {
  const retryAfter = response.headers.get('Retry-After');

  if (!retryAfter) {
    return null;
  }

  const seconds = Number(retryAfter);

  if (Number.isFinite(seconds)) {
    return Math.max(0, seconds * 1000);
  }

  const retryDate = Date.parse(retryAfter);

  if (Number.isFinite(retryDate)) {
    return Math.max(0, retryDate - Date.now());
  }

  return null;
}

function getRetryDelayMs(response, attemptIndex) {
  const retryAfterMs = parseRetryAfterMs(response);
  const exponentialDelay = GET_RETRY_POLICY.baseDelayMs * (2 ** attemptIndex);
  const jitter = Math.floor(Math.random() * GET_RETRY_POLICY.jitterMs);
  const delay = retryAfterMs ?? exponentialDelay + jitter;

  return Math.min(delay, GET_RETRY_POLICY.maxDelayMs);
}

function createResponseError(response, { attempts = 1 } = {}) {
  const retryAfterMs = parseRetryAfterMs(response);

  if (response.status === 429) {
    const error = new ApiError('Mirror is receiving too many requests. Please wait a moment, then try again.', {
      status: response.status,
      rateLimited: true,
      retryAfterMs,
      attempts,
    });

    logApiFailure(error, {
      surface: 'api-read',
      method: 'GET',
      path: response.url || '',
    });
    throw error;
  }

  const error = new ApiError(`Request failed with status ${response.status}.`, {
    status: response.status,
    authRequired: response.status === 401,
    attempts,
  });

  logApiFailure(error, {
    surface: 'api-read',
    method: 'GET',
    path: response.url || '',
  });
  throw error;
}

export async function getJson(path, { auth = false } = {}) {
  const headers = {};

  if (auth) {
    const token = getReadOnlyAuthToken();

    if (!token) {
      const error = new ApiError('Sign in to view personalized data here.', {
        status: 401,
        authRequired: true,
      });

      logApiFailure(error, {
        surface: 'api-read',
        method: 'GET',
        path,
        reason: 'missing-token',
      });
      throw error;
    }

    headers.Authorization = `Bearer ${token}`;
  }

  for (let attemptIndex = 0; attemptIndex <= GET_RETRY_POLICY.maxRetries; attemptIndex += 1) {
    const response = await fetch(path, { headers, method: 'GET' });

    if (response.ok) {
      return response.json();
    }

    if (response.status === 429 && attemptIndex < GET_RETRY_POLICY.maxRetries) {
      await wait(getRetryDelayMs(response, attemptIndex));
      continue;
    }

    createResponseError(response, { attempts: attemptIndex + 1 });
  }

  throw new ApiError('Request failed before the existing API returned a response.');
}
