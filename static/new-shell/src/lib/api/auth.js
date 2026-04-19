import { ApiError, clearLegacyAuthToken, storeLegacyAuthToken } from './http.js';
import { logApiFailure } from '../observability.js';

function getErrorMessage(data, fallback) {
  if (typeof data?.detail === 'string') {
    return data.detail;
  }

  if (Array.isArray(data?.detail) && data.detail.length > 0) {
    return data.detail
      .map((entry) => entry?.msg)
      .filter(Boolean)
      .join(' ') || fallback;
  }

  return fallback;
}

async function postLegacyAuth(path, payload, fallbackError) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const error = new ApiError(getErrorMessage(data, fallbackError), {
      status: response.status,
      authRequired: response.status === 401,
      rateLimited: response.status === 429,
    });

    logApiFailure(error, {
      surface: 'auth',
      path,
      method: 'POST',
    });
    throw error;
  }

  if (!data?.access_token) {
    throw new ApiError('Legacy auth succeeded without returning an access token.', {
      status: response.status,
    });
  }

  storeLegacyAuthToken(data.access_token);

  return data;
}

export function loginWithLegacyAuth({ email, password }) {
  return postLegacyAuth('/api/auth/login', { email, password }, 'Login failed.');
}

export function registerWithLegacyAuth({ username, email, password }) {
  return postLegacyAuth('/api/auth/register', { username, email, password }, 'Registration failed.');
}

export function logoutFromLegacyAuth() {
  clearLegacyAuthToken();
}
