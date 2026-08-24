import { ApiError, getReadOnlyAuthToken } from './http.js';
import { logApiFailure } from '../observability.js';

// POST /api/consent/recording — records that the user accepted the first-run
// recording notice (privacy policy §6). Acceptance is stored on the user record
// server-side, not in browser storage, so it follows the account across devices.
export async function acceptRecordingConsent() {
  const token = getReadOnlyAuthToken();

  if (!token) {
    const error = new ApiError('Sign in before recording.', {
      status: 401,
      authRequired: true,
    });

    logApiFailure(error, {
      surface: 'consent',
      method: 'POST',
      path: '/api/consent/recording',
      reason: 'missing-token',
    });
    throw error;
  }

  let response;

  try {
    response = await fetch('/api/consent/recording', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch (cause) {
    const error = new ApiError('Could not save your choice. Please try again.', {
      status: 0,
      cause,
    });

    logApiFailure(error, {
      surface: 'consent',
      method: 'POST',
      path: '/api/consent/recording',
    });
    throw error;
  }

  if (!response.ok) {
    const error = new ApiError('Could not save your choice. Please try again.', {
      status: response.status,
      authRequired: response.status === 401,
    });

    logApiFailure(error, {
      surface: 'consent',
      method: 'POST',
      path: '/api/consent/recording',
    });
    throw error;
  }

  return true;
}
