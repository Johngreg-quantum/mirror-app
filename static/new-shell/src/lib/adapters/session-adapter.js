import { reportTimezone } from '../api/timezone.js';

// Maps `/api/auth/me` into the app session snapshot. Auth mutations store or
// clear the existing `mirror_token`, then this adapter reads the verified user.
export function adaptSessionUser(rawUser) {
  if (!rawUser) {
    return null;
  }

  // Every authenticated session passes through here on both shells, which makes
  // it the one place the browser's timezone can be reported without adding a
  // boot step. No-ops when the server already has this zone.
  reportTimezone(rawUser.timezone || null);

  return {
    id: rawUser.id ?? null,
    username: rawUser.username || 'performer',
    displayName: rawUser.username || 'Performer',
    // Whether the first-run recording notice (§6) has been accepted. Anything
    // other than an explicit true reads as "not consented" so the notice shows
    // — never assume consent from a missing or malformed field.
    recordingConsent: rawUser.recording_consent === true,
    source: rawUser,
  };
}
