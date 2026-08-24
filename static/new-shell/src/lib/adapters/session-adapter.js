// Maps `/api/auth/me` into the app session snapshot. Auth mutations store or
// clear the existing `mirror_token`, then this adapter reads the verified user.
export function adaptSessionUser(rawUser) {
  if (!rawUser) {
    return null;
  }

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
