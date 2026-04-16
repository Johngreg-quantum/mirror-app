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
    source: rawUser,
  };
}
