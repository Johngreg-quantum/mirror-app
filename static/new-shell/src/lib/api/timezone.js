import { getReadOnlyAuthToken } from './http.js';

// POST /api/profile/timezone — tells the server which calendar day is this
// user's, which is the day the streak is counted in. Streaks used to roll at UTC
// midnight, which is 8pm in New York, so a user practising at 6pm Monday and 9pm
// Tuesday local recorded Monday and Wednesday and lost the streak.
//
// Fire-and-forget on purpose, and silent on failure. A user with no stored
// timezone is treated as America/New_York, which is right for most of them and
// never worse than the UTC boundary it replaces — so there is nothing here
// worth interrupting a session for, and nothing worth an error toast.
//
// Sent only when the browser's zone differs from what the server already holds,
// so a returning user costs no request.
export function reportTimezone(storedTimezone) {
  const token = getReadOnlyAuthToken();
  if (!token) return;

  let tz = null;
  try {
    tz = Intl.DateTimeFormat().resolvedOptions().timeZone || null;
  } catch (err) {
    return;   // no Intl, or it refused: the default stands
  }
  if (!tz || tz === storedTimezone) return;

  fetch('/api/profile/timezone', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ timezone: tz }),
  }).catch(() => {});
}
