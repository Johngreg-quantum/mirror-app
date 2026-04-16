# Mirror App Shell

This directory contains the production app entry served by:

- `/`
- `/auth`
- `/levels`
- `/scene/:sceneId`
- `/progress`
- `/daily`
- `/challenge/:challengeId`

`/legacy` remains available as a temporary rollback path, and
`/legacy/challenge/:challengeId` remains available for challenge rollback links.

## Routing

The app uses history-path routing when served from the primary routes above.
Opening `/static/new-shell/index.html` directly uses hash routing for isolated
debugging.

## Session

The app reads `localStorage.mirror_token`, verifies the session with
`/api/auth/me`, and uses the existing auth endpoints for login, registration,
and logout. There is no alternate token model.

## Runtime

Scene detail owns the browser-only local audio take lifecycle:

- microphone capture with `MediaRecorder`
- local playback from the current recording blob
- `/api/submit` analyze submission
- post-score refresh for progress, leaderboard, profile, daily, and challenge data

Local recordings stay in the browser until submitted. Resetting or leaving the
page drops the local runtime state.

## Rollback

The primary app is served for the promoted routes. Use `/legacy` only as a
temporary fallback if a production issue appears in the app shell. Challenge
rollback links should preserve the challenge id with `/legacy/challenge/:challengeId`.
