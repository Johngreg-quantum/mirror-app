# Legacy Transition Manifest

This manifest records the current split between the legacy `static/app.js` shell and the extracted browser modules. It is a migration guide, not a runtime contract file.

## Current File Map

### Legacy Shell

- `static/app.js`
  - Owns app state, startup routing, fetches, event wiring, modal lifecycle, media handles, timers, challenge flow, analyze submit flow, and scene-entry gating.
  - Should remain the source of truth until a replacement architecture owns those responsibilities.

### Render-Only Domains

- `static/auth-modal-domain.js`
  - Auth tab/error/submit button display only.
- `static/scene-browser-domain.js`
  - Scene card grid and level bar display only.
- `static/scene-modal-domain.js`
  - Scene modal display state only.
- `static/analyze-score-domain.js`
  - Score panel, points display, phoneme display, and challenge result display only.
- `static/recording-playback-domain.js`
  - Recording/playback button and timer display only.
- `static/daily-challenge-domain.js`
  - Daily card, streak card, and daily-complete display only.
- `static/level-panel-domain.js`
  - Cinematic level card stats and level panel display only.
- `static/progress-dashboard-domain.js`
  - Progress dashboard, division card, and personal best display only.

### Helpers

- `static/app-config.js`
  - Static frontend constants.
- `static/app-helpers.js`
  - Pure utility helpers and formatting/domain calculations.
- `static/app-view-helpers.js`
  - Reusable HTML builders for view fragments.
- `static/app-render-helpers.js`
  - Reusable render HTML builders.
- `static/app-state-helpers.js`
  - Read-only state derivation helpers.
- `static/app-dom-helpers.js`
  - Tiny guarded DOM setters.

### Runtime Utilities

- `static/app-runtime-utils.js`
  - Low-level cleanup, MIME detection, YouTube end-check, and waveform utility bodies.
  - Does not own timers, media state, YouTube player state, or app flow.

### Orchestration Helpers

- `static/app-progress-orchestration.js`
  - Refreshes scene cards, level bar, and level-card stats from passed state/callbacks.
- `static/app-post-score-orchestration.js`
  - Sequences post-score refresh: active leaderboard tab, leaderboard/progress reload callbacks, cards refresh, level-up callback.
- `static/app-leaderboard-orchestration.js`
  - Renders leaderboard tabs/panels and switches active leaderboard tab via passed setter.
- `static/app-level-panel-orchestration.js`
  - Coordinates level panel render/open and first-unlocked-scene play button wiring via passed callbacks.

## Dependency Map

`index.html` loads extracted files before `static/app.js`.

`static/app.js` depends on:

- Config: `app-config.js`
- Helpers: `app-helpers.js`, `app-view-helpers.js`, `app-render-helpers.js`, `app-state-helpers.js`, `app-dom-helpers.js`
- Render-only domains: `auth-modal-domain.js`, `scene-browser-domain.js`, `scene-modal-domain.js`, `analyze-score-domain.js`, `recording-playback-domain.js`, `daily-challenge-domain.js`, `level-panel-domain.js`, `progress-dashboard-domain.js`
- Runtime utilities: `app-runtime-utils.js`
- Orchestration helpers: `app-progress-orchestration.js`, `app-post-score-orchestration.js`, `app-leaderboard-orchestration.js`, `app-level-panel-orchestration.js`

## Ownership Map

### Keep In Legacy Shell

- Bootstrap/session entry: challenge URL detection, token verification, auth/app screen entry, onboarding timing, initial data fan-out.
- Auth/session state: `authToken`, `authUser`, local storage token ownership.
- Scene config state: `scenes`, `LEVEL_MAP`, `CLV_LEVELS`, `DEFAULT_UNLOCKED_SCENES`, `sceneConfigPromise`.
- Progress state: `userProgress`, progress fetching, post-fetch assignment.
- Daily state/timer: `dailyChallenge`, `countdownInterval`, daily fetch, streak/profile fetch, countdown reset behavior.
- Scene entry/modal lifecycle: `activeScene`, `openModal`, `closeModal`, `selectScene`, modal close priority.
- Recording/playback state: `mediaRecorder`, `audioChunks`, `audioBlob`, `audioEl`, `micStream`, `timerInterval`, `recSecs`, waveform refs.
- YouTube state: `ytPlayer`, `ytApiReady`, `ytEndInterval`, player setup and callback ownership.
- Analyze submit flow: `/api/submit`, 401 auth handoff, `showScore(data)` call, post-score refresh call.
- Challenge flow: `activeChallenge`, `challengeCtx`, challenge page load, accept/login handoff, challenge creation/share.

### Extract Later, Carefully

- Auth orchestration can become a dedicated session controller after startup routing is redesigned.
- Daily challenge fetch/countdown can move only after timer ownership and reset reloads have a new owner.
- Recording/playback can move only as one cohesive media controller, not piece by piece.
- YouTube player lifecycle can move with modal playback ownership, not as display code.
- Challenge accept/result flow can move after auth entry and modal scene entry have stable interfaces.

### Rewrite In New Architecture Instead Of Extracting

- Bootstrap/session routing.
- Modal/media/analyze coupling.
- Challenge URL/auth/score handoff.
- Dashboard hero/carousel/weak-words wrapper hooks.

These areas are easier to replace with explicit controllers than to keep shaving out of the legacy shell.

## Adapter Wrapper Map

These functions mostly bridge legacy `app.js` state to extracted modules:

- `renderCards`
- `renderDailyCard`
- `renderStreakCard`
- `showDailyComplete`
- `renderLevelBar`
- `getRecordingPlaybackDisplayRefs`
- `getSupportedMimeType`
- `stopRecordingCleanup`
- `startWaveform`
- `stopWaveform`
- `renderPhonemeBreakdown`
- `renderDivCard`
- `renderProgressDashboard`
- `renderPersonalBests`
- `showReplayLine`
- `hideReplayLine`
- `renderLeaderboard`
- `switchTab`
- `showChallengeResult`
- `updateLevelCardStats`
- `openLevelPanel`

Adapter wrappers should pass dependencies explicitly and should not begin owning new state.

## Runtime Invariants

Do not break these while migrating:

- Script order: all `window.MIRROR_*` modules must load before `static/app.js`.
- Boot order: challenge URL handling must run before normal token/app entry.
- Modal close priority: auth modal, then progress overlay, then scene modal.
- `activeScene` must be set before recording/analyze/playback work and cleared on modal close.
- Recording cleanup must stop recorder, stop mic tracks, clear recording timer, and stop waveform.
- Analyze 401 must clear auth, close scene modal, and return to auth screen.
- `showScore(data)` must run before post-score leaderboard/progress/card refresh.
- Challenge scoring must consume `challengeCtx` once and then clear it.
- Daily countdown reset must reload daily data and streak/profile display.
- Level bar render hook must still refresh level-card stats.
- Leaderboard render must preserve active tab or default to first scene.
- Level panel play button must close the panel before selecting/opening the first unlocked scene.

## High-Risk Seams

- Bootstrap/session entry: route order and auth timing are coupled.
- `selectScene`: app-vs-auth gating and modal opening meet here.
- `openModal`/`closeModal`: scene state, recording reset, YouTube, replay line, and body scroll meet here.
- Recording/playback: browser permission, recorder state, blob creation, playback state, timer, and waveform meet here.
- Analyze submit: auth expiry, upload, score display, post-score refresh, and level-up meet here.
- Challenge accept/result: URL entry, auth handoff, modal entry, and score result context meet here.
- Daily countdown: timer expiration triggers daily/streak reloads.

## Recommended Next Migration Order

1. Add characterization tests or browser smoke checks around boot, modal, recording, analyze, and challenge flows.
2. Stabilize app state access behind a small explicit state facade without changing behavior.
3. Replace dashboard hero/carousel/weak-words wrapper hooks with explicit startup calls.
4. Extract daily fetch/countdown only after timer ownership is specified.
5. Extract challenge flow only after auth and scene-entry contracts are explicit.
6. Extract recording/playback/analyze as a cohesive controller, or rewrite it in the new architecture.
7. Retire the legacy shell once boot/session routing has a replacement owner.

## Not Worth Extracting Further

- One-line DOM adapters that only pass refs to display domains.
- Tiny event bindings whose behavior is clearer in `app.js`.
- Bootstrap route checks before a new router exists.
- Modal close priority before overlays have a shared controller.
- Dashboard wrapper hooks if the next architecture will replace the dashboard composition.
