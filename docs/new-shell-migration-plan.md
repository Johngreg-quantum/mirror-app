# New Shell Migration Plan

This plan describes how to rebuild the app in a parallel architecture while the legacy app remains the source of truth. Do not cut over a feature until the new shell has verified parity for that feature.

## Target Architecture

- App shell
  - Owns route selection, layout chrome, authenticated app frame, overlays, and page-level composition.
- API layer
  - Small request functions for auth, scene config, progress, daily challenge, profile/streak, leaderboard, history, submit score, and challenges.
  - No direct DOM work.
- State layer
  - Session state, scene config state, progress state, recording state, modal state, challenge state, and dashboard state.
  - State updates should be explicit and observable by UI modules.
- Feature modules
  - Auth, scene browser, scene modal, recording/playback, analyze/score, progress dashboard, level panel, daily challenge, leaderboard, challenge entry/result.
- UI components
  - Render-only components with props/callbacks.
  - No fetch, storage, media, timers, or global mutation inside display components.
- Runtime services
  - Media recorder service, YouTube player service, countdown/timer service, clipboard/share service.
  - These replace the fragile legacy ownership points instead of extracting them directly.

## Route/Page Map

- `/`
  - Logged out: landing/auth entry.
  - Logged in: authenticated app shell with dashboard, daily surfaces, scene browser, leaderboard, and progress entry.
- `/challenge/:id`
  - Challenge entry page.
  - If authenticated: allow accept and scene entry.
  - If unauthenticated: show challenge summary and auth handoff.
- Modal routes or shell overlays
  - Scene modal.
  - Auth modal.
  - Progress dashboard.
  - Level panel.

Keep the legacy URL behavior as the reference until routing parity is verified.

## Shared State Map

- Session
  - Token, user, auth status, onboarding status.
  - Legacy source: `static/app.js`, auth orchestration.
- Scene config
  - Scene records, level map, level metadata, default unlocked scenes.
  - Legacy source: `static/app.js`, `static/app-config.js`.
- Progress
  - Level, best scores, unlocked scenes, next level.
  - Legacy source: `static/app.js`, `static/app-state-helpers.js`.
- Daily challenge
  - Daily scene, seconds until reset, streak/profile status.
  - Legacy source: `static/app.js`, `static/daily-challenge-domain.js`.
- Scene modal
  - Active scene, modal visibility, YouTube metadata.
  - Legacy source: `static/app.js`, `static/scene-modal-domain.js`.
- Recording/playback
  - Recorder instance, mic stream, audio chunks/blob, playback audio, recording timer.
  - Legacy source: `static/app.js`, `static/recording-playback-domain.js`, `static/app-runtime-utils.js`.
- Analyze/score
  - Submit state, score payload, phoneme breakdown, PB, points, challenge result context.
  - Legacy source: `static/app.js`, `static/analyze-score-domain.js`.
- Challenge
  - Active challenge, challenge context, challenge share result.
  - Legacy source: `static/app.js`.
- Dashboard/progress
  - History, profile, division, personal bests, weak words.
  - Legacy source: `static/app.js`, `static/progress-dashboard-domain.js`.

## Feature/Module Map From Legacy To New Shell

- App shell/layout
  - Legacy source: `index.html`, `static/app.js`.
  - Future module: `AppShell`, `AuthenticatedLayout`, `LandingLayout`.
  - Rewrite cleanly.
- Auth screen/modal
  - Legacy source: `static/app.js`, `static/auth-modal-domain.js`.
  - Future module: `AuthController`, `AuthModal`, `AuthForms`.
  - Reuse copy/behavior, rewrite state flow.
- Scene browser
  - Legacy source: `static/app.js`, `static/scene-browser-domain.js`, `static/app-progress-orchestration.js`.
  - Future module: `SceneBrowser`, `SceneCardGrid`, `LevelProgressBar`.
  - Rebuild early because many flows depend on scene selection.
- Scene modal shell
  - Legacy source: `static/app.js`, `static/scene-modal-domain.js`.
  - Future module: `SceneModalController`, `SceneModal`.
  - Rewrite around explicit modal state.
- Recording/playback shell
  - Legacy source: `static/app.js`, `static/recording-playback-domain.js`, `static/app-runtime-utils.js`.
  - Future module: `RecordingController`, `PlaybackController`, `MediaRecorderService`.
  - Rewrite cleanly; do not extract more legacy pieces.
- Analyze/score shell
  - Legacy source: `static/app.js`, `static/analyze-score-domain.js`, `static/app-post-score-orchestration.js`.
  - Future module: `AnalyzeController`, `ScorePanel`, `PostScoreRefresh`.
  - Keep submit semantics identical.
- Progress dashboard
  - Legacy source: `static/app.js`, `static/progress-dashboard-domain.js`.
  - Future module: `ProgressDashboard`, `DivisionCard`, `PersonalBests`, `HistoryList`.
  - Can rebuild after auth/scene/progress state exists.
- Level panel
  - Legacy source: `static/app.js`, `static/level-panel-domain.js`, `static/app-level-panel-orchestration.js`.
  - Future module: `LevelPanel`, `LevelCollectionCards`.
  - Rebuild after scene browser and progress state.
- Daily challenge surfaces
  - Legacy source: `static/app.js`, `static/daily-challenge-domain.js`.
  - Future module: `DailyChallengeController`, `DailyCard`, `StreakCard`, `DailyCountdown`.
  - Rebuild UI early, timer ownership later.
- Leaderboard
  - Legacy source: `static/app.js`, `static/app-leaderboard-orchestration.js`, `static/app-view-helpers.js`.
  - Future module: `Leaderboard`, `LeaderboardTabs`.
  - Rebuild after scene config is stable.
- Challenge entry/result surfaces
  - Legacy source: `static/app.js`, `static/analyze-score-domain.js`.
  - Future module: `ChallengeEntryPage`, `ChallengeAcceptController`, `ChallengeResult`.
  - Defer until auth, scene modal, and analyze score are stable.

## Recommended Build Order

1. Build the new app shell and API client
  - Add route skeletons, layout shell, request helpers, and a feature flag or parallel entry point.
  - Do not replace legacy routes yet.
2. Build session/auth state
  - Implement token restore, login, register, logout, and auth modal/page display.
  - Verify against legacy auth behavior.
3. Build scene config and progress state
  - Load scene config and progress.
  - Derive level map and default unlocked scenes.
4. Build scene browser and level progress bar
  - Render scene cards, lock state, daily badge, PB badges, and level bar.
5. Build scene modal shell
  - Open/close modal, set active scene, show daily badge, quote/title/year/video placeholder.
  - Keep media controls disabled or stubbed until recording is ready.
6. Build recording/playback controller
  - Implement mic permission, recorder lifecycle, timer, waveform, stop/cleanup, local playback.
7. Build analyze/score flow
  - Submit audio, handle 401, render score, phonemes, PB, points, daily completion, challenge result hook.
8. Build post-score refresh
  - Refresh leaderboard, progress, cards, and level-up display after score.
9. Build progress dashboard and leaderboard
  - Render history, division, PB list, leaderboard tabs/panels.
10. Build level panel
  - Render level cards/panel and first-unlocked scene button.
11. Build daily challenge controller
  - Daily load, streak load, countdown ownership, reset reloads.
12. Build challenge entry/share flow
  - Challenge URL entry, auth handoff, accept, share link creation, result context.
13. Cutover and retire legacy shell
  - Route by route, after parity checks pass.

## Verification Strategy Per Phase

- Shell/API phase
  - Verify route render, API base resolution, and script independence.
- Auth phase
  - Verify login/register errors, token restore, logout, onboarding timing, and challenge login handoff.
- Scene/progress phase
  - Verify unlocked scenes, default unlocked fallback, PB badges, level bar text/fill, and level-card stats.
- Modal phase
  - Verify open/close, Escape priority, backdrop close, daily badge, video placeholder/frame state.
- Recording/playback phase
  - Verify mic denial, no mic found, recording timer, empty blob handling, stop cleanup, playback ended state.
- Analyze/score phase
  - Verify submit payload, 401 path, score display, phoneme breakdown, PB banner, points panel, daily completion.
- Post-score refresh phase
  - Verify leaderboard tab selection, progress reload, card refresh, and level-up toast order.
- Dashboard/leaderboard phase
  - Verify history empty/error/loading, division card, PB list, tab switching.
- Daily/challenge phase
  - Verify countdown reset reloads, streak display, challenge URL entry, accept flow, share/copy, result win/loss.

## Legacy Parity Checklist

- Boot order matches legacy.
- Challenge URLs are handled before normal app entry.
- Auth modal close/backdrop/Escape behavior matches legacy.
- Modal close priority remains auth, progress, scene modal.
- Active scene is set before recording/analyze and cleared on scene modal close.
- Recording cleanup stops recorder, mic tracks, timer, and waveform.
- Analyze 401 clears auth, closes modal, and returns to auth.
- Score renders before post-score refresh.
- Challenge result consumes `challengeCtx` once.
- Daily countdown reset reloads daily and streak.
- Level bar refresh still updates level-card stats.
- Leaderboard defaults to first scene if active tab is invalid.
- Level panel play closes panel before selecting scene.
- Logged-out scene selection opens registration.

## Cutover Strategy

- Run legacy and new shell in parallel behind a route, flag, or local-only entry.
- Keep legacy data APIs unchanged during the first rebuild.
- Compare DOM-visible behavior feature by feature.
- Cut over low-risk pages first: auth display, scene browser, leaderboard.
- Cut over modal/recording/analyze only after browser smoke checks exist.
- Keep challenge URLs on legacy until auth, scene modal, analyze, and result parity are proven.
- Remove legacy modules only after the new shell owns all routes and invariants.

## What Stays Legacy Until The End

- Bootstrap/session route decision.
- Challenge URL accept/login/result flow.
- Scene modal lifecycle with recording/analyze coupling.
- Media recorder ownership.
- YouTube player ownership.
- Analyze submit and 401 auth handoff.
- Daily countdown reset ownership.
- Dashboard wrapper hooks for hero/carousel/weak words.

## Do Not Migrate Yet

- `openModal`/`closeModal`
  - Too much modal, media, YouTube, replay, scroll, and active scene coupling.
- `startRec`/`stopRec`/`togglePlayback`
  - Browser media behavior needs a cohesive controller rewrite.
- `analyze`
  - Upload, auth expiry, score rendering, and post-score refresh are still tightly coupled.
- `loadChallengePage`/`enterChallengeFromAuth`
  - Challenge routing depends on auth and scene modal contracts.
- `startDailyCountdown`
  - Reset behavior triggers app data reloads and needs explicit timer ownership.

## Rewrite Cleanly Rather Than Extract Further

- Bootstrap/router.
- Overlay manager and modal close priority.
- Recording/playback media controller.
- Analyze submit controller.
- Challenge controller.
- Daily countdown controller.
- Dashboard composition hooks.
