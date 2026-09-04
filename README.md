# Mirror v2

Mirror is an English learning app built around movie-scene practice.

Users can browse scenes, record themselves, play back their take, submit for scoring, track progress, and complete challenge flows.

## Live routes
- /
- /auth
- /levels
- /scene/:sceneId
- /progress
- /daily
- /challenge/:challengeId

## Rollback routes
- /legacy
- /legacy/challenge/:challengeId

## Checks
- Close buttons stay reachable: `npm i --no-save playwright && npx playwright install webkit`, run the app on port 8011, then `node scripts/verify-close-button.js`.

## Notes
- The new shell is now the primary app experience.
- Backend APIs remain the source of truth for auth, analyze, progress, daily, and challenge data.
- Legacy routes are still available temporarily as a rollback path.