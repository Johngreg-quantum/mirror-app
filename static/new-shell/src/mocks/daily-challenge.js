import { featuredScene } from './scenes.js';

// Shaped like adaptDailyChallenge()'s output, with a half-finished set: the
// mock is the only place the "resume" states get exercised without a server.
export const mockDailyChallenge = {
  id: 'daily-2026-04-12',
  scene: featuredScene,
  lines: [
    { index: 1, sceneId: featuredScene.id, scene: featuredScene, done: true, score: 88 },
    { index: 2, sceneId: featuredScene.id, scene: featuredScene, done: false, score: null },
    { index: 3, sceneId: featuredScene.id, scene: featuredScene, done: false, score: null },
  ],
  lineTotal: 3,
  linesDone: 1,
  allDone: false,
  nextSceneId: featuredScene.id,
  progressLabel: '1 of 3 lines',
  resetLabel: 'New lines in 8h 14m',
  resetWarning: '',
  rewardPoints: 250,
  streakBonus: '2x on the full set',
  status: 'Resume the set',
};
