import { renderAuthPage } from '../features/auth/AuthPage.js';
import { renderChallengePage } from '../features/challenge/ChallengePage.js';
import { renderDailyChallengePage } from '../features/daily/DailyChallengePage.js';
import { renderHomePage } from '../features/home/HomePage.js';
import { renderLevelPanelPage } from '../features/level-panel/LevelPanelPage.js';
import { renderProgressDashboardPage } from '../features/progress/ProgressDashboardPage.js';
import { renderSceneDetailPage } from '../features/scenes/SceneDetailPage.js';
import { withRouteReadiness } from './route-readiness.js';

const baseRoutes = [
  {
    id: 'home',
    path: '/',
    label: 'Scenes',
    navPath: '/',
    nav: true,
    render: renderHomePage,
  },
  {
    id: 'auth',
    path: '/auth',
    label: 'Sign in',
    navPath: '/auth',
    nav: true,
    render: renderAuthPage,
  },
  {
    id: 'scene-detail',
    path: '/scene/:sceneId',
    label: 'Scene detail',
    navPath: '/scene/network-monologue',
    nav: true,
    render: renderSceneDetailPage,
  },
  {
    id: 'progress',
    path: '/progress',
    label: 'Progress',
    navPath: '/progress',
    nav: true,
    protectedRead: true,
    render: renderProgressDashboardPage,
  },
  {
    id: 'levels',
    path: '/levels',
    label: 'Levels',
    navPath: '/levels',
    nav: true,
    protectedRead: true,
    render: renderLevelPanelPage,
  },
  {
    id: 'daily',
    path: '/daily',
    label: 'Daily',
    navPath: '/daily',
    nav: true,
    render: renderDailyChallengePage,
  },
  {
    id: 'challenge',
    path: '/challenge/:challengeId',
    label: 'Challenge',
    navPath: '/challenge/sample-challenge',
    nav: true,
    render: renderChallengePage,
  },
];

export const routes = baseRoutes.map((route) => withRouteReadiness(route));
