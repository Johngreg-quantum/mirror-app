import { createAppLayout } from './components/AppLayout.js';
import { createRouter } from './router/router.js';
import { routes } from './router/routes.js';
import {
  clearChallengeResults,
  createInitialAppState,
  ensureChallengeState,
  ensureReadCacheState,
  resetPostScoreState,
} from './state/app-state.js';
import { createSessionStore } from './state/session-store.js';

export function createAppShell({ root }) {
  const appState = createInitialAppState();
  const sessionStore = createSessionStore();
  let initialSessionSettled = false;
  let activeRouteId = null;
  let lastSessionUserId = null;
  let resolveInitialSession;
  const initialSessionReady = new Promise((resolve) => {
    resolveInitialSession = resolve;
  });
  const sessionActions = {
    refreshSession: sessionStore.refreshSession,
    loginWithLegacy: sessionStore.loginWithLegacy,
    registerWithLegacy: sessionStore.registerWithLegacy,
    logoutWithLegacy: sessionStore.logoutWithLegacy,
    waitForInitialSession: () => initialSessionReady,
  };
  const layout = createAppLayout({ routes, sessionActions });

  appState.session = sessionStore.getSnapshot();
  ensureReadCacheState(appState);
  ensureChallengeState(appState);

  root.replaceChildren(layout.root);
  const appActions = {
    session: sessionActions,
    navigation: {
      go(path) {
        router.go(path);
      },
    },
  };

  const router = createRouter({
    routes,
    outlet: layout.outlet,
    appState,
    actions: appActions,
    onRouteChange: (routeId) => {
      activeRouteId = routeId;
      layout.setActiveRoute(routeId);
    },
  });

  layout.setSession(appState.session);

  sessionStore.subscribe((session) => {
    const nextUserId = session?.user?.id || null;

    if (session.status !== 'authenticated' || lastSessionUserId !== nextUserId) {
      resetPostScoreState(appState);
      clearChallengeResults(appState);
    }

    lastSessionUserId = nextUserId;
    appState.session = session;
    layout.setSession(session);

    if (activeRouteId === 'home' && !initialSessionSettled) {
      return;
    }

    router.refresh();
  });

  router.start();
  const initialSessionLoad = sessionStore.load();
  initialSessionLoad.finally(() => {
    initialSessionSettled = true;
    resolveInitialSession(sessionStore.getSnapshot());
  });

  return {
    appState,
    router,
    sessionStore,
  };
}
