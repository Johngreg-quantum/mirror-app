import { renderErrorState, renderLoadingState } from '../components/AsyncState.js';
import { buttonLink, card, statusPill } from '../components/primitives.js';
import { h, mount } from '../lib/helpers/dom.js';
import { getCurrentAppPath, getRoutingMode, navigateToAppPath } from '../lib/routing/navigation.js';
import { getRouteFallbackConfig, isRouteEnabled, ROUTE_PROMOTION_STATUS } from './route-readiness.js';
import { renderSessionPrompt } from '../components/SessionState.js';

function normalizePath(path) {
  const cleanPath = path || '/';
  return cleanPath.startsWith('/') ? cleanPath : `/${cleanPath}`;
}

function splitPathAndQuery(path) {
  const [pathPart, queryPart = ''] = String(path || '/').split('?');
  return {
    path: normalizePath(pathPart),
    query: Object.fromEntries(new URLSearchParams(queryPart)),
  };
}

function matchRoute(routePath, currentPath) {
  const routeParts = normalizePath(routePath).split('/').filter(Boolean);
  const currentParts = normalizePath(currentPath).split('/').filter(Boolean);

  if (routeParts.length !== currentParts.length) {
    return null;
  }

  const params = {};

  for (let index = 0; index < routeParts.length; index += 1) {
    const routePart = routeParts[index];
    const currentPart = currentParts[index];

    if (routePart.startsWith(':')) {
      params[routePart.slice(1)] = decodeURIComponent(currentPart);
      continue;
    }

    if (routePart !== currentPart) {
      return null;
    }
  }

  return params;
}

function resolveRoute(routes, currentPath) {
  for (const route of routes) {
    const params = matchRoute(route.path, currentPath);

    if (params) {
      return { route, params };
    }
  }

  return { route: routes[0], params: {} };
}

function findMatchedRoute(routes, currentPath) {
  for (const route of routes) {
    const params = matchRoute(route.path, currentPath);

    if (params) {
      return { route, params };
    }
  }

  return null;
}

export function createRouter({ routes, outlet, appState, actions = {}, onRouteChange }) {
  let renderVersion = 0;
  let cleanupCallbacks = [];
  const routingMode = getRoutingMode();

  function runCleanup() {
    cleanupCallbacks.forEach((cleanup) => {
      try {
        cleanup();
      } catch {
        // Route cleanup should not block the next render.
      }
    });
    cleanupCallbacks = [];
  }

  function renderProtectedReadState(route) {
    const session = appState.session;

    if (session.status === 'unknown' || session.status === 'loading') {
      return renderLoadingState(`Checking session for ${route.label}`);
    }

    if (session.status === 'error') {
      return renderErrorState(session.error, { title: `${route.label} session check failed` });
    }

    if (session.status !== 'authenticated') {
      return renderSessionPrompt({
        session,
        title: `${route.label} needs sign-in`,
        body: 'Sign in to load your personalized practice data, then return here.',
        onLogout: actions.session?.logoutWithLegacy,
      });
    }

    return null;
  }

  function renderLegacyFallbackState(route, params) {
    const fallback = getRouteFallbackConfig(route, { params });
    const isLegacyOnly = fallback.status === ROUTE_PROMOTION_STATUS.LEGACY_ONLY;
    const routeStatusLabel = isLegacyOnly
      ? 'Rollback'
      : fallback.status === ROUTE_PROMOTION_STATUS.PRIMARY
        ? 'Rollback path'
        : 'Fallback';

    return h('article', { className: 'ns-page' }, [
      card({
        title: fallback.fallbackTitle,
        body: fallback.fallbackBody,
        children: [
          h('div', { className: 'ns-inline-list' }, [
            statusPill(routeStatusLabel),
            statusPill(route.id),
            statusPill(fallback.legacyPath),
            buttonLink({
              href: fallback.legacyPath,
              text: fallback.legacyActionText,
              variant: 'secondary',
            }),
          ]),
          h('p', {
            className: 'ns-muted',
            text: 'Use the rollback path below if this page is temporarily unavailable.',
          }),
        ],
      }),
    ]);
  }

  function renderCurrentRoute() {
    runCleanup();
    renderVersion += 1;
    const currentRenderVersion = renderVersion;
    const currentPath = getCurrentAppPath();
    const { path, query } = splitPathAndQuery(currentPath);
    const { route, params } = resolveRoute(routes, path);
    const fallbackPage = isRouteEnabled(route) ? null : renderLegacyFallbackState(route, params);
    const blockedPage = fallbackPage || (route.protectedRead ? renderProtectedReadState(route) : null);
    const onCleanup = (cleanup) => {
      if (currentRenderVersion !== renderVersion) {
        try {
          cleanup();
        } catch {
          // Late cleanup after a route change should stay isolated.
        }
        return;
      }

      cleanupCallbacks.push(cleanup);
    };
    const page = blockedPage || route.render({ appState, params, path, query, actions, onCleanup });

    mount(outlet, page);
    onRouteChange?.(route.id);
  }

  function handleDocumentNavigation(event) {
    if (routingMode !== 'path') {
      return;
    }

    if (
      event.defaultPrevented
      || event.button !== 0
      || event.metaKey
      || event.ctrlKey
      || event.shiftKey
      || event.altKey
    ) {
      return;
    }

    const anchor = event.target?.closest?.('a[href]');

    if (!anchor || anchor.target === '_blank' || anchor.hasAttribute('download')) {
      return;
    }

    const url = new URL(anchor.href, window.location.origin);

    if (url.origin !== window.location.origin) {
      return;
    }

    const matched = findMatchedRoute(routes, url.pathname);

    if (!matched) {
      return;
    }

    const { route } = matched;

    if (!route || route.readiness?.status === ROUTE_PROMOTION_STATUS.LEGACY_ONLY) {
      return;
    }

    event.preventDefault();
    navigateToAppPath(`${url.pathname}${url.search || ''}`);
    renderCurrentRoute();
  }

  window.addEventListener(routingMode === 'hash' ? 'hashchange' : 'popstate', renderCurrentRoute);

  if (routingMode === 'path') {
    document.addEventListener('click', handleDocumentNavigation);
  }

  return {
    start() {
      renderCurrentRoute();
    },
    go(path) {
      navigateToAppPath(path);
      renderCurrentRoute();
    },
    refresh() {
      renderCurrentRoute();
    },
  };
}
