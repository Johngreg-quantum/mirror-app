const STANDALONE_NEW_SHELL_PREFIX = '/static/new-shell/';

function normalizePath(path) {
  const cleanPath = path || '/';
  return cleanPath.startsWith('/') ? cleanPath : `/${cleanPath}`;
}

export function getRoutingMode() {
  return window.location.pathname.startsWith(STANDALONE_NEW_SHELL_PREFIX) ? 'hash' : 'path';
}

export function isStandaloneNewShell() {
  return getRoutingMode() === 'hash';
}

export function createAppHref(path) {
  const normalizedPath = normalizePath(path);
  return getRoutingMode() === 'hash' ? `#${normalizedPath}` : normalizedPath;
}

export function getCurrentAppPath() {
  if (getRoutingMode() === 'hash') {
    const hashPath = window.location.hash.replace(/^#/, '');
    return normalizePath(hashPath || '/');
  }

  return `${normalizePath(window.location.pathname)}${window.location.search || ''}`;
}

export function navigateToAppPath(path, { replace = false } = {}) {
  const normalizedPath = normalizePath(path);

  if (getRoutingMode() === 'hash') {
    if (replace) {
      const nextUrl = `${window.location.pathname}${window.location.search || ''}#${normalizedPath}`;
      window.history.replaceState(null, '', nextUrl);
      return;
    }

    window.location.hash = normalizedPath;
    return;
  }

  if (replace) {
    window.history.replaceState(null, '', normalizedPath);
    return;
  }

  window.history.pushState(null, '', normalizedPath);
}
