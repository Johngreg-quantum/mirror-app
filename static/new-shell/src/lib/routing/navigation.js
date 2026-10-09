const STANDALONE_NEW_SHELL_PREFIX = '/static/new-shell/';

// The server mounts this shell twice. `/app`, `/app/daily`, `/app/scene/:id`
// and so on are the app's own URLs; `/scene/:id` and `/challenge/:id` are also
// served at the root because that is the shape of a link people share.
//
// Route paths in routes.js are written WITHOUT the mount prefix (`/daily`), so
// the two have to be reconciled in exactly one place, here:
//
//   reading  — strip the prefix when the browser is on an `/app` URL, so
//              `/app/daily` matches the `/daily` route. Without this a deep
//              link to any `/app/*` page matched nothing and silently rendered
//              the home page instead.
//   writing  — always emit the prefix, so every in-app link lands on a path the
//              server actually serves. Root-mounted pages used to link to
//              `/daily`, `/levels` and `/progress`, which are not routed at the
//              root at all: following the nav from a shared challenge link gave
//              a 404.
//
// Shared links are unaffected: they are built server-side by build_app_url()
// and still point at `/challenge/:id`, which keeps working.
const APP_MOUNT_PREFIX = '/app';

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

function isOnAppMount() {
  const { pathname } = window.location;
  return pathname === APP_MOUNT_PREFIX || pathname.startsWith(`${APP_MOUNT_PREFIX}/`);
}

export function createAppHref(path) {
  const normalizedPath = normalizePath(path);

  if (getRoutingMode() === 'hash') {
    return `#${normalizedPath}`;
  }

  // Home is the bare mount path: `/app/` would be a 307 to `/app` on every
  // click of the brand, since FastAPI redirects the trailing slash.
  return normalizedPath === '/' ? APP_MOUNT_PREFIX : `${APP_MOUNT_PREFIX}${normalizedPath}`;
}

export function getCurrentAppPath() {
  if (getRoutingMode() === 'hash') {
    const hashPath = window.location.hash.replace(/^#/, '');
    return normalizePath(hashPath || '/');
  }

  const search = window.location.search || '';

  if (isOnAppMount()) {
    // '/app' itself is the home route, not an empty path.
    const withoutPrefix = window.location.pathname.slice(APP_MOUNT_PREFIX.length);
    return `${normalizePath(withoutPrefix || '/')}${search}`;
  }

  return `${normalizePath(window.location.pathname)}${search}`;
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

  // Through createAppHref so pushed history entries and rendered links are the
  // same URLs; otherwise a push would leave the browser on a path the next read
  // cannot match.
  const target = createAppHref(normalizedPath);

  if (replace) {
    window.history.replaceState(null, '', target);
    return;
  }

  window.history.pushState(null, '', target);
}
