import { h } from '../lib/helpers/dom.js';
import { createAppHref } from '../lib/routing/navigation.js';
import { buttonLink, card, statusPill } from './primitives.js';

export function getSessionLabel(session) {
  if (session?.status === 'authenticated') {
    return session.user?.displayName || 'Signed in';
  }

  if (session?.status === 'loading' || session?.status === 'unknown') {
    return 'Checking session';
  }

  if (session?.status === 'error') {
    return 'Session check failed';
  }

  return 'Signed out';
}

export function renderSessionPrompt({
  session,
  title = 'Sign in to Mirror',
  body = 'Sign in to sync progress, streaks, unlocks, and personalized scene data.',
  onLogout,
} = {}) {
  const isAuthenticated = session?.status === 'authenticated';
  const isError = session?.status === 'error';
  let action = buttonLink({ href: createAppHref('/auth'), text: 'Open auth', variant: 'secondary' });

  if (isAuthenticated) {
    action = onLogout
      ? h('button', {
          className: 'ns-button ns-button--secondary',
          type: 'button',
          on: {
            click: async (event) => {
              const button = event.currentTarget;
              button.disabled = true;
              button.textContent = 'Signing out...';
              try {
                await onLogout();
              } finally {
                button.disabled = false;
                button.textContent = 'Sign out';
              }
            },
          },
          text: 'Sign out',
        })
      : null;
  }

  const promptTitle = isAuthenticated
    ? `Signed in as ${getSessionLabel(session)}`
    : isError ? 'Session refresh failed' : title;
  const promptBody = isAuthenticated
    ? 'Personalized progress and streak data are active. Sign out clears this browser session.'
    : isError
      ? session.error?.message || 'Mirror could not refresh your session.'
      : body;

  return card({
    title: promptTitle,
    body: promptBody,
    children: [
      h('div', { className: 'ns-inline-list' }, [
        statusPill(session?.status || 'unknown'),
        session?.hasToken ? statusPill('session saved') : statusPill('no session'),
        session?.error?.rateLimited ? statusPill('Rate limited') : null,
        action,
      ]),
    ],
  });
}
