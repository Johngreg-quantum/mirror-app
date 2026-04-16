import { h } from '../lib/helpers/dom.js';
import { card, statusPill } from './primitives.js';

export function renderLoadingState(label = 'Loading live read-only data') {
  return card({
    title: label,
    body: 'Fetching from the existing API without taking ownership of writes or runtime flows.',
    children: [statusPill('Read-only')],
  });
}

export function renderEmptyState({ title = 'No data yet', body = 'This surface is ready for live data once records exist.' } = {}) {
  return card({
    title,
    body,
    children: [statusPill('Empty')],
  });
}

export function renderErrorState(error, { title = 'Live data unavailable' } = {}) {
  const isRateLimited = Boolean(error?.rateLimited);

  return h('section', { className: 'ns-card ns-state-card' }, [
    h('p', {
      className: 'ns-eyebrow',
      text: isRateLimited ? 'Rate limited' : error?.authRequired ? 'Auth required' : 'Read-only fetch failed',
    }),
    h('h3', { text: title }),
    h('p', { text: error?.message || 'Mirror could not load this surface right now.' }),
    h('div', { className: 'ns-inline-list' }, [
      statusPill(error?.status ? `Status ${error.status}` : 'Offline'),
      isRateLimited ? statusPill(`Attempts ${error.attempts || 1}`) : null,
      isRateLimited && error?.retryAfterMs ? statusPill(`Retry after ~${Math.ceil(error.retryAfterMs / 1000)}s`) : null,
    ]),
  ]);
}
