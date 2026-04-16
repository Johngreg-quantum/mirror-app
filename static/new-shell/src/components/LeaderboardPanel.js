import { h } from '../lib/helpers/dom.js';
import { buttonLink, statusPill } from './primitives.js';
import { sceneHref } from '../lib/routing/scene-routes.js';

export function renderLeaderboardPanel({ leaderboard, entrySource = 'leaderboard' }) {
  return h('section', { className: 'ns-leaderboard' }, [
    h('div', { className: 'ns-section-heading' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Scene leaderboard' }),
        h('h3', { text: leaderboard.title }),
      ]),
      h(
        'div',
        { className: 'ns-inline-list' },
        leaderboard.tabs.map((tab) => statusPill(tab)),
      ),
    ]),
    leaderboard.activeSceneId
      ? buttonLink({
          href: sceneHref(leaderboard.activeSceneId, { from: entrySource }),
          text: `Open ${leaderboard.title}`,
          variant: 'secondary',
        })
      : null,
    h(
      'ol',
      { className: 'ns-leaderboard__rows' },
      leaderboard.rows.map((row) => h('li', { className: 'ns-leaderboard__row' }, [
        h('span', { className: 'ns-leaderboard__rank', text: row.rank }),
        h('strong', { text: row.name }),
        h('span', { text: row.note }),
        h('b', { text: row.score }),
      ])),
    ),
  ]);
}
