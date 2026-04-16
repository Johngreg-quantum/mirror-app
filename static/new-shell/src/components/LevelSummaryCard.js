import { h } from '../lib/helpers/dom.js';
import { buttonLink, statusPill } from './primitives.js';
import { sceneHref } from '../lib/routing/scene-routes.js';

export function renderLevelSummaryCard({ level }) {
  const progressLabel = `${level.unlockedScenes}/${level.totalScenes} scenes`;

  return h('article', { className: `ns-level-card ns-level-card--${level.status}` }, [
    h('div', { className: 'ns-section-heading' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: `Level ${level.level}` }),
        h('h3', { text: level.title }),
      ]),
      statusPill(level.status),
    ]),
    h('p', { text: level.description }),
    h('div', { className: 'ns-meter', attrs: { 'aria-label': progressLabel } }, [
      h('span', {
        attrs: {
          style: `width: ${(level.unlockedScenes / level.totalScenes) * 100}%`,
        },
      }),
    ]),
    h('div', { className: 'ns-inline-list' }, [
      statusPill(progressLabel),
      statusPill(`Target ${level.requiredScore}+`),
    ]),
    buttonLink({
      href: sceneHref(level.firstUnlockedSceneId, { from: 'levels' }),
      text: level.status === 'locked' ? 'View first locked scene' : 'Enter first scene',
      variant: level.status === 'locked' ? 'secondary' : 'primary',
    }),
  ]);
}
