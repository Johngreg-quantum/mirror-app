import { h } from '../lib/helpers/dom.js';
import { buttonLink, statusPill } from './primitives.js';
import { sceneHref } from '../lib/routing/scene-routes.js';

// One row per line of the Daily Take. Each row is its own entry point, so a
// half-finished set can be resumed from whichever line the user wants rather
// than only from the next one in order.
function renderLineRow(line) {
  return h('li', {
    className: `ns-daily-line${line.done ? ' ns-daily-line--done' : ''}`,
  }, [
    h('span', { className: 'ns-daily-line__index', text: String(line.index) }),
    h('div', { className: 'ns-daily-line__body' }, [
      h('p', { className: 'ns-daily-line__film', text: line.scene?.film || line.sceneId }),
      h('blockquote', { text: line.scene?.quote || '' }),
    ]),
    line.done
      ? statusPill(line.score === null ? 'Done' : `Done · ${line.score}%`)
      : buttonLink({
          href: sceneHref(line.sceneId, { from: 'daily' }),
          text: 'Record',
        }),
  ]);
}

export function renderDailyChallengeCard({ daily }) {
  const lines = Array.isArray(daily.lines) ? daily.lines : [];

  return h('section', { className: 'ns-daily-card' }, [
    h('img', {
      className: 'ns-daily-card__image',
      src: daily.scene?.imageUrl || '/static/beginner-card.webp',
      alt: `${daily.scene?.film || 'Daily'} scene still reference`,
    }),
    h('div', { className: 'ns-daily-card__body' }, [
      h('p', { className: 'ns-eyebrow', text: 'Daily Take' }),
      h('h2', { text: lines.length > 1 ? 'Three short lines' : (daily.scene?.title || 'Daily') }),
      h('div', { className: 'ns-inline-list' }, [
        statusPill(daily.progressLabel),
        statusPill(daily.status),
        statusPill(daily.resetLabel),
        statusPill(daily.streakBonus),
      ]),
      daily.resetWarning
        ? h('p', { className: 'ns-daily-card__warning', text: daily.resetWarning })
        : null,
      lines.length
        ? h('ol', { className: 'ns-daily-lines' }, lines.map(renderLineRow))
        : null,
      daily.allDone
        ? h('p', {
            className: 'ns-daily-card__done',
            text: 'Set complete — your streak is counted for today.',
          })
        : buttonLink({
            href: sceneHref(daily.nextSceneId, { from: 'daily' }),
            text: daily.linesDone > 0 ? 'Next line' : 'Start the Daily Take',
          }),
    ]),
  ]);
}
