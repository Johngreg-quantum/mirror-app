import { h } from '../lib/helpers/dom.js';
import { statusPill } from './primitives.js';

export function renderChallengeResultCard({ entry, result }) {
  if (!result) {
    return h('section', { className: 'ns-result-card ns-result-card--empty' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Challenge result' }),
        h('h3', { text: 'Beat the benchmark' }),
        h('p', { text: 'Record the linked scene, analyze the take, and Mirror will settle the head-to-head here.' }),
      ]),
      h('div', { className: 'ns-score-compare ns-score-compare--challenge-empty' }, [
        h('div', { className: 'ns-score-compare__item ns-score-compare__item--primary' }, [
          h('span', { text: 'Your score' }),
          h('strong', { text: '--' }),
        ]),
        h('div', { className: 'ns-score-compare__item' }, [
          h('span', { text: 'Score to beat' }),
          h('strong', { text: entry?.targetScoreLabel || '--' }),
        ]),
      ]),
      h('div', { className: 'ns-inline-list' }, [
        statusPill(entry?.targetScoreLabel || 'No benchmark'),
        statusPill(entry?.createdLabel || 'Awaiting challenge data'),
      ]),
    ]);
  }

  const isWin = result.outcome === 'won';

  return h('section', { className: `ns-result-card ns-result-card--${isWin ? 'win' : 'loss'}` }, [
    h('div', {
      className: `ns-result-card__outcome ns-result-card__outcome--${isWin ? 'win' : 'loss'}`,
      text: isWin ? 'Won' : 'Retry',
    }),
    h('div', { className: 'ns-result-card__intro' }, [
      h('p', { className: 'ns-eyebrow', text: 'Challenge result' }),
      h('h3', { text: isWin ? 'Benchmark beaten' : 'Within striking distance' }),
      h('p', { text: result.message }),
    ]),
    h('div', { className: 'ns-score-compare' }, [
      h('div', { className: 'ns-score-compare__item ns-score-compare__item--primary' }, [
        h('span', { text: 'Your score' }),
        h('strong', { text: result.yourScore }),
      ]),
      h('div', { className: 'ns-score-compare__item' }, [
        h('span', { text: 'Score to beat' }),
        h('strong', { text: result.opponentScore }),
      ]),
    ]),
    h('div', { className: 'ns-inline-list' }, [
      statusPill(`${result.pointsEarned} points`),
      statusPill(result.divisionName),
      statusPill(result.comparisonLabel),
      result.isDaily ? statusPill('Daily result') : null,
      result.isNewPersonalBest ? statusPill('New PB') : null,
    ]),
  ]);
}
