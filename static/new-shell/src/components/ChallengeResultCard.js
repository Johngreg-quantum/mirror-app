import { h } from '../lib/helpers/dom.js';
import { statusPill } from './primitives.js';

export function renderChallengeResultCard({ entry, result }) {
  if (!result) {
    return h('section', { className: 'ns-result-card' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Challenge result' }),
        h('h3', { text: 'No scored take yet' }),
        h('p', { text: 'Take the challenge scene and submit a scored take to compare against the current benchmark.' }),
      ]),
      h('div', { className: 'ns-inline-list' }, [
        statusPill(entry?.targetScoreLabel || 'No benchmark'),
        statusPill(entry?.createdLabel || 'Awaiting challenge data'),
      ]),
    ]);
  }

  return h('section', { className: 'ns-result-card' }, [
    h('div', {}, [
      h('p', { className: 'ns-eyebrow', text: 'Challenge result' }),
      h('h3', { text: `${result.title} against ${entry.challengerName}` }),
      h('p', { text: result.message }),
    ]),
    h('div', { className: 'ns-score-compare' }, [
      h('div', {}, [
        h('span', { text: 'You' }),
        h('strong', { text: result.yourScore }),
      ]),
      h('div', {}, [
        h('span', { text: entry.challengerName }),
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
