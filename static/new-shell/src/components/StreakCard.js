import { h } from '../lib/helpers/dom.js';
import { statusPill } from './primitives.js';

export function renderStreakCard({ profile }) {
  return h('section', { className: 'ns-streak-card' }, [
    h('p', { className: 'ns-eyebrow', text: 'Streak' }),
    h('strong', { text: `${profile.streakDays} days` }),
    h('p', { text: `${profile.displayName} is in ${profile.division} with ${profile.points.toLocaleString()} points.` }),
    h('div', { className: 'ns-inline-list' }, [
      statusPill(profile.dailyStatus),
      statusPill(profile.levelTitle),
    ]),
  ]);
}
