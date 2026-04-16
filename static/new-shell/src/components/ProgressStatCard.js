import { h } from '../lib/helpers/dom.js';

export function renderProgressStatCard({ label, value, detail }) {
  return h('article', { className: 'ns-stat-card' }, [
    h('span', { text: label }),
    h('strong', { text: value }),
    h('p', { text: detail }),
  ]);
}
