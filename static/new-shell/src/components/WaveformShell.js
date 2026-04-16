import { h } from '../lib/helpers/dom.js';

const BAR_COUNT = 32;

export function createWaveformShell() {
  const bars = Array.from({ length: BAR_COUNT }, (_, index) => h('span', {
    attrs: { style: `--ns-wave-index: ${index}` },
  }));

  function update(state) {
    const active = state.status === 'recording' || state.status === 'playing';
    const progress = state.durationMs
      ? Math.min(1, state.elapsedMs / state.durationMs)
      : active ? 1 : 0;
    const activeBars = Math.max(1, Math.round(progress * BAR_COUNT));

    bars.forEach((bar, index) => {
      const isActive = active && index < activeBars;
      const baseHeight = 18 + ((index * 17) % 44);
      const liveLift = isActive ? Math.round((state.level || 0) * 34) : 0;

      bar.style.height = `${baseHeight + liveLift}px`;
      bar.classList.toggle('is-active', isActive);
    });
  }

  return {
    root: h('div', {
      className: 'ns-waveform-shell',
      attrs: { 'aria-label': 'Local recording waveform' },
    }, bars),
    update,
  };
}
