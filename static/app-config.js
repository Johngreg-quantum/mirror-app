window.MIRROR_FRONTEND_CONFIG = {
  LEVEL_NAMES: { 1: 'Beginner', 2: 'Intermediate', 3: 'Advanced' },
  LEVEL_UI_META: {
    1: { label: 'Beginner', cls: 'beg', desc: 'Short, clear lines. Get comfortable speaking on camera.' },
    2: { label: 'Intermediate', cls: 'int', desc: 'Longer phrases, rhythm, and emotion start to matter.' },
    3: { label: 'Advanced', cls: 'adv', desc: 'Accent precision and raw delivery. The real challenge begins.' },
  },
  DIVISIONS: [
    { name: 'Bronze', min: 0, max: 499, color: '#cd7f32' },
    { name: 'Silver', min: 500, max: 1999, color: '#b8b8b8' },
    { name: 'Gold', min: 2000, max: 4999, color: '#c9a84c' },
    { name: 'Diamond', min: 5000, max: 9999, color: '#67e8f9' },
    { name: 'Director', min: 10000, max: null, color: '#c9a84c' },
  ],
  WAVE_BARS: 48,
};
