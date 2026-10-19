import { h } from '../lib/helpers/dom.js';

// The scene player. Replaces the YouTube embed that used to sit in this slot:
// a scene is an mp3 we generated and serve ourselves, so there is no iframe, no
// third-party script and no `rel=0` negotiation -- just an <audio> element.
//
// The waveform is drawn, not analysed. Decoding every mp3 through the Web Audio
// API to read real peaks costs a fetch and a decode per scene and fails behind
// a strict CSP; bar heights come from a hash of the scene id instead, so a
// given scene always draws the same shape and playback only moves the fill.

const BAR_COUNT = 48;

function barHeights(sceneId) {
  let seed = 0;
  for (let i = 0; i < String(sceneId).length; i += 1) {
    seed = (seed * 31 + String(sceneId).charCodeAt(i)) >>> 0;
  }
  seed = seed || 1;
  const next = () => {
    seed ^= seed << 13; seed >>>= 0;
    seed ^= seed >> 17;
    seed ^= seed << 5;  seed >>>= 0;
    return seed / 4294967296;
  };
  return Array.from({ length: BAR_COUNT }, (_, i) => {
    // Taper the ends so it reads as a spoken phrase rather than a block.
    const edge = Math.sin(((i + 0.5) / BAR_COUNT) * Math.PI);
    return Math.max(8, Math.round((0.22 + next() * 0.78) * edge * 100));
  });
}

function formatTime(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

export function renderSceneAudioPlayer(scene, onCleanup) {
  const sceneId = scene?.id || '';
  const url = scene?.audioUrl || '';

  const bars = barHeights(sceneId).map((height) =>
    h('div', { className: 'ns-sa__bar', attrs: { style: `height:${height}%` } }));
  const wave = h('div', { className: 'ns-sa__wave' }, bars);
  const time = h('span', { className: 'ns-sa__time', text: formatTime(scene?.audioSeconds) });
  const button = h('button', {
    className: 'ns-sa__play',
    text: url ? 'Play' : 'No audio',
    attrs: url ? {} : { disabled: '' },
  });

  const root = h('div', { className: 'ns-sa' }, [button, wave, time]);
  if (!url) return root;

  const audio = new Audio(url);
  audio.preload = 'metadata';

  const paint = (ratio) => {
    const upto = Math.round(ratio * bars.length);
    bars.forEach((bar, i) => bar.classList.toggle('is-played', i < upto));
  };

  audio.addEventListener('loadedmetadata', () => {
    if (Number.isFinite(audio.duration)) time.textContent = formatTime(audio.duration);
  });
  audio.addEventListener('timeupdate', () => {
    const d = audio.duration;
    if (Number.isFinite(d) && d > 0) {
      paint(audio.currentTime / d);
      time.textContent = formatTime(d - audio.currentTime);
    }
  });
  audio.addEventListener('play', () => { button.textContent = 'Pause'; });
  audio.addEventListener('pause', () => { button.textContent = 'Play'; });
  audio.addEventListener('ended', () => { button.textContent = 'Play'; paint(1); });
  audio.addEventListener('error', () => {
    button.textContent = 'No audio';
    button.setAttribute('disabled', '');
  });

  button.addEventListener('click', () => {
    if (audio.paused) {
      // Autoplay rejection is normal on mobile until the gesture lands; the
      // button stays usable rather than the panel looking broken.
      const p = audio.play();
      if (p && typeof p.catch === 'function') p.catch(() => { button.textContent = 'Play'; });
    } else {
      audio.pause();
    }
  });

  wave.addEventListener('click', (event) => {
    if (!Number.isFinite(audio.duration) || !audio.duration) return;
    const box = wave.getBoundingClientRect();
    if (!box.width) return;
    const ratio = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    audio.currentTime = ratio * audio.duration;
    paint(ratio);
  });

  // The panel re-renders on every scene change and on every score; without this
  // the previous scene's audio keeps playing underneath the new one.
  onCleanup?.(() => {
    audio.pause();
    audio.removeAttribute('src');
    audio.load();
  });

  return root;
}
