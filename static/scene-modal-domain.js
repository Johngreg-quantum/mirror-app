(function() {
  // A scene's card colour is chosen for a poster tile, and many are dark --
  // No Signal's #1a2a4a is 1.4:1 on the modal's near-black, and 35 of the 47
  // fall under 4.5:1. Used as-is for the title (and as the Analyze button's
  // background under black text) they were hard or impossible to read. Same
  // hue, mixed toward white until it reads.
  function luminance(rgb) {
    const f = function(c) { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2]);
  }

  function readableOnDark(color) {
    const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(color || '').trim());
    if (!m) return color;
    let h = m[1];
    if (h.length === 3) h = h.split('').map(function(c) { return c + c; }).join('');
    const base = [0, 2, 4].map(function(i) { return parseInt(h.slice(i, i + 2), 16); });
    const BG = 0.0033;   // #0a0a0e
    for (let t = 0; t <= 1.0001; t += 0.06) {
      const rgb = base.map(function(c) { return Math.round(c + (255 - c) * t); });
      if ((luminance(rgb) + 0.05) / (BG + 0.05) >= 4.5) {
        return '#' + rgb.map(function(c) { return c.toString(16).padStart(2, '0'); }).join('');
      }
    }
    return '#f0ede6';
  }

  function renderSceneModalDisplay(options) {
    const scene = options.scene || {};
    // The scene colour still tints the card (--mc: the player's backdrop);
    // only the things that must be READ on the dark card get the lifted one.
    const tint = options.color || '';
    const color = readableOnDark(tint);
    const hasAudio = !!options.hasAudio;
    const isDaily = !!options.isDaily;
    const refs = options.refs || {};

    if (refs.modalEl) refs.modalEl.style.setProperty('--mc', tint);
    if (refs.yearEl) refs.yearEl.textContent = scene.genre || '';
    if (refs.titleEl) {
      refs.titleEl.textContent = scene.title || '';
      refs.titleEl.style.color = color;
    }
    if (refs.quoteEl) refs.quoteEl.textContent = scene.line ? `\u201c${scene.line}\u201d` : '';
    if (refs.targetQuoteEl) refs.targetQuoteEl.style.borderLeftColor = color;
    if (refs.analyzeBtn) refs.analyzeBtn.style.background = color;
    if (refs.videoFrameEl) refs.videoFrameEl.style.display = hasAudio ? '' : 'none';
    if (refs.videoPlaceholderEl) refs.videoPlaceholderEl.style.display = hasAudio ? 'none' : 'flex';
    if (refs.badgeEl) refs.badgeEl.classList.toggle('on', isDaily);
  }

  window.MIRROR_SCENE_MODAL_DOMAIN = {
    readableOnDark,
    renderSceneModalDisplay,
  };
})();
