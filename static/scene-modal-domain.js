(function() {
  function renderSceneModalDisplay(options) {
    const scene = options.scene || {};
    const color = options.color || '';
    const hasVideo = !!options.hasVideo;
    const isDaily = !!options.isDaily;
    const refs = options.refs || {};

    if (refs.modalEl) refs.modalEl.style.setProperty('--mc', color);
    if (refs.yearEl) refs.yearEl.textContent = scene.year || '';
    if (refs.titleEl) {
      refs.titleEl.textContent = scene.movie || '';
      refs.titleEl.style.color = color;
    }
    if (refs.quoteEl) refs.quoteEl.textContent = scene.quote ? `\u201c${scene.quote}\u201d` : '';
    if (refs.targetQuoteEl) refs.targetQuoteEl.style.borderLeftColor = color;
    if (refs.analyzeBtn) refs.analyzeBtn.style.background = color;
    if (refs.videoFrameEl) refs.videoFrameEl.style.display = hasVideo ? '' : 'none';
    if (refs.videoPlaceholderEl) refs.videoPlaceholderEl.style.display = hasVideo ? 'none' : 'flex';
    if (refs.badgeEl) refs.badgeEl.classList.toggle('on', isDaily);
  }

  window.MIRROR_SCENE_MODAL_DOMAIN = {
    renderSceneModalDisplay,
  };
})();
