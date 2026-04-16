(function() {
  function setButton(buttonEl, disabled, html) {
    if (!buttonEl) return;
    buttonEl.disabled = disabled;
    if (html !== undefined) buttonEl.innerHTML = html;
  }

  function renderRecordingResetDisplay(options) {
    const refs = options.refs || {};
    const helpers = options.helpers || {};

    setButton(refs.recordBtn, false, helpers.btnRecordHTML());
    setButton(refs.stopBtn, true);
    setButton(refs.playBtn, true, helpers.btnPlayHTML());
    setButton(refs.analyzeBtn, true);
    if (refs.recIndicatorEl) refs.recIndicatorEl.classList.toggle('on', false);
    if (refs.recTimeEl) refs.recTimeEl.textContent = '0:00';
  }

  function renderRecordingActiveDisplay(options) {
    const refs = options.refs || {};

    setButton(refs.recordBtn, true);
    setButton(refs.stopBtn, false);
    if (refs.recIndicatorEl) refs.recIndicatorEl.classList.toggle('on', true);
  }

  function renderRecordingStoppedDisplay(options) {
    const refs = options.refs || {};

    setButton(refs.stopBtn, true);
    if (refs.recIndicatorEl) refs.recIndicatorEl.classList.toggle('on', false);
  }

  function renderRecordingEmptyDisplay(options) {
    const refs = options.refs || {};
    const helpers = options.helpers || {};

    setButton(refs.recordBtn, false, helpers.btnRecordHTML());
    setButton(refs.playBtn, true, helpers.btnPlayHTML());
    setButton(refs.analyzeBtn, true);
  }

  function renderRecordingReadyDisplay(options) {
    const refs = options.refs || {};
    const helpers = options.helpers || {};

    setButton(refs.playBtn, false, helpers.btnPlayHTML());
    setButton(refs.analyzeBtn, false);
    setButton(refs.recordBtn, false, helpers.btnRecordHTML());
  }

  function renderPlaybackActiveDisplay(options) {
    const refs = options.refs || {};
    const helpers = options.helpers || {};

    setButton(refs.playBtn, false, helpers.btnStopPlayHTML());
  }

  function renderPlaybackStoppedDisplay(options) {
    const refs = options.refs || {};
    const helpers = options.helpers || {};

    setButton(refs.playBtn, false, helpers.btnPlayHTML());
  }

  function renderRecordingTimerDisplay(options) {
    const refs = options.refs || {};
    if (refs.recTimeEl) refs.recTimeEl.textContent = options.text || '0:00';
  }

  function renderReplayLineDisplay(options) {
    const refs = options.refs || {};
    if (!refs.replayLineWrapEl) return;
    refs.replayLineWrapEl.style.display = options.isVisible ? 'flex' : 'none';
  }

  window.MIRROR_RECORDING_PLAYBACK_DOMAIN = {
    renderPlaybackActiveDisplay,
    renderPlaybackStoppedDisplay,
    renderRecordingActiveDisplay,
    renderRecordingEmptyDisplay,
    renderRecordingReadyDisplay,
    renderRecordingResetDisplay,
    renderRecordingStoppedDisplay,
    renderRecordingTimerDisplay,
    renderReplayLineDisplay,
  };
})();
