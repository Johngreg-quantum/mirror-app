(function() {
  function getSupportedMimeType(mediaRecorderCtor) {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/ogg;codecs=opus',
      'audio/ogg',
    ];
    for (const type of candidates) {
      if (typeof mediaRecorderCtor !== 'undefined' && mediaRecorderCtor.isTypeSupported(type)) return type;
    }
    return '';
  }

  function cleanupRecordingRuntime(options) {
    const mediaRecorder = options.mediaRecorder;
    const micStream = options.micStream;
    const timerInterval = options.timerInterval;
    const clearIntervalFn = options.clearIntervalFn;
    const stopWaveform = options.stopWaveform;

    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    if (micStream) micStream.getTracks().forEach(function(track) { track.stop(); });
    clearIntervalFn(timerInterval);
    stopWaveform();
  }

  function startYouTubeEndCheck(options) {
    const getPlayer = options.getPlayer;
    const getTimes = options.getTimes;
    const onEnded = options.onEnded;
    const setIntervalFn = options.setIntervalFn;
    const stopCurrent = options.stopCurrent;

    stopCurrent();
    const times = getTimes();
    const player = getPlayer();
    if (!times || !player) return null;

    return setIntervalFn(function() {
      const currentPlayer = getPlayer();
      if (!currentPlayer) return stopCurrent();
      if (currentPlayer.getCurrentTime() >= times.end - 1) {
        currentPlayer.pauseVideo();
        stopCurrent();
        onEnded();
      }
    }, 250);
  }

  function stopYouTubeEndCheck(options) {
    const intervalId = options.intervalId;
    const clearIntervalFn = options.clearIntervalFn;
    if (intervalId) clearIntervalFn(intervalId);
    return null;
  }

  function renderWaveformBars(options) {
    const wrap = options.wrap;
    const barCount = options.barCount;
    const createElement = options.createElement;
    const random = options.random;

    wrap.innerHTML = '';

    for (let i = 0; i < barCount; i++) {
      const bar = createElement('div');
      bar.className = 'waveform-bar';
      const centre = (barCount - 1) / 2;
      const dist = Math.abs(i - centre) / centre;
      const maxH = Math.round(48 - dist * 28 + random() * 10);
      const dur = (0.45 + random() * 0.65).toFixed(2);
      const delay = (-random()).toFixed(2);
      bar.style.setProperty('--wh', `${maxH}px`);
      bar.style.setProperty('--wd', `${dur}s`);
      bar.style.setProperty('--wdl', `${delay}s`);
      wrap.appendChild(bar);
    }

    wrap.classList.add('on');
  }

  function stopWaveformRuntime(options) {
    const refs = options.refs || {};
    const cancelAnimationFrameFn = options.cancelAnimationFrameFn;
    const wrap = options.wrap;

    if (refs.animFrame) cancelAnimationFrameFn(refs.animFrame);
    if (refs.analyser) { try { refs.analyser.disconnect(); } catch {} }
    if (refs.audioCtx) { try { refs.audioCtx.close(); } catch {} }
    if (wrap) wrap.classList.remove('on');
  }

  window.MIRROR_APP_RUNTIME_UTILS = {
    cleanupRecordingRuntime,
    getSupportedMimeType,
    renderWaveformBars,
    startYouTubeEndCheck,
    stopWaveformRuntime,
    stopYouTubeEndCheck,
  };
})();
