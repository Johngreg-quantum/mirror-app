export function formatElapsedTime(ms = 0) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const tenths = Math.floor((ms % 1000) / 100);

  return `${minutes}:${String(seconds).padStart(2, '0')}.${tenths}`;
}

export function createRuntimeTimer(onTick) {
  let intervalId = null;
  let startedAt = 0;
  let baseElapsedMs = 0;

  function getElapsedMs() {
    if (!startedAt) {
      return baseElapsedMs;
    }

    return baseElapsedMs + Date.now() - startedAt;
  }

  function tick() {
    onTick?.(getElapsedMs());
  }

  function start(fromMs = 0) {
    stop();
    baseElapsedMs = fromMs;
    startedAt = Date.now();
    tick();
    intervalId = window.setInterval(tick, 150);
  }

  function stop() {
    if (intervalId) {
      window.clearInterval(intervalId);
      intervalId = null;
    }

    if (startedAt) {
      baseElapsedMs = getElapsedMs();
      startedAt = 0;
    }

    return baseElapsedMs;
  }

  function reset() {
    stop();
    baseElapsedMs = 0;
    tick();
  }

  return {
    start,
    stop,
    reset,
    getElapsedMs,
    cleanup: stop,
  };
}
