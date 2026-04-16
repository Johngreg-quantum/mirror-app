export function createAudioPlayback() {
  let audio = null;
  let objectUrl = null;

  function stop() {
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.src = '';
      audio = null;
    }

    if (objectUrl) {
      window.URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
  }

  async function play(blob, { onEnded, onError } = {}) {
    if (!blob) {
      throw new Error('No recorded audio is available for playback.');
    }

    stop();
    objectUrl = window.URL.createObjectURL(blob);
    audio = new Audio(objectUrl);
    audio.addEventListener('ended', () => {
      stop();
      onEnded?.();
    }, { once: true });
    audio.addEventListener('error', () => {
      stop();
      onError?.(new Error('Recorded audio could not play.'));
    }, { once: true });

    try {
      await audio.play();
    } catch (error) {
      stop();
      throw new Error(error?.message || 'Recorded audio could not play.');
    }
  }

  return {
    play,
    stop,
    cleanup: stop,
  };
}
