export function isRecordingSupported() {
  return Boolean(
    window.navigator?.mediaDevices?.getUserMedia
    && typeof window.MediaRecorder !== 'undefined',
  );
}

function stopTracks(stream) {
  stream?.getTracks().forEach((track) => track.stop());
}

export async function startMediaRecording() {
  if (!isRecordingSupported()) {
    throw new Error('Recording is not supported in this browser.');
  }

  let stream = null;

  try {
    stream = await window.navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
      throw new Error('Microphone permission was denied.');
    }

    throw new Error('Microphone capture could not start.');
  }

  let recorder = null;

  try {
    recorder = new window.MediaRecorder(stream);
  } catch {
    stopTracks(stream);
    throw new Error('Recording is not supported for this microphone stream.');
  }

  const chunks = [];
  let stopPromise = null;

  recorder.addEventListener('dataavailable', (event) => {
    if (event.data?.size) {
      chunks.push(event.data);
    }
  });

  function stop() {
    if (stopPromise) {
      return stopPromise;
    }

    stopPromise = new Promise((resolve) => {
      recorder.addEventListener('stop', () => {
        stopTracks(stream);
        resolve(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }));
      }, { once: true });

      if (recorder.state === 'inactive') {
        stopTracks(stream);
        resolve(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }));
        return;
      }

      recorder.stop();
    });

    return stopPromise;
  }

  try {
    recorder.start();
  } catch {
    stopTracks(stream);
    throw new Error('Recording could not start.');
  }

  return {
    stop,
    cleanup() {
      if (recorder.state !== 'inactive') {
        recorder.stop();
      }

      stopTracks(stream);
    },
  };
}
