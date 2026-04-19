import { createAudioPlayback } from './audio-playback.js';
import { startMediaRecording } from './media-recorder.js';
import { createRuntimeTimer } from './runtime-timer.js';
import { logFrontendError, trackEvent } from '../../../lib/observability.js';

function createSnapshot(overrides = {}) {
  return {
    status: 'idle',
    elapsedMs: 0,
    durationMs: 0,
    audioBlob: null,
    error: null,
    level: 0,
    ...overrides,
  };
}

function waveformLevel(status) {
  if (status !== 'recording' && status !== 'playing') {
    return 0;
  }

  return 0.3 + Math.random() * 0.7;
}

export function createSceneRuntimeStore({ canRecord = false, disabledReason = '' } = {}) {
  let snapshot = createSnapshot();
  let recorder = null;
  let startRequest = null;
  let disposed = false;
  const subscribers = new Set();
  const playback = createAudioPlayback();
  const timer = createRuntimeTimer((elapsedMs) => {
    setSnapshot({
      elapsedMs,
      level: waveformLevel(snapshot.status),
    });
  });

  function setSnapshot(overrides = {}) {
    if (disposed) {
      return;
    }

    snapshot = {
      ...snapshot,
      ...overrides,
    };
    subscribers.forEach((subscriber) => subscriber(snapshot));
  }

  function requireAvailable() {
    if (!canRecord) {
      throw new Error(disabledReason || 'Recording is not available for this scene.');
    }
  }

  async function startRecording() {
    try {
      requireAvailable();

      if (snapshot.status === 'recording') {
        return;
      }

      playback.stop();
      snapshot.audioBlob = null;
      snapshot.durationMs = 0;
      setSnapshot({
        status: 'recording',
        elapsedMs: 0,
        error: null,
        audioBlob: null,
        durationMs: 0,
      });
      timer.start(0);
      startRequest = startMediaRecording();
      const activeRecorder = await startRequest;
      startRequest = null;

      if (disposed || snapshot.status !== 'recording') {
        activeRecorder.cleanup();
        return;
      }

      recorder = activeRecorder;
      trackEvent('recording_started', {
        status: snapshot.status,
      });
    } catch (error) {
      timer.stop();
      startRequest = null;
      logFrontendError(error, {
        phase: 'recording-start',
        surface: 'scene-runtime',
      });
      setSnapshot({
        status: 'error',
        error,
        level: 0,
      });
    }
  }

  async function stopRecording() {
    if (snapshot.status !== 'recording') {
      return;
    }

    const durationMs = timer.stop();
    setSnapshot({
      status: 'idle',
      elapsedMs: durationMs,
      durationMs,
      level: 0,
    });

    try {
      const activeRecorder = recorder || await startRequest;
      const audioBlob = await activeRecorder?.stop();
      recorder = null;
      startRequest = null;

      if (!audioBlob?.size) {
        throw new Error('No audio was captured.');
      }

      setSnapshot({
        status: 'recorded',
        audioBlob,
        durationMs,
        elapsedMs: durationMs,
        error: null,
      });
      trackEvent('recording_stopped', {
        durationMs,
        audioBytes: audioBlob.size,
        audioType: audioBlob.type || '',
      });
    } catch (error) {
      logFrontendError(error, {
        phase: 'recording-stop',
        surface: 'scene-runtime',
      });
      setSnapshot({
        status: 'error',
        error,
        audioBlob: null,
      });
    }
  }

  async function playRecording() {
    try {
      requireAvailable();

      if (!snapshot.audioBlob) {
        throw new Error('No recorded audio is available for playback.');
      }

      if (snapshot.status === 'playing') {
        return;
      }

      setSnapshot({
        status: 'playing',
        elapsedMs: 0,
        error: null,
      });
      timer.start(0);
      await playback.play(snapshot.audioBlob, {
        onEnded: () => {
          timer.stop();
          setSnapshot({
            status: 'recorded',
            elapsedMs: snapshot.durationMs,
            level: 0,
          });
        },
        onError: (error) => {
          timer.stop();
          setSnapshot({
            status: 'error',
            error,
            level: 0,
          });
        },
      });
    } catch (error) {
      timer.stop();
      setSnapshot({
        status: 'error',
        error,
        level: 0,
      });
    }
  }

  function stopPlayback() {
    if (snapshot.status !== 'playing') {
      return;
    }

    playback.stop();
    timer.stop();
    setSnapshot({
      status: 'recorded',
      elapsedMs: snapshot.durationMs,
      level: 0,
    });
  }

  function resetTake() {
    recorder?.cleanup();
    recorder = null;
    startRequest = null;
    playback.stop();
    timer.reset();
    setSnapshot(createSnapshot());
  }

  function cleanup() {
    disposed = true;
    recorder?.cleanup();
    recorder = null;
    startRequest?.then((activeRecorder) => activeRecorder.cleanup()).catch(() => {});
    startRequest = null;
    playback.cleanup();
    timer.cleanup();
    subscribers.clear();
  }

  return {
    getSnapshot() {
      return snapshot;
    },
    startRecording,
    stopRecording,
    playRecording,
    stopPlayback,
    resetTake,
    cleanup,
    subscribe(subscriber) {
      subscribers.add(subscriber);
      subscriber(snapshot);
      return () => subscribers.delete(subscriber);
    },
  };
}
