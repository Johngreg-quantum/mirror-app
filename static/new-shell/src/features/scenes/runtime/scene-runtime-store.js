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

export function createSceneRuntimeStore({
  canRecord = false,
  disabledReason = '',
  consentGate,
} = {}) {
  // Required, and deliberately not defaulted. A default would either fail open
  // (recording without the §6 notice — the bug this exists to prevent) or fail
  // closed silently (recording mysteriously dead). Throwing turns an omission
  // into an immediate, obvious construction error.
  if (!consentGate || typeof consentGate.hasConsented !== 'function') {
    throw new Error('createSceneRuntimeStore requires a consentGate.');
  }

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

  // Split so getUserMedia can be issued inside whichever click gesture
  // triggered it — the Record button when already consented, or the notice's
  // Accept button on the first run. Nothing may be awaited before it.
  function startRecording() {
    try {
      requireAvailable();
    } catch (error) {
      logFrontendError(error, { phase: 'recording-start', surface: 'scene-runtime' });
      setSnapshot({ status: 'error', error, level: 0 });
      return;
    }

    if (snapshot.status === 'recording') {
      return;
    }

    // Consent gate (§6). Synchronous: an unconsented tap shows the notice and
    // stops here, leaving the take untouched and the mic never opened, so the
    // browser's permission prompt cannot precede the explanation. Accepting
    // calls back into beginRecording() from its own gesture.
    if (!consentGate.hasConsented()) {
      consentGate.show();
      return;
    }

    beginRecording(startMediaRecording());
  }

  async function beginRecording(recorderPromise) {
    try {
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
      // Already issued by the caller inside the click gesture — do not call
      // startMediaRecording() here, or getUserMedia moves out of the gesture.
      startRequest = recorderPromise;
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
    // Called from the consent notice's Accept click. Issues getUserMedia as the
    // first statement so it stays inside that gesture.
    beginRecordingFromGesture() {
      beginRecording(startMediaRecording());
    },
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
