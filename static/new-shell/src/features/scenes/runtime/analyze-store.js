import { getAnalyzeAudioExtension, submitLegacyAnalyze } from '../../../lib/api/analyze.js';

function createSnapshot(overrides = {}) {
  return {
    status: 'disabled',
    disabledCode: 'no-take',
    disabledReason: 'Record a take before analyzing.',
    canSubmit: false,
    result: null,
    error: null,
    ...overrides,
  };
}

function deriveAvailability({ runtimeSnapshot, sessionStatus, sceneLocked }) {
  if (sessionStatus !== 'authenticated') {
    return {
      status: 'disabled',
      disabledCode: 'auth-required',
      disabledReason: 'Sign in before analyzing a take.',
      canSubmit: false,
    };
  }

  if (sceneLocked) {
    return {
      status: 'disabled',
      disabledCode: 'locked',
      disabledReason: 'This scene is locked for your current session.',
      canSubmit: false,
    };
  }

  if (!runtimeSnapshot?.audioBlob) {
    return {
      status: 'disabled',
      disabledCode: 'no-take',
      disabledReason: 'Record a take before analyzing.',
      canSubmit: false,
    };
  }

  if (!runtimeSnapshot.audioBlob.size) {
    return {
      status: 'disabled',
      disabledCode: 'empty-take',
      disabledReason: 'The current take is empty. Reset and record again before analyzing.',
      canSubmit: false,
    };
  }

  try {
    getAnalyzeAudioExtension(runtimeSnapshot.audioBlob);
  } catch (error) {
    return {
      status: 'disabled',
      disabledCode: 'unsupported',
      disabledReason: error?.message || 'This take format is not supported for analysis.',
      canSubmit: false,
    };
  }

  return {
    status: 'idle',
    disabledCode: '',
    disabledReason: '',
    canSubmit: true,
  };
}

export function createAnalyzeStore({
  runtime,
  sceneId,
  sessionStatus = 'unknown',
  sceneLocked = false,
  onAuthFailure = null,
} = {}) {
  let snapshot = createSnapshot();
  let runtimeSnapshot = runtime?.getSnapshot?.() || null;
  let currentBlob = runtimeSnapshot?.audioBlob || null;
  let requestVersion = 0;
  let disposed = false;
  let activeAbortController = null;
  const subscribers = new Set();

  function publish(nextSnapshot) {
    if (disposed) {
      return;
    }

    snapshot = nextSnapshot;
    subscribers.forEach((subscriber) => subscriber(snapshot));
  }

  function abortActiveRequest() {
    if (!activeAbortController) {
      return;
    }

    activeAbortController.abort();
    activeAbortController = null;
  }

  function applyAvailability(availability) {
    publish(createSnapshot({
      ...availability,
      result: null,
      error: null,
    }));
  }

  function syncRuntime(nextRuntimeSnapshot) {
    runtimeSnapshot = nextRuntimeSnapshot;
    const nextBlob = runtimeSnapshot?.audioBlob || null;
    const availability = deriveAvailability({ runtimeSnapshot, sessionStatus, sceneLocked });
    const takeChanged = nextBlob !== currentBlob;

    if (takeChanged) {
      currentBlob = nextBlob;
      requestVersion += 1;
      abortActiveRequest();
      applyAvailability(availability);
      return;
    }

    if (snapshot.status === 'disabled') {
      publish(createSnapshot({
        ...availability,
        result: null,
        error: null,
      }));
      return;
    }

    if (availability.status === 'disabled') {
      requestVersion += 1;
      abortActiveRequest();
      applyAvailability(availability);
      return;
    }

    if (snapshot.status === 'idle' || snapshot.status === 'error') {
      publish({
        ...snapshot,
        canSubmit: snapshot.status !== 'submitting' && snapshot.status !== 'success' && !snapshot.error?.authRequired,
      });
    }
  }

  async function submit() {
    const availability = deriveAvailability({ runtimeSnapshot, sessionStatus, sceneLocked });

    if (availability.status === 'disabled') {
      applyAvailability(availability);
      return snapshot;
    }

    const submissionBlob = runtimeSnapshot.audioBlob;
    const submissionVersion = requestVersion + 1;
    requestVersion = submissionVersion;
    abortActiveRequest();
    activeAbortController = new AbortController();

    publish({
      ...snapshot,
      status: 'submitting',
      disabledCode: '',
      disabledReason: '',
      canSubmit: false,
      error: null,
    });

    try {
      const result = await submitLegacyAnalyze({
        sceneId,
        audioBlob: submissionBlob,
        signal: activeAbortController.signal,
      });

      if (
        disposed
        || requestVersion !== submissionVersion
        || runtimeSnapshot?.audioBlob !== submissionBlob
      ) {
        return snapshot;
      }

      activeAbortController = null;
      publish(createSnapshot({
        status: 'success',
        disabledCode: '',
        disabledReason: '',
        canSubmit: false,
        result,
        error: null,
      }));
      return snapshot;
    } catch (error) {
      if (error?.name === 'AbortError') {
        return snapshot;
      }

      if (
        disposed
        || requestVersion !== submissionVersion
        || runtimeSnapshot?.audioBlob !== submissionBlob
      ) {
        return snapshot;
      }

      activeAbortController = null;
      publish(createSnapshot({
        status: 'error',
        disabledCode: '',
        disabledReason: '',
        canSubmit: !error?.authRequired,
        result: null,
        error,
      }));

      if (error?.authRequired) {
        onAuthFailure?.(error);
      }

      return snapshot;
    }
  }

  const unsubscribeRuntime = runtime?.subscribe?.((nextRuntimeSnapshot) => {
    syncRuntime(nextRuntimeSnapshot);
  });

  syncRuntime(runtimeSnapshot);

  return {
    getSnapshot() {
      return snapshot;
    },
    submit,
    cleanup() {
      disposed = true;
      requestVersion += 1;
      abortActiveRequest();
      unsubscribeRuntime?.();
      subscribers.clear();
    },
    subscribe(subscriber) {
      subscribers.add(subscriber);
      subscriber(snapshot);
      return () => subscribers.delete(subscriber);
    },
  };
}
