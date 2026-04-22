import { createRecordingControls } from './RecordingControls.js';
import { createRecordingStatus } from './RecordingStatus.js';
import { createWaveformShell } from './WaveformShell.js';
import { h } from '../lib/helpers/dom.js';
import { createAppHref } from '../lib/routing/navigation.js';
import { buttonLink, card, statusPill } from './primitives.js';

function getLockLabel({ scene, session }) {
  if (session?.status !== 'authenticated') {
    return 'Unlock state needs auth';
  }

  return scene.locked ? 'Locked' : 'Unlocked';
}

function renderPersonalizationPanel({ scene, session, progressError }) {
  if (progressError) {
    return card({
      eyebrow: 'Read state',
      title: 'Personalization unavailable',
      body: progressError.message || 'Progress could not load for this scene.',
      className: 'ns-state-card ns-state-card--error ns-support-card',
      children: [statusPill(progressError.rateLimited ? 'Rate limited' : 'Read-only fetch failed')],
    });
  }

  if (session?.status !== 'authenticated') {
    return card({
      eyebrow: 'Personal signal',
      title: 'Sign in for your scene state',
      body: 'Unlock state, personal best, and scene progress appear here once your session is verified.',
      className: 'ns-state-card ns-state-card--auth ns-support-card',
      children: [
        statusPill('Auth required'),
        buttonLink({ href: createAppHref('/auth'), text: 'Sign in', variant: 'secondary' }),
      ],
    });
  }

  return card({
    eyebrow: 'Personal signal',
    title: scene.locked ? 'Scene locked' : 'Scene ready',
    body: scene.locked
      ? 'This scene is locked for your current session. It remains visible here, but recording and analyze stay disabled.'
      : 'This scene is available for your current session. Personal bests update from saved scoring data.',
    className: 'ns-state-card ns-state-card--ready ns-support-card',
    children: [
      h('div', { className: 'ns-inline-list' }, [
        statusPill(scene.locked ? 'Locked' : 'Unlocked'),
        statusPill(`PB ${scene.personalBest ?? '--'}`),
      ]),
    ],
  });
}

function getAnalyzeStatusLabel(snapshot) {
  if (snapshot.status === 'idle') {
    return 'Ready';
  }

  if (snapshot.status === 'submitting') {
    return 'Submitting';
  }

  if (snapshot.status === 'success') {
    return 'Scored';
  }

  if (snapshot.status === 'error') {
    return snapshot.error?.authRequired ? 'Auth required' : 'Error';
  }

  if (snapshot.disabledCode === 'locked') {
    return 'Locked';
  }

  if (snapshot.disabledCode === 'auth-required') {
    return 'Auth required';
  }

  return 'Disabled';
}

function getAnalyzeDetail(snapshot) {
  if (snapshot.status === 'submitting') {
    return 'Submitting the current local take for analysis.';
  }

  if (snapshot.status === 'success') {
    return 'This take has a returned score in the score panel. Reset or record a new take to clear it.';
  }

  if (snapshot.status === 'error') {
    return snapshot.error?.message || 'Analyze failed for the current take.';
  }

  if (snapshot.status === 'idle') {
    return 'The current local take is ready for analyze submit.';
  }

  return snapshot.disabledReason || 'Record a take before analyzing.';
}

function renderLocalRuntimePanel({ canRecord, disabledReason, runtime, onCleanup }) {
  const controls = createRecordingControls({ runtime, canRecord });
  const status = createRecordingStatus({ disabledReason });
  const waveform = createWaveformShell();
  const runtimeRoot = card({
    eyebrow: 'Take one',
    title: 'Recording studio',
    body: canRecord
      ? 'Capture the take, listen back, then send only the keeper for scoring.'
      : disabledReason,
    className: `ns-runtime-card ns-support-card${canRecord ? ' is-ready' : ' is-disabled'}`,
    children: [
      status.root,
      waveform.root,
      controls.root,
      h('p', {
        className: 'ns-muted',
        text: 'Audio stays in this browser until analyze submit. Reset clears the take and its current result.',
      }),
    ],
  });
  const unsubscribe = runtime.subscribe((state) => {
    runtimeRoot.classList.toggle('is-recording', state.status === 'recording');
    runtimeRoot.classList.toggle('is-recorded', state.status === 'recorded');
    runtimeRoot.classList.toggle('is-playing', state.status === 'playing');
    runtimeRoot.classList.toggle('is-runtime-error', state.status === 'error');
    controls.update(state);
    status.update(state);
    waveform.update(state);
  });

  onCleanup?.(() => {
    unsubscribe();
  });

  return runtimeRoot;
}

function renderAnalyzePanel({ analyzeStore, onCleanup }) {
  const statePill = statusPill('Disabled');
  const endpointPill = statusPill('Scoring ready');
  const button = h('button', {
    className: 'ns-button',
    type: 'button',
    text: 'Analyze take',
    on: {
      click: () => analyzeStore.submit(),
    },
  });
  const detailEl = h('p', { className: 'ns-muted' });
  const authLink = buttonLink({ href: createAppHref('/auth'), text: 'Sign in', variant: 'secondary' });
  authLink.hidden = true;
  const analyzeRoot = card({
    eyebrow: 'Take two',
    title: 'Analyze take',
    body: 'Submit the selected take and turn the recording into score, points, and feedback.',
    className: 'ns-analyze-card ns-support-card',
    children: [
      h('div', { className: 'ns-inline-list' }, [statePill, endpointPill]),
      detailEl,
      h('div', { className: 'ns-action-row' }, [button, authLink]),
    ],
  });

  const unsubscribe = analyzeStore.subscribe((snapshot) => {
    const needsAuth = snapshot.disabledCode === 'auth-required' || Boolean(snapshot.error?.authRequired);

    analyzeRoot.classList.toggle('is-idle', snapshot.status === 'idle');
    analyzeRoot.classList.toggle('is-success', snapshot.status === 'success');
    analyzeRoot.classList.toggle('is-submitting', snapshot.status === 'submitting');
    analyzeRoot.classList.toggle('is-error', snapshot.status === 'error' && !needsAuth);
    analyzeRoot.classList.toggle('is-auth-required', needsAuth);
    statePill.classList.toggle('ns-pill--success', snapshot.status === 'success');
    statePill.classList.toggle('ns-pill--accent', needsAuth || snapshot.status === 'error');
    button.classList.toggle('ns-button--success', snapshot.status === 'success');
    button.classList.toggle('ns-button--danger', needsAuth || snapshot.status === 'error');
    button.classList.toggle('ns-button--loading', snapshot.status === 'submitting');
    statePill.textContent = getAnalyzeStatusLabel(snapshot);
    detailEl.textContent = getAnalyzeDetail(snapshot);
    button.textContent = snapshot.status === 'submitting'
      ? 'Analyzing...'
      : snapshot.status === 'success'
        ? 'Scored'
        : 'Analyze take';
    button.disabled = !snapshot.canSubmit;
    authLink.hidden = !needsAuth;
  });

  onCleanup?.(() => {
    unsubscribe();
  });

  return analyzeRoot;
}

function getRuntimeDisabledReason({ scene, session, progressError }) {
  if (progressError) {
    return 'Progress could not be verified, so recording stays disabled for this scene.';
  }

  if (session?.status !== 'authenticated') {
    return 'Sign in before recording a local take.';
  }

  if (scene.locked) {
    return 'This scene is locked for your current session.';
  }

  return '';
}

export function createSceneDetailPanel({
  scene,
  session,
  progressError,
  runtime,
  analyzeStore,
  runtimeDisabledReason,
  onCleanup,
}) {
  let currentScene = scene;
  let currentProgressError = progressError;
  let currentRuntimeDisabledReason = runtimeDisabledReason;
  const canRecord = !currentRuntimeDisabledReason;
  const mediaSlot = h('div');

  function renderMedia(s) {
    const youtubeId = s.source?.ui?.youtube_id;
    const clipStart = Number(s.source?.ui?.clip_start || 0);
    const clipEnd = Number(s.source?.ui?.clip_end || 0);

    if (youtubeId) {
      const src = `https://www.youtube.com/embed/${youtubeId}?start=${clipStart}&autoplay=0&rel=0&modestbranding=1`;
      mediaSlot.replaceChildren(
        h('div', { className: 'ns-scene-detail__video-wrap' }, [
          h('iframe', {
            className: 'ns-scene-detail__video',
            src,
            attrs: {
              frameborder: '0',
              allowfullscreen: '',
              allow: 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share',
              referrerpolicy: 'strict-origin-when-cross-origin',
            },
          }),
        ]),
      );
      return;
    }

    mediaSlot.replaceChildren(
      h('img', {
        className: 'ns-scene-detail__image',
        src: s.imageUrl,
        alt: `${s.film} scene reference`,
      }),
    );
  }
  const detailBody = h('div', { className: 'ns-scene-detail__body' });
  const personalizationSlot = h('div');
  const runtimeCard = renderLocalRuntimePanel({
    canRecord,
    disabledReason: currentRuntimeDisabledReason,
    runtime,
    onCleanup,
  });
  const analyzeCard = renderAnalyzePanel({ analyzeStore, onCleanup });

  function renderDetailBody() {
    const lockLabel = getLockLabel({ scene: currentScene, session });
    const recordLabel = session?.status !== 'authenticated'
      ? 'Sign in before recording later'
      : currentScene.locked ? 'Recording locked for now' : 'Use local runtime below';

    detailBody.replaceChildren(
      h('p', { className: 'ns-eyebrow', text: currentScene.levelName }),
      h('h2', { text: currentScene.title }),
      h('p', { className: 'ns-scene-detail__meta', text: `${currentScene.film} (${currentScene.year})` }),
      h('blockquote', { text: currentScene.quote }),
      h('div', { className: 'ns-inline-list ns-scene-detail__primary-pills' }, [
        statusPill(currentScene.difficulty),
        statusPill(currentScene.runtime),
        statusPill(`Target ${currentScene.targetScore}`),
        statusPill(lockLabel),
        currentScene.isDaily ? statusPill('Daily scene') : statusPill('Standard scene'),
      ]),
      h('div', { className: 'ns-inline-list ns-scene-detail__flow-pills' }, [
        statusPill(recordLabel),
        statusPill('Playback check'),
        statusPill(`Analyze ${getAnalyzeStatusLabel(analyzeStore.getSnapshot()).toLowerCase()}`),
      ]),
      h('p', {
        className: 'ns-muted',
        text: 'Record locally, play back your take, submit for analysis, and review refreshed progress after scoring.',
      }),
    );
  }

  function renderPersonalization() {
    personalizationSlot.replaceChildren(
      renderPersonalizationPanel({
        scene: currentScene,
        session,
        progressError: currentProgressError,
      }),
    );
  }

  function update(nextState = {}) {
    currentScene = nextState.scene || currentScene;
    currentProgressError = nextState.progressError === undefined ? currentProgressError : nextState.progressError;
    currentRuntimeDisabledReason = nextState.runtimeDisabledReason === undefined
      ? currentRuntimeDisabledReason
      : nextState.runtimeDisabledReason;

    renderMedia(currentScene);
    renderDetailBody();
    renderPersonalization();
  }

  const unsubscribeAnalyze = analyzeStore.subscribe(() => {
    renderDetailBody();
  });

  onCleanup?.(() => {
    unsubscribeAnalyze();
  });

  const root = h('div', { className: 'ns-scene-entry-stack' }, [
    h('section', { className: 'ns-scene-detail' }, [
      h('div', { className: 'ns-scene-detail__media' }, [mediaSlot]),
      detailBody,
    ]),
    h('div', { className: 'ns-grid ns-grid--three ns-scene-workflow' }, [
      personalizationSlot,
      runtimeCard,
      analyzeCard,
    ]),
  ]);

  update({
    scene: currentScene,
    progressError: currentProgressError,
    runtimeDisabledReason: currentRuntimeDisabledReason,
  });

  return {
    root,
    update,
  };
}

export { getRuntimeDisabledReason };
