// ══════════════════════════════════════════════
// CONFIG
// ══════════════════════════════════════════════
// Scene content and UI metadata are loaded from /api/scene-config so the
// frontend reads the same shared scene records as the backend.
// Stable frontend constants are loaded from /static/app-config.js.
// Pure helper utilities are loaded from /static/app-helpers.js.
// Pure view builders are loaded from /static/app-view-helpers.js.
// Render-only feature builders are loaded from /static/app-render-helpers.js.
// Read-only state derivation helpers are loaded from /static/app-state-helpers.js.
// Tiny generic DOM guard helpers are loaded from /static/app-dom-helpers.js.
// Cinematic level panel display helpers are loaded from /static/level-panel-domain.js.
// Progress dashboard display helpers are loaded from /static/progress-dashboard-domain.js.
// Scene browser display helpers are loaded from /static/scene-browser-domain.js.
// Scene modal display helpers are loaded from /static/scene-modal-domain.js.
// Analyze / score display helpers are loaded from /static/analyze-score-domain.js.
// Recording / playback display helpers are loaded from /static/recording-playback-domain.js.
// Auth modal display helpers are loaded from /static/auth-modal-domain.js.
// Daily challenge display helpers are loaded from /static/daily-challenge-domain.js.
// Runtime timer / cleanup / YouTube utility helpers are loaded from /static/app-runtime-utils.js.
// Progress refresh orchestration helpers are loaded from /static/app-progress-orchestration.js.
// Post-score refresh orchestration helpers are loaded from /static/app-post-score-orchestration.js.
// Leaderboard orchestration helpers are loaded from /static/app-leaderboard-orchestration.js.
// Level-panel orchestration helpers are loaded from /static/app-level-panel-orchestration.js.

const FRONTEND_CONFIG = window.MIRROR_FRONTEND_CONFIG || {};
const APP_HELPERS = window.MIRROR_APP_HELPERS || {};
const APP_VIEW_HELPERS = window.MIRROR_APP_VIEW_HELPERS || {};
const APP_RENDER_HELPERS = window.MIRROR_APP_RENDER_HELPERS || {};
const APP_STATE_HELPERS = window.MIRROR_APP_STATE_HELPERS || {};
const APP_DOM_HELPERS = window.MIRROR_APP_DOM_HELPERS || {};
const LEVEL_PANEL_DOMAIN = window.MIRROR_LEVEL_PANEL_DOMAIN || {};
const PROGRESS_DASHBOARD_DOMAIN = window.MIRROR_PROGRESS_DASHBOARD_DOMAIN || {};
const SCENE_BROWSER_DOMAIN = window.MIRROR_SCENE_BROWSER_DOMAIN || {};
const SCENE_MODAL_DOMAIN = window.MIRROR_SCENE_MODAL_DOMAIN || {};
const ANALYZE_SCORE_DOMAIN = window.MIRROR_ANALYZE_SCORE_DOMAIN || {};
const RECORDING_PLAYBACK_DOMAIN = window.MIRROR_RECORDING_PLAYBACK_DOMAIN || {};
const AUTH_MODAL_DOMAIN = window.MIRROR_AUTH_MODAL_DOMAIN || {};
const DAILY_CHALLENGE_DOMAIN = window.MIRROR_DAILY_CHALLENGE_DOMAIN || {};
const APP_RUNTIME_UTILS = window.MIRROR_APP_RUNTIME_UTILS || {};
const APP_PROGRESS_ORCHESTRATION = window.MIRROR_APP_PROGRESS_ORCHESTRATION || {};
const APP_POST_SCORE_ORCHESTRATION = window.MIRROR_APP_POST_SCORE_ORCHESTRATION || {};
const APP_LEADERBOARD_ORCHESTRATION = window.MIRROR_APP_LEADERBOARD_ORCHESTRATION || {};
const APP_LEVEL_PANEL_ORCHESTRATION = window.MIRROR_APP_LEVEL_PANEL_ORCHESTRATION || {};
const LEVEL_NAMES = FRONTEND_CONFIG.LEVEL_NAMES;
const LEVEL_UI_META = FRONTEND_CONFIG.LEVEL_UI_META;
const WAVE_BARS = FRONTEND_CONFIG.WAVE_BARS;
const {
  buildCircleSVG,
  formatAvgPb,
  getDivision,
  streakMessage,
  timeAgo,
} = APP_HELPERS;
const {
  buildPanelHTML,
  btnPlayHTML,
  btnRecordHTML,
  btnStopPlayHTML,
} = APP_VIEW_HELPERS;
const {
  buildHistoryItemHTML,
  buildLevelPanelCountHTML,
  buildLevelPanelSceneCardHTML,
  buildPersonalBestItemHTML,
} = APP_RENDER_HELPERS;
const {
  computeImprovedIds,
  findFirstUnlockedSceneId,
  getBestScores,
  getPositiveSceneScores,
  getUnlockedSceneIds,
  hasUnlockedScene,
} = APP_STATE_HELPERS;
const {
  setDisplayIfPresent,
  setHtmlIfPresent,
  setTextIfPresent,
} = APP_DOM_HELPERS;
const {
  renderLevelPanelDisplay,
  updateLevelCardStatsDisplay,
} = LEVEL_PANEL_DOMAIN;
const {
  renderDivCardDisplay,
  renderPersonalBestsDisplay,
  renderProgressDashboardDisplay,
} = PROGRESS_DASHBOARD_DOMAIN;
const {
  renderLevelBarDisplay,
  renderSceneCardsDisplay,
} = SCENE_BROWSER_DOMAIN;
const {
  renderSceneModalDisplay,
} = SCENE_MODAL_DOMAIN;
const {
  renderChallengeResultDisplay,
  renderPhonemeBreakdownDisplay,
  renderPointsEarnedDisplay,
  renderScoreDisplay,
} = ANALYZE_SCORE_DOMAIN;
const {
  renderPlaybackActiveDisplay,
  renderPlaybackStoppedDisplay,
  renderRecordingActiveDisplay,
  renderRecordingEmptyDisplay,
  renderRecordingReadyDisplay,
  renderRecordingResetDisplay,
  renderRecordingStoppedDisplay,
  renderRecordingTimerDisplay,
  renderReplayLineDisplay,
} = RECORDING_PLAYBACK_DOMAIN;
const {
  renderAuthErrorDisplay,
  renderAuthSubmitDisplay,
  renderAuthTabDisplay,
} = AUTH_MODAL_DOMAIN;
const {
  renderDailyCardDisplay,
  renderDailyCompleteDisplay,
  renderStreakCardDisplay,
} = DAILY_CHALLENGE_DOMAIN;
const {
  cleanupRecordingRuntime,
  getSupportedMimeType: getSupportedMimeTypeRuntime,
  renderWaveformBars,
  startYouTubeEndCheck,
  stopWaveformRuntime,
  stopYouTubeEndCheck,
} = APP_RUNTIME_UTILS;
const {
  refreshLevelBarSurface,
  refreshLevelCardStatsSurface,
  refreshSceneCardsSurface,
} = APP_PROGRESS_ORCHESTRATION;
const {
  refreshPostScoreSurfaces,
} = APP_POST_SCORE_ORCHESTRATION;
const {
  renderLeaderboardSurface,
  switchLeaderboardTabSurface,
} = APP_LEADERBOARD_ORCHESTRATION;
const {
  openLevelPanelSurface,
} = APP_LEVEL_PANEL_ORCHESTRATION;

let LEVEL_MAP = {};
let CLV_LEVELS = [];
let DEFAULT_UNLOCKED_SCENES = [];

const APP_BASE = (window.MIRROR_APP_BASE || '').replace(/\/$/, '');
const API = APP_BASE;

// ══════════════════════════════════════════════
// SHARED ORCHESTRATION UTILITIES
// ══════════════════════════════════════════════
// These helpers fall into three extraction buckets:
// 1. direct DOM helpers (`el`, `setText`, `show`, `setOn`, ...)
// 2. pure display/state selectors (`averageScore`, `formatAvgPb`, ...)
// 3. stateful UI flow helpers (`setOverlayActive`, `handleGlobalEscape`)
function el(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  el(id).textContent = value;
}

function setHtml(id, value) {
  el(id).innerHTML = value;
}

function setDisplay(id, value) {
  el(id).style.display = value;
}

function show(id, display = '') {
  setDisplay(id, display);
}

function hide(id, display = 'none') {
  setDisplay(id, display);
}

function setClassOn(id, className, isOn) {
  el(id).classList.toggle(className, isOn);
}

function setOn(id, isOn) {
  setClassOn(id, 'on', isOn);
}

function setBodyScrollLocked(locked) {
  if (locked) {
    document.body.style.position = 'fixed';
    document.body.style.width = '100%';
  } else {
    document.body.style.position = '';
    document.body.style.width = '';
  }
}

function isOverlayOpen(id) {
  return el(id).classList.contains('open');
}

function setOverlayOpen(id, isOpen) {
  setClassOn(id, 'open', isOpen);
}

function setOverlayActive(id, isOpen) {
  setOverlayOpen(id, isOpen);
  setBodyScrollLocked(isOpen);
}

function setPanelOpen(panelId, backdropId, isOpen) {
  setClassOn(panelId, 'open', isOpen);
  setClassOn(backdropId, 'open', isOpen);
  setBodyScrollLocked(isOpen);
}

function on(id, eventName, handler) {
  el(id).addEventListener(eventName, handler);
}

function onClick(id, handler) {
  on(id, 'click', handler);
}

function onSubmit(id, handler) {
  on(id, 'submit', handler);
}

function bindBackdropDismiss(id, onClose) {
  onClick(id, e => {
    if (e.target === el(id)) onClose();
  });
}

function handleGlobalEscape() {
  // Modal close priority is intentional: auth > progress > scene modal.
  if (isOverlayOpen('authModalOverlay')) {
    closeAuthModal();
  } else if (isOverlayOpen('progressOverlay')) {
    closeProgressDashboard();
  } else {
    closeModal();
  }
}

function resolveAppUrl(path) {
  if (!path) return path;
  return new URL(path, window.location.origin).toString();
}

function applySceneConfig(config) {
  scenes = (config && config.scenes) ? config.scenes : {};
  const levels = Array.isArray(config && config.levels) ? config.levels : [];
  LEVEL_MAP = {};
  CLV_LEVELS = levels.map(lv => {
    const sceneIds = Array.isArray(lv.scenes) ? lv.scenes.slice() : [];
    const meta = LEVEL_UI_META[lv.level] || {};
    sceneIds.forEach(sid => { LEVEL_MAP[sid] = lv.level; });
    return {
      level: lv.level,
      label: meta.label || `Level ${lv.level}`,
      cls: meta.cls || '',
      unlock: lv.unlock_score,
      desc: meta.desc || '',
      scenes: sceneIds,
    };
  });
  DEFAULT_UNLOCKED_SCENES = CLV_LEVELS.length ? CLV_LEVELS[0].scenes.slice() : [];
  if (!userProgress.unlocked_scenes || !userProgress.unlocked_scenes.length) {
    userProgress.unlocked_scenes = DEFAULT_UNLOCKED_SCENES.slice();
  }
}

let sceneConfigPromise = null;

function ensureSceneConfig() {
  if (!sceneConfigPromise) {
    sceneConfigPromise = (async () => {
      const r = await fetch(`${API}/api/scene-config`);
      if (!r.ok) throw new Error('Failed to load scene config');
      const data = await r.json();
      applySceneConfig(data);
      return data;
    })().catch(err => {
      sceneConfigPromise = null;
      throw err;
    });
  }
  return sceneConfigPromise;
}

function getDefaultUnlockedScenes() {
  return DEFAULT_UNLOCKED_SCENES.slice();
}

function getSceneUiMeta(sceneId) {
  const scene = scenes[sceneId];
  const meta = scene && scene.ui;
  return (meta && typeof meta === 'object' && !Array.isArray(meta)) ? meta : {};
}

function getSceneColor(sceneId, fallback = '#c9a84c') {
  const color = getSceneUiMeta(sceneId).card_color;
  return (typeof color === 'string' && color.trim()) ? color : fallback;
}

function getSceneYouTubeId(sceneId) {
  const ytId = getSceneUiMeta(sceneId).youtube_id;
  return (typeof ytId === 'string') ? ytId.trim() : '';
}

function getSceneTimes(sceneId) {
  const meta = getSceneUiMeta(sceneId);
  const start = Number(meta.clip_start);
  const end = Number(meta.clip_end);
  if (Number.isFinite(start) && Number.isFinite(end)) return { start, end };
  return null;
}

function getScenePoster(sceneId) {
  const poster = getSceneUiMeta(sceneId).poster_image;
  return (typeof poster === 'string') ? poster : '';
}

function getSceneBackdrop(sceneId) {
  const backdrop = getSceneUiMeta(sceneId).backdrop_image;
  if (typeof backdrop === 'string' && backdrop) return backdrop;
  return getScenePoster(sceneId);
}

function getScenePlaybackMeta(sceneId) {
  const ytRaw = getSceneYouTubeId(sceneId);
  const times = getSceneTimes(sceneId);
  return {
    ytRaw,
    videoId: ytRaw ? ytRaw.split('?')[0] : '',
    times,
    startSec: times ? times.start : 0,
  };
}

// ══════════════════════════════════════════════
// STATE
// ══════════════════════════════════════════════
// Shared cross-domain state map:
// - `scenes`, `LEVEL_MAP`, `CLV_LEVELS`: cards, level browser, level panel, modal, leaderboard
// - `userProgress`: cards, level bar, level panel, score UI, progress dashboard
// - `dailyChallenge`: daily card, scene cards, modal badge, challenge-style UI copy
// - `activeScene`: modal, recording/playback, analyze/score, replay controls
// - `challengeCtx` / `activeChallenge`: challenge page entry + post-score result rendering
// - `ytPlayer` / `ytApiReady`: modal playback, replay line, hear-actor controls
// Remaining core ownership points:
// - session entry owns auth/challenge routing and the first app data fan-out
// - scene entry owns gating between app scenes, auth signup, and modal opening
// - modal/recording/analyze own activeScene, media handles, timers, and submit flow
// - challenge owns accept/auth handoff and post-score result context
// - daily countdown owns reset timing and daily/streak reloads
// Adapter wrappers below delegate display/refresh work but should not own state.
let authToken   = localStorage.getItem('mirror_token') || null;
let authUser    = null;

if (authToken) {
  const _mlLanding = document.getElementById('mirrorLanding');
  if (_mlLanding) _mlLanding.style.display = 'none';
}

let scenes      = {};
let activeScene = null;
let activeLbTab = null;

let userProgress = {
  level: 1,
  best_scores: {},
  unlocked_scenes: [],
  next_level: { level: 2, required_score: 60, best_score: 0 },
};
let userProfile = { streak: 0, total_points: 0 };

let dailyChallenge    = null;
let countdownInterval = null;

let ytPlayer      = null;
let ytApiReady    = false;
let ytEndInterval = null;

let challengeCtx    = null;  // { score_to_beat } when scoring for a challenge
let activeChallenge = null;  // full challenge object when on challenge screen

window.onYouTubeIframeAPIReady = function() { ytApiReady = true; };

let mediaRecorder  = null;
let audioChunks    = [];
let audioBlob      = null;
let audioEl        = null;
let micStream      = null;
let recordingStart = 0;
let timerInterval = null;
let recSecs       = 0;

let waveAudioCtx   = null;
let waveAnalyser   = null;
let waveAnimFrame  = null;

// ══════════════════════════════════════════════
// APP BOOTSTRAP / SESSION ENTRY
// ══════════════════════════════════════════════
// Boot is the main domain router today:
// auth landing, normal app entry, and challenge-page entry all start here.
// Coupling note: authenticated entry fans out into progress, scenes, leaderboard,
// daily challenge, and streak loading before optional challenge handoff.
// Ownership point: do not extract casually; route order affects challenge URLs,
// token validation, onboarding timing, and authenticated app startup.
(async () => {
  sceneConfigPromise = ensureSceneConfig();
  // Check if we're on a challenge URL first
  const pathParts = window.location.pathname.split('/').filter(Boolean);
  if (pathParts[0] === 'challenge' && pathParts[1]) {
    await loadChallengePage(pathParts[1]);
    return;
  }
  if (authToken) {
    const ok = await verifyToken();
    if (ok) {
      await enterAuthenticatedApp();
      return;
    }
  }
  showAuthScreen();
})();

// ══════════════════════════════════════════════
// AUTH ORCHESTRATION — session state
// ══════════════════════════════════════════════
// Owns token verification, app/auth screen switching, and the authenticated
// startup sequence. Challenge entry can resume here after login/register.
async function verifyToken() {
  try {
    const r = await fetch(`${API}/api/auth/me`, {
      headers: { Authorization: `Bearer ${authToken}` },
    });
    if (!r.ok) throw new Error();
    authUser = await r.json();
    return true;
  } catch {
    clearAuth();
    return false;
  }
}

function showAuthScreen() {
  show('authScreen');
  hide('appScreen');
  setOn('challengeScreen', false);
  setOverlayOpen('authModalOverlay', false);
  setBodyScrollLocked(false);
  const mlLanding = document.getElementById('mirrorLanding');
  if (mlLanding) mlLanding.style.display = 'flex';
}

function showApp() {
  hide('authScreen');
  show('appScreen');
  setOn('challengeScreen', false);
  setText('userChipName', authUser.username);
  updateDivDot(0);  // default Bronze until profile loads
  enterAppMode();
}

function updateDivDot(points) {
  const d   = getDivision(points);
  const dot = el('divDot');
  dot.style.background = d.color;
  dot.title = d.name;
}

function clearAuth() {
  authToken = null;
  authUser  = null;
  localStorage.removeItem('mirror_token');
}

function logout() {
  authToken = null;
  userProfile = null;
  if (typeof userProgress !== 'undefined') userProgress = { level: 1, best_scores: {}, unlocked_scenes: [], next_level: { level: 2, required_score: 60, best_score: 0 } };
  localStorage.removeItem('mirror_token');
  sessionStorage.clear();
  document.body.classList.remove('app-mode');
  document.body.style.overflow = '';
  document.body.style.position = '';
  document.body.style.height = '';
  document.body.style.top = '';
  document.documentElement.style.overflow = '';
  const bottomNav = document.getElementById('bottomNav');
  if (bottomNav) bottomNav.classList.remove('visible');
  const appShell = document.getElementById('appShell');
  if (appShell) appShell.style.display = 'none';
  const siteNav = document.getElementById('siteNav');
  if (siteNav) siteNav.style.display = '';
  const scenesGrid = document.getElementById('scenesGrid');
  if (scenesGrid) scenesGrid.style.display = 'none';
  const lbSection = document.querySelector('.lb-section');
  if (lbSection) lbSection.style.display = 'none';
  const heroFeatured = document.getElementById('heroFeatured');
  if (heroFeatured) heroFeatured.style.display = 'none';
  const appScreen = document.getElementById('appScreen');
  if (appScreen) appScreen.style.display = 'none';
  const mlLanding = document.getElementById('mirrorLanding');
  if (mlLanding) mlLanding.style.display = 'flex';
  window.scrollTo(0, 0);
  window.location.href = '/';
}

async function enterAuthenticatedApp(options = {}) {
  // Ownership point: session entry fan-out. This coordinates auth UI, optional
  // onboarding, progress, scenes, scores, daily, streak, and challenge resume.
  const { showOnboarding = false } = options;
  showApp();
  if (showOnboarding && !activeChallenge) maybeShowOnboarding();
  await loadProgress();
  await Promise.all([loadScenes(), loadScores(), loadDaily(), loadStreakCard()]);
  if (activeChallenge) enterChallengeFromAuth();
}

onClick('btnLogout', logout);

// ══════════════════════════════════════════════
// AUTH ORCHESTRATION — modal open / close
// ══════════════════════════════════════════════
function openAuthModal(tab) {
  switchAuthTab(tab || 'login');
  setOverlayActive('authModalOverlay', true);
  document.body.classList.add('auth-open');
}

function closeAuthModal() {
  setOverlayActive('authModalOverlay', false);
  document.body.classList.remove('auth-open');
}

onClick('authModalClose', closeAuthModal);
bindBackdropDismiss('authModalOverlay', closeAuthModal);

onClick('navLoginBtn',     () => openAuthModal('login'));
onClick('navRegisterBtn',  () => openAuthModal('register'));
onClick('heroStartBtn',    () => openAuthModal('register'));
onClick('pricingFreeBtn',  () => openAuthModal('register'));
onClick('pricingProBtn',   () => openAuthModal('register'));

// ══════════════════════════════════════════════
// AUTH ORCHESTRATION — tab switching
// ══════════════════════════════════════════════
onClick('tabLoginBtn', () => switchAuthTab('login'));
onClick('tabRegBtn',   () => switchAuthTab('register'));

function getAuthModalDisplayRefs() {
  return {
    loginErrorEl: el('loginError'),
    loginForm: el('loginForm'),
    loginSubmitBtn: el('loginSubmit'),
    loginTabBtn: el('tabLoginBtn'),
    registerErrorEl: el('registerError'),
    registerForm: el('registerForm'),
    registerSubmitBtn: el('registerSubmit'),
    registerTabBtn: el('tabRegBtn'),
  };
}

function switchAuthTab(tab) {
  const isLogin = tab === 'login';
  renderAuthTabDisplay({
    isLogin: isLogin,
    refs: getAuthModalDisplayRefs(),
  });
}

// ══════════════════════════════════════════════
// AUTH ORCHESTRATION — login
// ══════════════════════════════════════════════
onSubmit('loginForm', async e => {
  e.preventDefault();
  const email    = el('loginEmail').value.trim();
  const password = el('loginPassword').value;

  renderAuthErrorDisplay({
    message: '',
    refs: { errorEl: el('loginError') },
  });
  renderAuthSubmitDisplay({
    disabled: true,
    refs: { buttonEl: el('loginSubmit') },
    text: 'Signing in\u2026',
  });

  try {
    const r    = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Login failed');

    authToken = data.access_token;
    authUser  = { username: data.username };
    localStorage.setItem('mirror_token', authToken);
    await enterAuthenticatedApp();
  } catch (err) {
    renderAuthErrorDisplay({
      message: err.message,
      refs: { errorEl: el('loginError') },
    });
  } finally {
    renderAuthSubmitDisplay({
      disabled: false,
      refs: { buttonEl: el('loginSubmit') },
      text: 'Sign In',
    });
  }
});

// ══════════════════════════════════════════════
// AUTH ORCHESTRATION — register
// ══════════════════════════════════════════════
onSubmit('registerForm', async e => {
  e.preventDefault();
  const username = el('regUsername').value.trim();
  const email    = el('regEmail').value.trim();
  const password = el('regPassword').value;
  const confirm  = el('regConfirm').value;

  renderAuthErrorDisplay({
    message: '',
    refs: { errorEl: el('registerError') },
  });

  if (password !== confirm) {
    renderAuthErrorDisplay({
      message: 'Passwords do not match',
      refs: { errorEl: el('registerError') },
    });
    const confirmEl = el('regConfirm');
    confirmEl.classList.add('shake');
    setTimeout(() => confirmEl.classList.remove('shake'), 400);
    return;
  }

  renderAuthSubmitDisplay({
    disabled: true,
    refs: { buttonEl: el('registerSubmit') },
    text: 'Creating account\u2026',
  });

  try {
    const r    = await fetch(`${API}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Registration failed');

    authToken = data.access_token;
    authUser  = { username: data.username };
    localStorage.setItem('mirror_token', authToken);
    await enterAuthenticatedApp({ showOnboarding: true });
  } catch (err) {
    renderAuthErrorDisplay({
      message: err.message,
      refs: { errorEl: el('registerError') },
    });
  } finally {
    renderAuthSubmitDisplay({
      disabled: false,
      refs: { buttonEl: el('registerSubmit') },
      text: 'Create Account',
    });
  }
});

// ══════════════════════════════════════════════
// PROGRESS / LEVEL PANEL ORCHESTRATION — scene browser shell
// ══════════════════════════════════════════════
// `renderCards` depends on both scene config and shared `userProgress` unlock state.
// Coupling note: cards also read `dailyChallenge` for the daily badge and open
// the scene modal when an unlocked card is selected.
async function loadScenes() {
  try {
    await ensureSceneConfig();
  } catch {
    return;
  }
  renderCards();
  updateLevelCardStats();
  const dailyEl = document.getElementById('homeDailyTitle');
  if (dailyEl && userProfile.daily_scene_id && scenes[userProfile.daily_scene_id]) {
    const s = scenes[userProfile.daily_scene_id];
    dailyEl.textContent = s.title || s.movie || userProfile.daily_scene_id;
  }
}

function renderCards() {
  refreshSceneCardsSurface({
    createCardElement: makeCard,
    grids: {
      Beginner: document.getElementById('gridBeginner'),
      Intermediate: document.getElementById('gridIntermediate'),
      Advanced: document.getElementById('gridAdvanced'),
    },
    renderSceneCardsDisplay: renderSceneCardsDisplay,
    scenes: scenes,
    setTextIfPresent: setTextIfPresent,
    userProgress: userProgress,
  });
}

function makeCard(id, s) {
  const locked  = !userProgress.unlocked_scenes.includes(id);
  const isDaily = dailyChallenge && dailyChallenge.scene_id === id;
  const color   = locked ? 'var(--muted)' : getSceneColor(id);
  const pb      = !locked && userProgress.best_scores[id];
  const el      = document.createElement('div');
  el.className  = 'scene-card' + (locked ? ' locked' : '') + (isDaily ? ' daily' : '');
  el.style.setProperty('--c', color);
  el.innerHTML = `
    ${isDaily ? '<div class="daily-card-badge">&#9733; Daily Challenge &nbsp;&bull;&nbsp; 2&times; pts</div>' : ''}
    ${locked ? `
    <div class="lock-overlay">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="11" width="18" height="11" rx="2"/>
        <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
      </svg>
      <span>Level ${LEVEL_MAP[id]} Required</span>
    </div>` : ''}
    <div class="card-top" ${isDaily ? 'style="margin-top:18px"' : ''}>
      <span class="movie-year">${s.year}</span>
      <div style="display:flex;gap:6px;align-items:center">
        ${s.mature ? '<span class="badge mature">18+</span>' : ''}
        <span class="badge ${s.difficulty.toLowerCase()}">${s.difficulty}</span>
      </div>
    </div>
    ${pb ? `<div class="pb-badge-row"><span class="pb-badge">&#11088; PB: ${Math.round(pb)}%</span></div>` : ''}
    <div class="card-movie">${s.movie}</div>
    <div class="card-quote">&ldquo;${s.quote}&rdquo;</div>
    <div class="card-foot">
      <span class="card-actor">${s.actor}</span>
      <span class="card-cta">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
        ${locked ? 'Locked' : 'Open Scene'}
      </span>
    </div>`;
  if (!locked) el.addEventListener('click', () => openModal(id, s));
  return el;
}

// ══════════════════════════════════════════════
// PROGRESS / LEVEL PANEL ORCHESTRATION — progress state
// ══════════════════════════════════════════════
// Progress feeds cards, the level bar, score refresh, and the cinematic level panel.
async function loadProgress() {
  try {
    const r = await fetch(`${API}/api/progress`, {
      headers: { Authorization: `Bearer ${authToken}` },
    });
    if (r.ok) {
      const data = await r.json();
      userProgress = data;
      if (data.quiz_passed) userProgress._quizPassed = true;
      if (typeof checkLevel2Unlock === 'function') checkLevel2Unlock();
    }
  } catch { /* keep defaults so offline dev still works */ }
  try {
    const rp = await fetch(`${API}/api/profile`, {
      headers: { Authorization: `Bearer ${authToken}` },
    });
    if (rp.ok) userProfile = await rp.json();
  } catch { /* keep defaults */ }
  renderLevelBar();
}

// ══════════════════════════════════════════════
// APP BOOTSTRAP / SESSION ENTRY — daily challenge handoff
// ══════════════════════════════════════════════
// This domain couples into scene cards and scene modal through `dailyChallenge.scene_id`.
// Stateful challenge loading and countdown ownership stay here; display-only
// daily/streak rendering is delegated to the daily challenge domain.
async function loadDaily() {
  try {
    const r = await fetch(`${API}/api/daily`);
    if (!r.ok) return;
    dailyChallenge = await r.json();
    renderDailyCard(dailyChallenge);
    startDailyCountdown(dailyChallenge.secs_until_reset);
    if (Object.keys(scenes).length) renderCards();
  } catch { /* silent — section stays hidden */ }
}

function renderDailyCard(daily) {
  renderDailyCardDisplay({
    daily: daily,
    refs: {
      actorEl: el('dcActor'),
      levelEl: el('dcLevel'),
      movieEl: el('dcMovie'),
      quoteEl: el('dcQuote'),
      sectionEl: el('dailySection'),
    },
    scenes: scenes,
  });
}

// ══════════════════════════════════════════════
// TIMER / YOUTUBE / CLEANUP UTILITIES — daily reset timer
// ══════════════════════════════════════════════
// Countdown ownership stays in app.js because reset triggers daily and streak reloads.
// Ownership point: do not extract casually; this timer initiates app data reloads.
function startDailyCountdown(initialSecs) {
  if (countdownInterval) clearInterval(countdownInterval);
  let secs = initialSecs;
  function tick() {
    if (secs < 0) secs = 0;
    const h  = Math.floor(secs / 3600);
    const m  = Math.floor((secs % 3600) / 60);
    const s  = secs % 60;
    setTextIfPresent(
      'dcCountdown',
      `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`
    );
    if (secs === 0) { clearInterval(countdownInterval); loadDaily(); loadStreakCard(); return; }
    secs--;
  }
  tick();
  countdownInterval = setInterval(tick, 1000);
}

async function loadStreakCard() {
  try {
    const r = await fetch(`${API}/api/profile`, { headers: { Authorization: `Bearer ${authToken}` } });
    if (!r.ok) return;
    const prof = await r.json();
    renderStreakCard(prof.streak || 0, prof.daily_done_today || false);
    if (prof.daily_done_today) showDailyComplete('Completed today!');
  } catch {}
}

function renderStreakCard(streak, doneToday) {
  renderStreakCardDisplay({
    streak: streak,
    doneToday: doneToday,
    refs: {
      dotRowEl: el('streakDotRow'),
      messageEl: el('streakMsg'),
      numberEl: el('streakNumber'),
    },
    days: ['Su','Mo','Tu','We','Th','Fr','Sa'],
    getNow: function() { return new Date(); },
    createElement: function(tagName) { return document.createElement(tagName); },
    getStreakMessage: streakMessage,
  });
}

function showDailyComplete(ptsText) {
  renderDailyCompleteDisplay({
    ptsText: ptsText,
    refs: {
      overlayEl: el('dcCompleteOverlay'),
      pointsEl: el('dcCompletePts'),
    },
  });
}

// ══════════════════════════════════════════════
// PROGRESS / LEVEL PANEL ORCHESTRATION — level bar
// ══════════════════════════════════════════════
// The level bar lives in the main browser surface, but it is also driven by the
// same `userProgress` state that powers score UI, cards, and the level panel.
function renderLevelBar() {
  refreshLevelBarSurface({
    levelNames: LEVEL_NAMES,
    refs: {
      detailsEl: el('levelDetails'),
      levelNumEl: el('levelNum'),
    },
    getFillEl: function() { return el('lvlFill'); },
    renderLevelBarDisplay: renderLevelBarDisplay,
    requestAnimationFrameFn: function(callback) { requestAnimationFrame(callback); },
    userProgress: userProgress,
  });
}

function showLevelUp(newLevel) {
  const t = document.createElement('div');
  t.className = 'level-up-toast';
  t.innerHTML = `
    <div class="level-up-title">Level ${newLevel} Unlocked!</div>
    <div class="level-up-sub">New scenes are now available</div>`;
  document.body.appendChild(t);
  requestAnimationFrame(() => requestAnimationFrame(() => t.classList.add('on')));
  setTimeout(() => {
    t.classList.remove('on');
    setTimeout(() => t.remove(), 400);
  }, 3200);
}

// ══════════════════════════════════════════════
// SCENE MODAL ORCHESTRATION
// ══════════════════════════════════════════════
// Cross-domain coupling: opening a modal primes recording, YouTube playback,
// daily badge display, and score state; closing it performs recording/media cleanup.
// Ownership point: do not extract casually; this owns activeScene and starts the
// recording/playback/analyze lifecycle for every scene-entry path.
function openModal(id, s) {
  activeScene = id;
  const color = getSceneColor(id);
  const playback = getScenePlaybackMeta(id);

  renderSceneModalDisplay({
    color: color,
    hasVideo: !!playback.ytRaw,
    isDaily: !!(dailyChallenge && id === dailyChallenge.scene_id),
    refs: {
      analyzeBtn: el('btnAnalyze'),
      badgeEl: el('dailyModalBadge'),
      modalEl: el('modal'),
      quoteEl: el('mQuote'),
      targetQuoteEl: document.querySelector('.target-quote'),
      titleEl: el('mTitle'),
      videoFrameEl: el('videoFrame'),
      videoPlaceholderEl: el('videoPlaceholder'),
      yearEl: el('mYear'),
    },
    scene: s,
  });

  stopEndCheck();
  hideReplayLine();
  if (playback.ytRaw) {
    if (ytApiReady) {
      initYTPlayer(playback.videoId, playback.startSec);
    } else {
      const waitId = setInterval(() => {
        if (!ytApiReady) return;
        clearInterval(waitId);
        initYTPlayer(playback.videoId, playback.startSec);
      }, 100);
    }
  } else {
    if (ytPlayer) ytPlayer.stopVideo();
  }

  resetRec();
  setOverlayActive('overlay', true);
}

function closeModal() {
  // Ownership point: modal teardown order protects media handles, replay UI,
  // YouTube state, body scroll, and activeScene reset.
  setOverlayActive('overlay', false);
  stopRecordingCleanup();
  stopEndCheck();
  hideReplayLine();
  if (ytPlayer) ytPlayer.stopVideo();
  activeScene = null;
}

onClick('btnClose', closeModal);
bindBackdropDismiss('overlay', closeModal);
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  handleGlobalEscape();
});

// ══════════════════════════════════════════════
// RECORDING / PLAYBACK ORCHESTRATION
// ══════════════════════════════════════════════
// Reads `activeScene`, owns microphone/audio state, and hands off to analyze once
// a non-empty recording exists.
// Ownership point: do not extract casually; this section owns MediaRecorder,
// mic stream, local audio blob, playback audio, recording timer, and waveform state.
function getRecordingPlaybackDisplayRefs() {
  return {
    analyzeBtn: el('btnAnalyze'),
    playBtn: el('btnPlay'),
    recIndicatorEl: el('recInd'),
    recTimeEl: el('recTime'),
    recordBtn: el('btnRecord'),
    replayLineWrapEl: el('replayLineWrap'),
    stopBtn: el('btnStop'),
  };
}

function resetRec() {
  // Ownership point: reset spans recording, playback, score, PB, points,
  // transcription, challenge-share, and analyze UI.
  stopRecordingCleanup();
  audioBlob = null; audioChunks = [];
  if (audioEl) { audioEl.pause(); audioEl = null; }

  renderRecordingResetDisplay({
    helpers: {
      btnPlayHTML: btnPlayHTML,
      btnRecordHTML: btnRecordHTML,
    },
    refs: getRecordingPlaybackDisplayRefs(),
  });
  setOn('scorePanel', false);
  setOn('pbCompare', false);
  setDisplay('phonSection', 'none');
  setOn('pbBanner', false);
  setOn('perfectBadge', false);
  const ptsPanel = el('ptsEarned');
  setOn('ptsEarned', false);
  const ex = ptsPanel.querySelector('.pts-extra');
  if (ex) ex.innerHTML = '';
  setOn('transReveal', false);
  setOn('challengeShare', false);
  el('challengeResult').className = 'challenge-result';
  setText('analyzeLabel', 'Analyze');
  setOn('spinner', false);
  stopWaveform();
}

// ══════════════════════════════════════════════
// TIMER / YOUTUBE / CLEANUP UTILITIES — recording cleanup
// ══════════════════════════════════════════════
// Shared by modal close, recording reset, and failed/finished recording paths.
function stopRecordingCleanup() {
  cleanupRecordingRuntime({
    mediaRecorder: mediaRecorder,
    micStream: micStream,
    timerInterval: timerInterval,
    clearIntervalFn: clearInterval,
    stopWaveform: stopWaveform,
  });
  if (micStream) micStream = null;
}

onClick('btnRecord', startRec);
onClick('btnStop', stopRec);
onClick('btnPlay', togglePlayback);
onClick('btnAnalyze', analyze);

async function startRec() {
  // Ownership point: browser permission, MediaRecorder construction, blob
  // creation, timer start, and empty-recording handling all stay together.
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      alert('Microphone access denied. Please allow microphone access in your browser settings and try again.');
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      alert('No microphone found. Please connect a microphone and try again.');
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      alert('Microphone is in use by another application. Please close other apps using the mic and try again.');
    } else {
      alert(`Could not access microphone: ${err.message}`);
    }
    return;
  }

  audioChunks = [];
  audioBlob = null;
  const mimeType = getSupportedMimeType();
  mediaRecorder = new MediaRecorder(micStream, mimeType ? { mimeType } : {});
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blobType = mediaRecorder.mimeType || mimeType || 'audio/webm';
    audioBlob = new Blob(audioChunks, { type: blobType });
    if (audioBlob.size === 0) {
      alert('No audio was captured — the recording was empty. Please try again and speak clearly into your microphone.');
      audioBlob = null;
      renderRecordingEmptyDisplay({
        helpers: {
          btnPlayHTML: btnPlayHTML,
          btnRecordHTML: btnRecordHTML,
        },
        refs: getRecordingPlaybackDisplayRefs(),
      });
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
      return;
    }
    renderRecordingReadyDisplay({
      helpers: {
        btnPlayHTML: btnPlayHTML,
        btnRecordHTML: btnRecordHTML,
      },
      refs: getRecordingPlaybackDisplayRefs(),
    });
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  };

  recordingStart = Date.now();
  mediaRecorder.start(100);
  startWaveform();
  renderRecordingActiveDisplay({
    refs: getRecordingPlaybackDisplayRefs(),
  });

  recSecs = 0;
  timerInterval = setInterval(() => {
    recSecs++;
    const m = Math.floor(recSecs / 60), s = recSecs % 60;
    renderRecordingTimerDisplay({
      refs: getRecordingPlaybackDisplayRefs(),
      text: `${m}:${s.toString().padStart(2,'0')}`,
    });
    if (recSecs >= 30) stopRec();
  }, 1000);
}

function stopRec() {
  // Ownership point: stop order affects MediaRecorder onstop and timer cleanup.
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  clearInterval(timerInterval);
  stopWaveform();
  renderRecordingStoppedDisplay({
    refs: getRecordingPlaybackDisplayRefs(),
  });
}

function togglePlayback() {
  // Ownership point: playback owns the transient Audio instance and button state.
  if (!audioBlob) return;
  if (audioEl && !audioEl.paused) {
    audioEl.pause(); audioEl = null;
    renderPlaybackStoppedDisplay({
      helpers: {
        btnPlayHTML: btnPlayHTML,
      },
      refs: getRecordingPlaybackDisplayRefs(),
    });
    return;
  }
  audioEl = new Audio(URL.createObjectURL(audioBlob));
  audioEl.play();
  audioEl.onended = () => {
    audioEl = null;
    renderPlaybackStoppedDisplay({
      helpers: {
        btnPlayHTML: btnPlayHTML,
      },
      refs: getRecordingPlaybackDisplayRefs(),
    });
  };
  renderPlaybackActiveDisplay({
    helpers: {
      btnStopPlayHTML: btnStopPlayHTML,
    },
    refs: getRecordingPlaybackDisplayRefs(),
  });
}

// ══════════════════════════════════════════════
// ANALYZE / SCORE ORCHESTRATION — submit recording
// ══════════════════════════════════════════════
// Coupling note: a 401 returns control to auth; a successful score refreshes
// leaderboard/progress/cards and may unlock levels.
// Ownership point: do not extract casually; submit flow bridges auth expiry,
// activeScene, audio blob upload, score rendering, and post-score refresh.
async function analyze() {
  if (!audioBlob || !activeScene) return;

  // Guard against uploading an empty blob
  if (audioBlob.size === 0) {
    alert('Error: No audio was recorded. Please record again before analyzing.');
    return;
  }

  setAnalyzeUiBusy(true);
  setBtn('btnRecord',  true);
  setOn('scorePanel', false);

  const form = new FormData();
  form.append('scene_id', activeScene);
  const ext = audioBlob.type.includes('mp4') ? 'mp4' : audioBlob.type.includes('ogg') ? 'ogg' : 'webm';
  form.append('audio', audioBlob, `recording.${ext}`);

  const recordingEnd = Date.now();
  form.append('duration_seconds', Math.round((recordingEnd - recordingStart) / 1000));

  try {
    const res = await fetch(`${API}/api/submit`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${authToken}` },
      body: form,
    });

    if (res.status === 401) {
      clearAuth();
      closeModal();
      showAuthScreen();
      return;
    }

    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: 'Server error' }));
      throw new Error(e.detail);
    }

    const data      = await res.json();
    const prevLevel = userProgress.level;
    showScore(data);
    if (typeof MissionsController !== 'undefined') MissionsController.onSubmitResponse(data);
    await refreshPostScoreSurfaces({
      activeScene: activeScene,
      previousLevel: prevLevel,
      setActiveLeaderboardTab: function(sceneId) { activeLbTab = sceneId; },
      loadScores: loadScores,
      loadProgress: loadProgress,
      renderCards: renderCards,
      getCurrentLevel: function() { return userProgress.level; },
      showLevelUp: showLevelUp,
    });
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setAnalyzeUiBusy(false);
    setBtn('btnRecord',  false, btnRecordHTML());
  }
}

// ══════════════════════════════════════════════
// ANALYZE / SCORE ORCHESTRATION — result display
// ══════════════════════════════════════════════
// Score rendering updates daily completion display, level/progress state, PB UI,
// and challenge result UI while leaving points/streak ownership in app state.
function showScore(data) {
  // Ownership point: score display is also the handoff for PB UI, points/daily
  // display, phoneme breakdown, and pending challenge result context.
  const pct = data.sync_score;
  renderScoreDisplay({
    data: data,
    hasYt: !!getSceneYouTubeId(activeScene),
    helpers: {
      animateNum: animateNum,
    },
    refs: {
      cmpOrigEl: el('cmpOrig'),
      cmpYouEl: el('cmpYou'),
      hearActorBtn: el('btnHearActor'),
      msgEl: el('scoreMsg'),
      panelEl: el('scorePanel'),
      pbCompareEl: el('pbCompare'),
      scoreBarEl: el('scoreBar'),
      scoreValEl: el('scoreVal'),
    },
  });
  renderPhonemeBreakdown(data.expected, data.transcription);
  showPointsEarned(data);

  if (data.is_new_pb) {
    setOn('pbBanner', true);
    showPBBlast();
  }

  if (challengeCtx) {
    showChallengeResult(pct, challengeCtx.score_to_beat);
    challengeCtx = null;
  }
}

function showPBBlast() {
  const COLORS = ['#C9A84C', '#fff', '#06d6a0', '#ffd166', '#f4a261', '#67e8f9'];
  const el = document.createElement('div');
  el.className = 'pb-blast';
  let html = `<div class="pb-blast-text">&#11088; New Personal Best!</div>`;
  for (let i = 0; i < 70; i++) {
    const color = COLORS[i % COLORS.length];
    const left  = Math.random() * 100;
    const delay = Math.random() * 0.6;
    const dur   = 1.4 + Math.random() * 1.4;
    const size  = 6 + Math.floor(Math.random() * 6);
    html += `<div class="pb-confetti" style="left:${left}%;width:${size}px;height:${size}px;background:${color};animation-duration:${dur}s;animation-delay:${delay}s"></div>`;
  }
  el.innerHTML = html;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

function animateNum(el, from, to, ms) {
  const start = performance.now();
  const tick = now => {
    const t = Math.min((now - start) / ms, 1);
    el.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - t, 3)));
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function showPointsEarned(data) {
  renderPointsEarnedDisplay({
    data: data,
    refs: {
      perfectBadgeEl: el('perfectBadge'),
      ptsAmountEl: el('ptsAmount'),
      ptsPanelEl: el('ptsEarned'),
      ptsTotalValEl: el('ptsTotalVal'),
      transRevealEl: el('transReveal'),
      transTextEl: el('transText'),
    },
  });
  if ((data.points_earned > 0 || data.total_points !== undefined) && data.division) {
    updateDivDot(data.total_points || 0);
  }
  // Show completion overlay on DC card if daily just completed
  if (data.is_daily && !data.daily_already_done) {
    showDailyComplete(`+${data.points_earned} pts earned today!`);
    renderStreakCard(data.streak || 0, true);
  }
}

// ══════════════════════════════════════════════
// PROGRESS / LEVEL PANEL ORCHESTRATION — leaderboard
// ══════════════════════════════════════════════
// Transition note: leaderboard is mostly display-only, but it shares `scenes` and
// `activeLbTab` with the scene-browser surface.
async function loadScores() {
  try {
    await ensureSceneConfig();
    const r    = await fetch(`${API}/api/leaderboard`);
    if (!r.ok) return;
    const data = await r.json();
    renderLeaderboard(data);
  } catch { /* silent when API offline */ }
}

function renderLeaderboard(data) {
  renderLeaderboardSurface({
    activeTab: activeLbTab,
    buildPanelHTML: buildPanelHTML,
    createElement: function(tagName) { return document.createElement(tagName); },
    data: data,
    getSceneColor: getSceneColor,
    onTabSelected: switchTab,
    refs: {
      tabsEl: document.getElementById('lbTabs'),
      panelsEl: document.getElementById('lbPanels'),
    },
    scenes: scenes,
    setActiveTab: function(sceneId) { activeLbTab = sceneId; },
  });
}

function switchTab(sid) {
  switchLeaderboardTabSurface({
    panels: document.querySelectorAll('.lb-panel'),
    sceneId: sid,
    scenes: scenes,
    setActiveTab: function(sceneId) { activeLbTab = sceneId; },
    tabs: document.querySelectorAll('.lb-tab'),
  });
}

// ══════════════════════════════════════════════
// TIMER / YOUTUBE / CLEANUP UTILITIES — media helpers
// ══════════════════════════════════════════════
// Mixed helper bucket:
// - media capability helpers
// - button UI state helpers
function getSupportedMimeType() {
  return getSupportedMimeTypeRuntime(
    typeof MediaRecorder === 'undefined' ? undefined : MediaRecorder
  );
}

function setBtn(id, disabled, html) {
  const el = document.getElementById(id);
  el.disabled = disabled;
  if (html !== undefined) el.innerHTML = html;
}

function setAnalyzeUiBusy(isBusy) {
  setBtn('btnAnalyze', isBusy);
  setOn('spinner', isBusy);
  setText('analyzeLabel', isBusy ? 'Analyzing\u2026' : 'Analyze');
}

// ══════════════════════════════════════════════
// AUTH ORCHESTRATION — onboarding handoff
// ══════════════════════════════════════════════
// This is intentionally left in-place for now; it touches auth/app entry timing.
function maybeShowOnboarding() {
  if (localStorage.getItem('mirror_onboarded')) return;

  const screen = document.getElementById('onboardScreen');
  show('onboardScreen', 'flex');
  requestAnimationFrame(() => requestAnimationFrame(() => screen.classList.add('visible')));

  screen.querySelectorAll('.onboard-step').forEach((step, i) => {
    setTimeout(() => step.classList.add('in'), 420 + i * 160);
  });
}

onClick('btnStartActing', () => {
  localStorage.setItem('mirror_onboarded', '1');
  const screen = el('onboardScreen');
  screen.classList.add('out');
  setTimeout(() => screen.remove(), 580);
});

// ══════════════════════════════════════════════
// APP BOOTSTRAP / SESSION ENTRY — landing page chrome
// ══════════════════════════════════════════════

// Custom cursor
const cursorDot = el('cursorDot');
document.addEventListener('mousemove', e => {
  cursorDot.style.left = e.clientX + 'px';
  cursorDot.style.top  = e.clientY + 'px';
});

// Nav scroll effect
window.addEventListener('scroll', () => {
  el('siteNav').classList.toggle('scrolled', window.scrollY > 50);
}, { passive: true });

// Hamburger menu
el('hamburger').addEventListener('click', () => {
  el('navLinks').classList.toggle('open');
  el('hamburger').classList.toggle('open');
});

// Close mobile nav on outside click
document.addEventListener('click', e => {
  const nav  = el('navLinks');
  const hamb = el('hamburger');
  if (nav.classList.contains('open') && !nav.contains(e.target) && !hamb.contains(e.target)) {
    nav.classList.remove('open');
    hamb.classList.remove('open');
  }
});

// Smooth scroll for anchor links + close mobile nav
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ behavior: 'smooth' });
    el('navLinks').classList.remove('open');
    el('hamburger').classList.remove('open');
  });
});

// Fade-up scroll animations via IntersectionObserver
const fadeObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      fadeObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.fade-up').forEach(el => fadeObserver.observe(el));

// ══════════════════════════════════════════════
// TIMER / YOUTUBE / CLEANUP UTILITIES — waveform visualization
// ══════════════════════════════════════════════
// Recording-only display helper domain.
function startWaveform() {
  stopWaveform();

  const wrap = document.getElementById('waveformWrap');
  renderWaveformBars({
    wrap: wrap,
    barCount: WAVE_BARS,
    createElement: function(tagName) { return document.createElement(tagName); },
    random: Math.random,
  });
}

function stopWaveform() {
  stopWaveformRuntime({
    refs: {
      animFrame: waveAnimFrame,
      analyser: waveAnalyser,
      audioCtx: waveAudioCtx,
    },
    cancelAnimationFrameFn: cancelAnimationFrame,
    wrap: document.getElementById('waveformWrap'),
  });
  if (waveAnimFrame) waveAnimFrame = null;
  if (waveAnalyser) waveAnalyser = null;
  if (waveAudioCtx) waveAudioCtx = null;
}

// ══════════════════════════════════════════════
// ANALYZE / SCORE ORCHESTRATION — phoneme breakdown
// ══════════════════════════════════════════════
function renderPhonemeBreakdown(expected, transcribed) {
  renderPhonemeBreakdownDisplay({
    expected: expected,
    helpers: {
      esTranslate: APP_HELPERS.esTranslate,
      wordBreakdown: APP_HELPERS.wordBreakdown,
    },
    refs: {
      sectionEl: el('phonSection'),
      wordsEl: el('phonWords'),
    },
    transcribed: transcribed,
  });
}

// ══════════════════════════════════════════════
// PROGRESS / LEVEL PANEL ORCHESTRATION — progress dashboard
// ══════════════════════════════════════════════
// Shared-state coupling: reads `userProgress`, historical API data, and division UI state.
onClick('btnMyProgress', openProgressDashboard);
onClick('btnProgressClose', closeProgressDashboard);
bindBackdropDismiss('progressOverlay', closeProgressDashboard);

function openProgressDashboard() {
  setOverlayActive('progressOverlay', true);
  loadHistory();
}

function closeProgressDashboard() {
  setOverlayActive('progressOverlay', false);
}

async function loadHistory() {
  setHtml('historyList', `<div class="history-empty">Loading\u2026</div>`);
  const headers = { Authorization: `Bearer ${authToken}` };
  try {
    const [histRes, profRes] = await Promise.all([
      fetch(`${API}/api/history`, { headers }),
      fetch(`${API}/api/profile`, { headers }),
    ]);
    if (!histRes.ok) throw new Error();
    const data = await histRes.json();
    renderProgressDashboard(data);
    renderPersonalBests(data.history);
    if (profRes.ok) {
      const prof = await profRes.json();
      renderDivCard(prof);
      updateDivDot(prof.total_points || 0);
    }
  } catch {
    setHtml('historyList', `<div class="history-empty">Could not load history</div>`);
  }
}

function renderDivCard(prof) {
  renderDivCardDisplay({
    profile: prof,
    refs: {
      badge: el('divCardBadge'),
      card: el('divCard'),
      nameEl: el('divCardName'),
      nextEl: el('divCardNext'),
    },
    setHtml: setHtml,
  });
}

function renderProgressDashboard({ history, stats }) {
  renderProgressDashboardDisplay({
    helpers: {
      buildCircleSVG: buildCircleSVG,
      buildHistoryItemHTML: buildHistoryItemHTML,
      computeImprovedIds: computeImprovedIds,
    },
    history: history,
    refs: {
      historyListEl: el('historyList'),
      improvementEl: el('progImprovement'),
    },
    setHtml: setHtml,
    setText: setText,
    stats: stats,
  });
}

function renderPersonalBests(history) {
  renderPersonalBestsDisplay({
    bestScores: userProgress.best_scores,
    helpers: {
      buildPersonalBestItemHTML: buildPersonalBestItemHTML,
      getSceneColor: getSceneColor,
    },
    history: history,
    pbEl: el('pbList'),
    scenes: scenes,
  });
}

// ══════════════════════════════════════════════
// RECORDING / PLAYBACK ORCHESTRATION — compare controls
// ══════════════════════════════════════════════
// Same domain as recording/playback; kept separate in-file because it depends on score visibility.
onClick('dcOpenBtn', () => {
  if (!dailyChallenge) return;
  const s = dailyChallenge.scene || scenes[dailyChallenge.scene_id];
  if (s) openModal(dailyChallenge.scene_id, s);
});

onClick('btnHearActor', hearActor);
onClick('btnHearSelf',  hearSelf);

// Flip word cards on tap/click
on('phonWords', 'click', e => {
  const card = e.target.closest('.phon-word');
  if (card) card.classList.toggle('flipped');
});
onClick('btnTryAgain', () => {
  resetRec();
  el('modal').scrollTo({ top: 0, behavior: 'smooth' });
});

function hearActor() {
  const playback = getScenePlaybackMeta(activeScene);
  if (!playback.ytRaw) return;
  hideReplayLine();
  if (ytPlayer) {
    ytPlayer.seekTo(playback.startSec, true);
    ytPlayer.playVideo();
  } else {
    show('videoFrame');
    hide('videoPlaceholder');
    initYTPlayer(playback.videoId, playback.startSec);
  }
  el('modal').scrollTo({ top: 0, behavior: 'smooth' });
}

function hearSelf() {
  if (!audioBlob) return;
  const audio = new Audio(URL.createObjectURL(audioBlob));
  audio.play();
}

// ══════════════════════════════════════════════
// TIMER / YOUTUBE / CLEANUP UTILITIES — YouTube player
// ══════════════════════════════════════════════
// Scene modal + playback shared infrastructure.
function initYTPlayer(videoId, startSec) {
  if (ytPlayer) {
    ytPlayer.loadVideoById({ videoId, startSeconds: startSec });
    return;
  }
  ytPlayer = new YT.Player('videoFrame', {
    videoId,
    playerVars: { autoplay: 1, start: startSec, rel: 0, modestbranding: 1 },
    events: { onStateChange: onYTStateChange },
  });
}

function onYTStateChange(e) {
  if (e.data === YT.PlayerState.PLAYING) {
    startEndCheck();
  } else {
    stopEndCheck();
  }
}

function startEndCheck() {
  ytEndInterval = startYouTubeEndCheck({
    getPlayer: function() { return ytPlayer; },
    getTimes: function() { return getSceneTimes(activeScene); },
    onEnded: showReplayLine,
    setIntervalFn: setInterval,
    stopCurrent: stopEndCheck,
  });
}

function stopEndCheck() {
  ytEndInterval = stopYouTubeEndCheck({
    intervalId: ytEndInterval,
    clearIntervalFn: clearInterval,
  });
}

function showReplayLine() {
  renderReplayLineDisplay({
    isVisible: true,
    refs: getRecordingPlaybackDisplayRefs(),
  });
}

function hideReplayLine() {
  renderReplayLineDisplay({
    isVisible: false,
    refs: getRecordingPlaybackDisplayRefs(),
  });
}

onClick('btnReplayLine', () => {
  const playback = getScenePlaybackMeta(activeScene);
  hideReplayLine();
  if (ytPlayer) { ytPlayer.seekTo(playback.startSec, true); ytPlayer.playVideo(); }
});

// ══════════════════════════════════════════════
// CHALLENGE ORCHESTRATION
// ══════════════════════════════════════════════
// Left as-is for now; this is one of the higher-coupling domains because it bridges
// auth entry, modal opening, score display, and share UI.
// Ownership point: do not extract casually; challenge URL entry, auth handoff,
// active challenge state, accept flow, and result context are coupled.
async function loadChallengePage(cid) {
  try {
    const r = await fetch(`${API}/api/challenge/${cid}`);
    if (!r.ok) { showAuthScreen(); return; }
    activeChallenge = await r.json();
    setText('chlgChallenger', activeChallenge.challenger_username);
    setText('chlgScoreVal', Math.round(activeChallenge.score_to_beat));
    setText('chlgMovie', activeChallenge.scene.movie || '');
    const noteEl = el('chlgAuthNote');
    if (authToken) {
      const ok = await verifyToken();
      if (ok) {
        noteEl.textContent = `Playing as ${authUser.username}`;
      } else {
        noteEl.innerHTML = `<a id="chlgLoginLink">Log in</a> to record your score`;
        onClick('chlgLoginLink', showAuthFromChallenge);
      }
    } else {
      noteEl.innerHTML = `<a id="chlgLoginLink">Log in or register</a> to record your score`;
      onClick('chlgLoginLink', showAuthFromChallenge);
    }
    setOn('challengeScreen', true);
    hide('authScreen');
    hide('appScreen');
  } catch {
    showAuthScreen();
  }
}

function showAuthFromChallenge() {
  setOn('challengeScreen', false);
  showAuthScreen();
  openAuthModal('login');
}

function enterChallengeFromAuth() {
  // Ownership point: challenge accept resumes into scene modal and seeds
  // challengeCtx for post-score result rendering.
  if (!activeChallenge) return;
  challengeCtx = { score_to_beat: activeChallenge.score_to_beat };
  closeAuthModal();
  const sid = activeChallenge.scene_id;
  const s   = scenes[sid] || activeChallenge.scene;
  if (s) openModal(sid, s);
}

onClick('btnAcceptChallenge', () => {
  if (!activeChallenge) return;
  if (authToken && authUser) {
    enterChallengeFromAuth();
  } else {
    showAuthFromChallenge();
  }
});

onClick('btnChallenge', createChallenge);

async function createChallenge() {
  if (!authToken || !activeScene) return;
  const score = parseFloat(el('scoreVal').textContent) || 0;
  setBtn('btnChallenge', true, '&#9876; Generating\u2026');
  try {
    const r = await fetch(`${API}/api/challenge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${authToken}` },
      body: JSON.stringify({ scene_id: activeScene, score }),
    });
    if (!r.ok) throw new Error('Failed');
    const data = await r.json();
    const challengeUrl = resolveAppUrl(data.url);
    const movie = scenes[activeScene]?.movie || 'MIRROR';
    const msg   = `I scored ${score}% on ${movie} in MIRROR! Can you beat it? ${challengeUrl}`;
    setText('chlgLinkInput', challengeUrl);
    el('btnCopyLink').onclick = () => {
      navigator.clipboard.writeText(challengeUrl).then(() => {
        setText('btnCopyLink', '\u2713 Copied!');
        setTimeout(() => { setText('btnCopyLink', 'Copy'); }, 2000);
      });
    };
    el('btnWhatsapp').onclick = () => {
      window.open(`https://api.whatsapp.com/send?text=${encodeURIComponent(msg)}`, '_blank');
    };
    setOn('challengeShare', true);
  } catch {
    alert('Could not create challenge link. Please try again.');
  } finally {
    setBtn('btnChallenge', false, '&#9876; Challenge a Friend');
  }
}

function showChallengeResult(score, scoreToBeat) {
  // Adapter wrapper: display-only result rendering; challengeCtx ownership stays
  // in showScore/enterChallengeFromAuth.
  renderChallengeResultDisplay({
    refs: {
      resultEl: el('challengeResult'),
    },
    score: score,
    scoreToBeat: scoreToBeat,
  });
}

// ══════════════════════════════════════════════
// PROGRESS / LEVEL PANEL ORCHESTRATION — cinematic level cards
// ══════════════════════════════════════════════
// Cross-domain coupling: this domain mirrors browser-card unlock state and opens scene modals.
function updateLevelCardStats() {
  refreshLevelCardStatsSurface({
    formatAvgPb: formatAvgPb,
    getBestScores: getBestScores,
    getDefaultUnlockedScenes: getDefaultUnlockedScenes,
    getPositiveSceneScores: getPositiveSceneScores,
    getUnlockedSceneIds: getUnlockedSceneIds,
    hasUnlockedScene: hasUnlockedScene,
    levels: CLV_LEVELS,
    setDisplayIfPresent: setDisplayIfPresent,
    setTextIfPresent: setTextIfPresent,
    updateLevelCardStatsDisplay: updateLevelCardStatsDisplay,
    userProgress: userProgress,
  });
}

async function openLevelPanel(level) {
  // Adapter wrapper: app still owns scene-config readiness and scene-entry
  // callbacks; panel render/open sequencing is delegated.
  try {
    await ensureSceneConfig();
  } catch {
    return;
  }
  openLevelPanelSurface({
    level: level,
    levels: CLV_LEVELS,
    userProgress: userProgress,
    scenes: scenes,
    getBestScores: getBestScores,
    getDefaultUnlockedScenes: getDefaultUnlockedScenes,
    getUnlockedSceneIds: getUnlockedSceneIds,
    renderLevelPanelDisplay: renderLevelPanelDisplay,
    setText: setText,
    setTextIfPresent: setTextIfPresent,
    setPanelOpen: setPanelOpen,
    helpers: {
      buildLevelPanelCountHTML: buildLevelPanelCountHTML,
      buildLevelPanelSceneCardHTML: buildLevelPanelSceneCardHTML,
      findFirstUnlockedSceneId: findFirstUnlockedSceneId,
      formatAvgPb: formatAvgPb,
      getPositiveSceneScores: getPositiveSceneScores,
      getSceneColor: getSceneColor,
      getScenePoster: getScenePoster,
      setHtmlIfPresent: setHtmlIfPresent,
      setTextIfPresent: setTextIfPresent,
    },
    refs: {
      badgeEl: el('clvPanelBadge'),
      listEl: el('clvClipList'),
      playBtn: el('clvPanelPlayBtn'),
      subEl: el('clvPanelSub'),
      titleEl: el('clvPanelTitle'),
    },
    onPlayFirstScene: function(firstScene) { closeLevelPanel(); selectScene(firstScene); },
  });
}

function closeLevelPanel() {
  setPanelOpen('clvPanel', 'clvPanelBackdrop', false);
}

function selectScene(sid) {
  // Ownership point: scene-entry gating decides between authenticated app scene
  // modal opening and logged-out registration. Do not extract casually.
  closeLevelPanel();
  const appScreen = el('appScreen');
  if (appScreen && appScreen.style.display !== 'none') {
    const s = scenes && scenes[sid];
    if (s) openModal(sid, s);
  } else {
    openAuthModal('register');
  }
}

// Hook renderLevelBar to also refresh level card stats after progress loads.
// Cross-domain coupling to keep in mind for extraction:
// level-bar updates on the main surface also refresh cinematic level-panel stats.
const _origRenderLevelBar = renderLevelBar;
renderLevelBar = function () { _origRenderLevelBar(); updateLevelCardStats(); };

// ══════════════════════════════════════════════
// APP BOOTSTRAP / SESSION ENTRY — cinematic dashboard hero
// ══════════════════════════════════════════════
// Coupling note: this wraps daily loading so the dashboard hero follows the
// same daily challenge data without owning challenge creation or scoring.
function renderHeroFeatured() {
  if (!dailyChallenge) return;
  const sid = dailyChallenge.scene_id;
  const s = dailyChallenge.scene || scenes[sid];
  if (!s) return;

  const heroImg = getSceneBackdrop(sid);
  const posterImg = document.getElementById('heroPosterImg');
  if (posterImg && heroImg) {
    posterImg.src = heroImg;
    posterImg.alt = s.movie;
  }

  const titleEl = document.getElementById('heroTitle');
  if (titleEl) titleEl.textContent = s.movie.toUpperCase();

  const yearEl = document.getElementById('heroYear');
  if (yearEl) yearEl.textContent = s.year || '';

  const quoteEl = document.getElementById('heroQuote');
  if (quoteEl) quoteEl.textContent = s.quote || '';
}

// Hero play button → open recording modal for daily scene
(function() {
  const playBtn = document.getElementById('heroPlayBtn');
  if (playBtn) {
    playBtn.addEventListener('click', () => {
      if (!dailyChallenge) return;
      const s = dailyChallenge.scene || scenes[dailyChallenge.scene_id];
      if (s) openModal(dailyChallenge.scene_id, s);
    });
  }
})();

// Hook into loadDaily to also render the hero
const _origLoadDaily = loadDaily;
loadDaily = async function() {
  await _origLoadDaily();
  renderHeroFeatured();
};

// ══════════════════════════════════════════════
// PROGRESS / LEVEL PANEL ORCHESTRATION — poster carousel
// ══════════════════════════════════════════════
// Coupling note: this wraps scene loading so the carousel follows the same
// scene config and unlock state as the primary scene browser.
function renderCarousel() {
  const track = document.getElementById('carouselTrack');
  if (!track) return;
  track.innerHTML = '';

  const unlocked = userProgress.unlocked_scenes || [];
  const sceneIds = Object.keys(scenes).filter(sid => unlocked.includes(sid));
  if (!sceneIds.length) return;

  // Build card elements
  const cards = [];
  sceneIds.forEach((sid, i) => {
    const s = scenes[sid];
    if (!s) return;
    const poster = getScenePoster(sid);

    const card = document.createElement('div');
    card.className = 'carousel-card';
    card.dataset.sid = sid;
    card.dataset.index = i;

    const posterDiv = document.createElement('div');
    posterDiv.className = 'carousel-poster';

    if (poster) {
      const img = document.createElement('img');
      img.src = poster;
      img.alt = s.movie;
      img.loading = 'lazy';
      posterDiv.appendChild(img);

      // Reflection
      const ref = document.createElement('div');
      ref.className = 'carousel-reflection';
      const refImg = document.createElement('img');
      refImg.src = poster;
      refImg.alt = '';
      refImg.loading = 'lazy';
      ref.appendChild(refImg);
      card.appendChild(posterDiv);
      card.appendChild(ref);
    } else {
      posterDiv.style.background = `linear-gradient(135deg, ${getSceneColor(sid)}22, #111)`;
      card.appendChild(posterDiv);
    }

    card.addEventListener('click', () => {
      const idx = parseInt(card.dataset.index);
      if (idx === cfState.center) {
        openModal(sid, s);
      } else {
        const diff = idx - cfState.center;
        if (Math.abs(diff) <= 3) {
          cfRotate(diff > 0 ? 1 : -1);
        }
      }
    });

    track.appendChild(card);
    cards.push({ card, sid, scene: s });
  });

  // Coverflow state
  const cfState = { center: 0, total: cards.length };

  function cfPositions(center, total) {
    const slots = [
      { x: -440, z: -320, ry: 62,  b: 0.15, zi: 1 },
      { x: -320, z: -230, ry: 56,  b: 0.3,  zi: 3 },
      { x: -185, z: -130, ry: 46,  b: 0.55, zi: 6 },
      { x: 0,    z: 0,    ry: 0,   b: 1,    zi: 10 },
      { x: 185,  z: -130, ry: -46, b: 0.55, zi: 6 },
      { x: 320,  z: -230, ry: -56, b: 0.3,  zi: 3 },
      { x: 440,  z: -320, ry: -62, b: 0.15, zi: 1 },
    ];

    const result = [];
    for (let i = 0; i < total; i++) {
      let offset = i - center;
      if (offset > total / 2) offset -= total;
      if (offset < -total / 2) offset += total;
      const slotIdx = offset + 3;
      if (slotIdx >= 0 && slotIdx <= 6) {
        result.push({ i, slot: slots[slotIdx] });
      } else {
        const side = offset > 0 ? 1 : -1;
        result.push({ i, slot: { x: side * 600, z: -400, ry: side * 65, b: 0, zi: 0 } });
      }
    }
    return result;
  }

  function cfRender() {
    const positions = cfPositions(cfState.center, cfState.total);
    positions.forEach(({ i, slot }) => {
      const { card } = cards[i];
      card.style.transform = `translateX(${slot.x}px) translateZ(${slot.z}px) rotateY(${slot.ry}deg)`;
      card.style.filter = `brightness(${slot.b})`;
      card.style.zIndex = slot.zi;

      const poster = card.querySelector('.carousel-poster');
      if (slot.zi === 10 && poster) {
        poster.style.boxShadow = '0 30px 80px rgba(0,0,0,0.85), 0 0 0 0.5px rgba(200,169,110,0.35), 0 0 50px rgba(200,169,110,0.12)';
        poster.style.borderColor = 'rgba(200,169,110,0.4)';
      } else if (poster) {
        poster.style.boxShadow = '0 20px 60px rgba(0,0,0,0.7)';
        poster.style.borderColor = 'rgba(200,169,110,0.15)';
      }
    });

    const c = cards[cfState.center];
    const titleEl = document.getElementById('coverflowTitle');
    const actorEl = document.getElementById('coverflowActor');
    if (titleEl && c) titleEl.textContent = c.scene.movie || c.scene.title || '';
    if (actorEl && c) actorEl.textContent = (c.scene.actor || '') + (c.scene.difficulty ? ' · ' + c.scene.difficulty : '');
  }

  function cfRotate(dir) {
    cfState.center = (cfState.center + dir + cfState.total) % cfState.total;
    cfRender();
  }

  const prevBtn = document.getElementById('carouselPrev');
  const nextBtn = document.getElementById('carouselNext');
  if (prevBtn) prevBtn.onclick = () => cfRotate(-1);
  if (nextBtn) nextBtn.onclick = () => cfRotate(1);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') cfRotate(-1);
    if (e.key === 'ArrowRight') cfRotate(1);
  });

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      cfRender();
    });
  });
}

// Hook into loadScenes to also render carousel
const _origLoadScenes = loadScenes;
loadScenes = async function() {
  await _origLoadScenes();
  renderCarousel();
  if (typeof buildGatewayLevel1 === 'function') buildGatewayLevel1();
};

// Also rebuild the gateway whenever progress reloads (best_scores can change after submit)
const _origLoadProgress = loadProgress;
loadProgress = async function() {
  await _origLoadProgress();
  if (typeof buildGatewayLevel1 === 'function') buildGatewayLevel1();
};

// ══════════════════════════════════════════════
// ANALYZE / SCORE ORCHESTRATION — weak words dashboard
// ══════════════════════════════════════════════
// Coupling note: weak words reads score history after authenticated app entry
// but does not participate in submit/scoring ownership.
async function renderWeakWords() {
  const tbody = document.getElementById('weakWordsBody');
  if (!tbody) return;

  try {
    const r = await fetch(`${API}/api/history`, {
      headers: { Authorization: `Bearer ${authToken}` },
    });
    if (!r.ok) return;
    const data = await r.json();
    const history = data.history || [];
    if (!history.length) return;

    // Gather all words from all scenes the user has tried,
    // looking at latest attempt per scene and comparing expected vs transcribed
    const wordMisses = {};

    for (const entry of history) {
      const scene = scenes[entry.scene_id];
      if (!scene) continue;
      const expected = (scene.quote || '').toLowerCase().replace(/[^\w\s']/g, '').split(/\s+/).filter(Boolean);
      // We don't have word-level data from history alone, so score each scene's words
      // by comparing the scene's sync_score — lower score = weaker words
      const score = entry.sync_score || 0;
      for (const word of expected) {
        if (word.length < 3) continue; // skip tiny words
        if (!wordMisses[word]) wordMisses[word] = { total: 0, count: 0 };
        wordMisses[word].total += score;
        wordMisses[word].count += 1;
      }
    }

    // Sort by lowest average score
    const sorted = Object.entries(wordMisses)
      .map(([word, data]) => ({
        word,
        avg: Math.round(data.total / data.count),
        count: data.count,
      }))
      .sort((a, b) => a.avg - b.avg)
      .slice(0, 5);

    if (!sorted.length) return;

    tbody.innerHTML = sorted.map(item => `
      <tr>
        <td>${item.word}</td>
        <td>${item.avg}/100
          <span class="weak-word-bar">
            <span class="weak-word-bar-fill" style="width:${item.avg}%"></span>
          </span>
        </td>
      </tr>
    `).join('');
  } catch { /* silent */ }
}

// Hook renderWeakWords into the auth entry flow
const _origEnterAuth = enterAuthenticatedApp;
enterAuthenticatedApp = async function(options) {
  await _origEnterAuth(options);
  renderWeakWords();
};

// ══════════════════════════════════════════════
// MY WORDS — flashcard controller
// ══════════════════════════════════════════════
const TYPE_COLORS = { verb: '#c8a96e', noun: '#64b4ff', adj: '#b464ff', adv: '#64dc96' };

const WordsController = {
  scenes:         [],
  sceneIdx:       0,
  vocab:          [],
  queue:          [],
  mastery:        {},
  isFlipped:      false,
  exitInProgress: false,

  init() {
    if (typeof userProfile === 'undefined' || !userProfile) return;
    const ids = userProfile.translations_unlocked || [];
    this.scenes = ids.map(id => ({
      id,
      movie: (scenes && scenes[id] && (scenes[id].movie || scenes[id].title)) || id,
    }));
    this.sceneIdx = 0;
    this.mastery  = {};
    this._buildPills();
    if (!this.scenes.length) { this._renderEmpty(); return; }
    this.loadScene(this.scenes[0].id);
  },

  _buildPills() {
    const cont = document.getElementById('wordsScenePills');
    if (!cont) return;
    cont.innerHTML = this.scenes.map((s, i) =>
      `<button class="words-scene-pill${i === this.sceneIdx ? ' active' : ''}" data-idx="${i}">${s.movie}</button>`
    ).join('');
    cont.querySelectorAll('.words-scene-pill').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx, 10);
        if (idx === this.sceneIdx) return;
        this.sceneIdx = idx;
        cont.querySelectorAll('.words-scene-pill').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const targetId = this.scenes[idx].id;
        console.log('[WordsController] pill click → loadScene scene_id:', JSON.stringify(targetId), 'idx:', idx, 'movie:', this.scenes[idx].movie);
        this.loadScene(targetId);
      });
    });
  },

  async loadScene(sceneId) {
    if (this.loading) return;
    this.loading = true;
    const url = `${API}/api/vocab/${sceneId}`;
    let status = null;
    let bodyText = null;
    try {
      const r = await fetch(url, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      status = r.status;
      if (!r.ok) {
        try { bodyText = await r.text(); } catch { /* ignore */ }
        throw new Error('vocab fetch failed: ' + r.status);
      }
      this.vocab = await r.json();
      this.queue = this._shuffle(this.vocab.map((_, i) => i));
      await this.fetchMastery(sceneId);
      this.renderCard();
      this.updateProgress();
      this.loading = false;
    } catch (err) {
      console.error('[WordsController] loadScene failed', { url, status, bodyText: bodyText && bodyText.slice(0, 500), error: err && (err.message || err) });
      this.vocab = [
        {en:"choose",   es:"elegir",     phonetic:"/tʃuːz/",       example:"You have to choose your path.",      type:"verb"},
        {en:"believe",  es:"creer",      phonetic:"/bɪˈliːv/",     example:"I believe in you.",                   type:"verb"},
        {en:"escape",   es:"escapar",    phonetic:"/ɪˈskeɪp/",     example:"There is no escape.",                 type:"verb"},
        {en:"reality",  es:"realidad",   phonetic:"/riˈæləti/",    example:"What is reality?",                    type:"noun"},
        {en:"illusion", es:"ilusión",    phonetic:"/ɪˈluːʒən/",    example:"It was all an illusion.",             type:"noun"},
        {en:"free",     es:"libre",      phonetic:"/friː/",        example:"Your mind must be free.",             type:"adj"},
        {en:"truth",    es:"verdad",     phonetic:"/truːθ/",       example:"The truth will shock you.",           type:"noun"},
        {en:"discover", es:"descubrir",  phonetic:"/dɪˈskʌvər/",   example:"She had to discover it herself.",     type:"verb"},
      ];
      this.queue = [0,1,2,3,4,5,6,7];
      this.renderCard();
      this.loading = false;
    }
  },

  async fetchMastery(sceneId) {
    try {
      const r = await fetch(`${API}/api/vocab/mastery?scene_id=${encodeURIComponent(sceneId)}`, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (!r.ok) return;
      const data = await r.json();
      Object.assign(this.mastery, data);
      this.updateStars();
    } catch { /* silent */ }
  },

  renderCard() {
    if (!this.queue.length) { this.showAllMastered(); return; }
    const word = this.vocab[this.queue[0]];
    if (!word) { this.showAllMastered(); return; }

    const en = document.getElementById('wordsEnWord');
    const ph = document.getElementById('wordsPhonetic');
    const es = document.getElementById('wordsEsWord');
    const ex = document.getElementById('wordsExample');
    const tb = document.getElementById('wordsTypeBadge');
    if (en) en.textContent = word.en       || '';
    if (ph) ph.textContent = word.phonetic || '';
    if (es) es.textContent = word.es       || '';
    if (ex) ex.textContent = word.example  || '';
    if (tb) {
      const t = (word.type || '').toLowerCase();
      const c = TYPE_COLORS[t] || '#888';
      tb.textContent       = (word.type || '').toUpperCase();
      tb.className         = 'words-type-badge';
      tb.style.color       = c;
      tb.style.background  = `${c}26`;
      tb.style.outline     = `0.5px solid ${c}55`;
    }

    const inner = document.getElementById('wordsCardInner');
    const acts  = document.getElementById('wordsActions');
    const rem   = document.getElementById('wordsRemaining');
    if (inner) {
      inner.classList.remove('flipped', 'exit-left', 'exit-right');
      this.isFlipped = false;
    }
    if (acts) acts.classList.remove('visible');
    if (rem)  rem.textContent = `${this.queue.length} remaining`;
    this.updateStars();
  },

  updateStars() {
    if (!this.queue.length) return;
    const word = this.vocab[this.queue[0]];
    if (!word) return;
    const count = Math.max(0, Math.min(3, this.mastery[word.en] || 0));
    ['wordsStarsFront', 'wordsStarsBack'].forEach(id => {
      const c = document.getElementById(id);
      if (!c) return;
      c.innerHTML = '';
      for (let i = 0; i < 3; i++) {
        const sp = document.createElement('span');
        sp.className   = 'words-star' + (i < count ? ' lit' : '');
        sp.textContent = '★';
        c.appendChild(sp);
      }
    });
  },

  updateProgress() {
    const total    = this.vocab.length;
    const mastered = this.vocab.filter(w => (this.mastery[w.en] || 0) >= 3).length;
    const fill = document.getElementById('wordsProgressFill');
    const lab  = document.getElementById('wordsMasteredCount');
    if (fill) fill.style.width = total ? (mastered / total * 100) + '%' : '0%';
    if (lab)  lab.textContent  = `${mastered}/${total} mastered`;

    const list = document.getElementById('wordsMasteredList');
    if (!list) return;
    const masteredWords = this.vocab.filter(w => (this.mastery[w.en] || 0) >= 3);
    if (!masteredWords.length) { list.innerHTML = ''; return; }
    list.innerHTML =
      '<div class="words-mastered-label">MASTERED</div>' +
      '<div class="words-mastered-pills">' +
      masteredWords.map(w => `<span class="words-mastered-pill">${w.en}</span>`).join('') +
      '</div>';
  },

  flipCard() {
    if (this.exitInProgress) return;
    if (!this.queue.length)  return;
    this.isFlipped = !this.isFlipped;
    const inner = document.getElementById('wordsCardInner');
    const acts  = document.getElementById('wordsActions');
    if (inner) inner.classList.toggle('flipped', this.isFlipped);
    if (acts)  acts.classList.toggle('visible',  this.isFlipped);
  },

  animateExit(dir, callback) {
    if (this.exitInProgress) return;
    this.exitInProgress = true;
    const inner = document.getElementById('wordsCardInner');
    const acts  = document.getElementById('wordsActions');
    if (inner) inner.classList.remove('flipped');
    if (acts)  acts.classList.remove('visible');
    this.isFlipped = false;
    const cls = dir === 'right' ? 'exit-right' : 'exit-left';
    if (inner) inner.classList.add(cls);
    setTimeout(() => {
      if (inner) inner.classList.remove(cls);
      this.exitInProgress = false;
      if (typeof callback === 'function') callback();
      this.renderCard();
    }, 500);
  },

  gotIt() {
    if (!this.queue.length) return;
    const word = this.vocab[this.queue[0]];
    if (!word) return;
    const newCount = Math.min((this.mastery[word.en] || 0) + 1, 3);
    this.mastery[word.en] = newCount;
    const sceneId = this.scenes[this.sceneIdx] && this.scenes[this.sceneIdx].id;
    fetch(`${API}/api/vocab/mastery`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${authToken}` },
      body:    JSON.stringify({ scene_id: sceneId, word_en: word.en, correct: true }),
    }).catch(() => {});
    this.updateProgress();
    this.animateExit('right', () => {
      if (newCount >= 3) this.queue.shift();
      else               this.queue.push(this.queue.shift());
    });
  },

  again() {
    if (!this.queue.length) return;
    const word = this.vocab[this.queue[0]];
    if (!word) return;
    const sceneId = this.scenes[this.sceneIdx] && this.scenes[this.sceneIdx].id;
    fetch(`${API}/api/vocab/mastery`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${authToken}` },
      body:    JSON.stringify({ scene_id: sceneId, word_en: word.en, correct: false }),
    }).catch(() => {});
    this.animateExit('left', () => {
      this.queue.push(this.queue.shift());
    });
  },

  showAllMastered() {
    const inner = document.getElementById('wordsCardInner');
    const acts  = document.getElementById('wordsActions');
    const rem   = document.getElementById('wordsRemaining');
    if (acts) acts.classList.remove('visible');
    if (rem)  rem.textContent = '';
    if (!inner) return;
    inner.classList.remove('flipped', 'exit-left', 'exit-right');
    inner.innerHTML =
      '<div class="words-face" style="background:linear-gradient(145deg, rgba(25,18,8,0.6), rgba(10,8,3,0.4)); gap:1rem;">' +
        '<div style="font-size:36px; letter-spacing:8px; color:#c8a96e;">★★★</div>' +
        '<div style="font-family:\'Bebas Neue\',sans-serif; font-size:1.4rem; letter-spacing:0.18em; color:#fff;">ALL MASTERED</div>' +
        '<button id="wordsRestartBtn" style="margin-top:0.5rem; padding:0.6rem 1.2rem; border-radius:18px; border:none; outline:1px solid rgba(200,169,110,0.35); background:rgba(200,169,110,0.1); color:#c8a96e; font-family:\'DM Sans\',sans-serif; cursor:pointer;">Restart</button>' +
      '</div>';
    const rb = document.getElementById('wordsRestartBtn');
    if (rb) rb.addEventListener('click', (e) => { e.stopPropagation(); this.init(); });
  },

  _renderEmpty() {
    const inner = document.getElementById('wordsCardInner');
    const acts  = document.getElementById('wordsActions');
    const rem   = document.getElementById('wordsRemaining');
    if (acts) acts.classList.remove('visible');
    if (rem)  rem.textContent = '';
    if (!inner) return;
    inner.classList.remove('flipped', 'exit-left', 'exit-right');
    inner.innerHTML =
      '<div class="words-face" style="background:linear-gradient(145deg, rgba(15,25,45,0.55), rgba(5,10,20,0.35)); gap:0.5rem;">' +
        '<div style="font-size:30px; margin-bottom:6px;">🔒</div>' +
        '<div style="font-family:\'Bebas Neue\',sans-serif; font-size:1.2rem; letter-spacing:0.18em; color:#fff;">NO VOCAB YET</div>' +
        '<div style="font-size:0.74rem; color:rgba(255,255,255,0.4); text-align:center; max-width:240px; line-height:1.5;">Practice any scene 3 times and score 70%+ to unlock its vocabulary deck.</div>' +
      '</div>';
  },

  _shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  },
};

function wireWordsControls() {
  const card = document.getElementById('wordsCard');
  if (card) card.addEventListener('click', () => WordsController.flipCard());
  const got = document.getElementById('wordsGotItBtn');
  if (got)  got.addEventListener('click', (e) => { e.stopPropagation(); WordsController.gotIt(); });
  const ag  = document.getElementById('wordsAgainBtn');
  if (ag)   ag.addEventListener('click',  (e) => { e.stopPropagation(); WordsController.again(); });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', wireWordsControls);
} else {
  wireWordsControls();
}

// Expose globally so the inline switchTab() in index.html can call init()
window.WordsController = WordsController;

// ══════════════════════════════════════════════
// RANKS — social/standings/rewards controller
// ══════════════════════════════════════════════
const RanksController = {
  data: null,
  loaded: false,

  init() {
    this.bindTabs();
    this.bindShare();
    if (!this.loaded) this.load();
  },

  bindTabs() {
    document.querySelectorAll('.ranks-tab').forEach(tab => {
      tab.onclick = () => {
        document.querySelectorAll('.ranks-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.ranks-section').forEach(s => s.classList.remove('active'));
        tab.classList.add('active');
        const target = tab.dataset.tab;
        if (target === 'feed') document.getElementById('ranksFeed').classList.add('active');
        if (target === 'standings') document.getElementById('ranksStandings').classList.add('active');
        if (target === 'rewards') document.getElementById('ranksRewards').classList.add('active');
      };
    });
  },

  bindShare() {
    const btn = document.getElementById('ranksShareBtn');
    if (btn) btn.onclick = () => this.shareProgress();
  },

  async load() {
    const authToken = localStorage.getItem('mirror_token');
    if (!authToken) return;
    try {
      const res = await fetch('/api/ranks/social', { headers: { 'Authorization': 'Bearer ' + authToken } });
      if (!res.ok) return;
      this.data = await res.json();
      this.loaded = true;
      this.render();
    } catch(e) { console.error('RanksController load error', e); }
  },

  render() {
    const d = this.data;
    if (!d) return;
    this.renderFeed(d);
    this.renderStandings(d);
    this.renderRewards(d);
  },

  renderFeed(d) {
    document.getElementById('ranksPctNum').textContent = d.percentile + '%';
    document.getElementById('rStat1').textContent = d.scenes_completed;
    document.getElementById('rStat2').textContent = d.streak;
    document.getElementById('rStat3').textContent = d.weekly_xp.toLocaleString();
    document.getElementById('rStat4').textContent = d.words_mastered;

    const dotsEl = document.getElementById('ranksDotsRow');
    if (dotsEl) {
      const total = 15;
      const lit = Math.round((d.percentile / 100) * total);
      dotsEl.innerHTML = Array.from({length: total}, (_, i) =>
        `<span class="ranks-dot ${i < lit ? 'ranks-dot-on' : 'ranks-dot-off'}"></span>`
      ).join('');
    }

    const motivations = [
      { title: 'Consistency is your superpower.', sub: "You're showing up when most people don't." },
      { title: 'The camera is always rolling.',   sub: 'Every practice makes you a better actor.' },
      { title: 'Stars are made through repetition.', sub: 'Keep going — the Director sees everything.' },
    ];
    const m = motivations[Math.floor(Math.random() * motivations.length)];
    document.getElementById('ranksMotivationTitle').textContent = m.title;
    document.getElementById('ranksMotivationSub').textContent = m.sub;

    const feedEl = document.getElementById('ranksFeedList');
    if (!feedEl || !d.feed) return;
    const timeAgo = (iso) => {
      const diff = (Date.now() - new Date(iso)) / 1000;
      if (diff < 60) return 'just now';
      if (diff < 3600) return Math.floor(diff/60) + 'm ago';
      if (diff < 86400) return Math.floor(diff/3600) + 'h ago';
      return Math.floor(diff/86400) + 'd ago';
    };
    feedEl.innerHTML = d.feed.map(f => {
      const score = Math.round(f.score);
      const badge = score >= 90 ? '<span class="ranks-feed-badge badge-gold-pill">★ ' + score + '%</span>'
                  : score >= 70 ? '<span class="ranks-feed-badge badge-blue-pill">' + score + '%</span>'
                  : '<span class="ranks-feed-badge badge-fire">🔥 ' + score + '%</span>';
      return `<div class="ranks-feed-item">
        <div class="ranks-avatar">${f.initials}</div>
        <div style="flex:1;">
          <div class="ranks-feed-name">${f.username}</div>
          <div class="ranks-feed-text">practiced <em style="color:rgba(255,255,255,0.55)">${f.scene_id.replace(/_/g,' ')}</em></div>
          ${badge}
        </div>
        <div class="ranks-feed-time">${timeAgo(f.created_at)}</div>
      </div>`;
    }).join('');
  },

  renderStandings(d) {
    const podiumEl = document.getElementById('ranksPodium');
    const lbEl     = document.getElementById('ranksLbList');
    const youEl    = document.getElementById('ranksYouRow');
    if (!d.top3 || !podiumEl) return;

    const podiumOrder = [d.top3[1], d.top3[0], d.top3[2]].filter(Boolean);
    const heights    = ['70px', '105px', '50px'];
    const sizes      = ['42px', '52px', '38px'];
    const colors     = ['#b8b8b8', '#c8a96e', '#cd7f32'];
    const labels     = ['2', '1', '3'];
    const fontSizes  = ['2.4rem', '2.8rem', '2.1rem'];
    const rankLabels = ['', 'CHAMPION', ''];

    podiumEl.innerHTML = podiumOrder.map((u, i) => `
      <div class="podium-col">
        ${rankLabels[i] ? `<div style="font-family:'Bebas Neue',sans-serif;font-size:0.58rem;color:rgba(200,169,110,0.55);letter-spacing:0.1em;">${rankLabels[i]}</div>` : ''}
        <div class="podium-uname" style="color:${colors[i]}">${u.username.substring(0,8)}</div>
        <div class="podium-av" style="width:${sizes[i]};height:${sizes[i]};background:${colors[i]}22;border:2px solid ${colors[i]};font-size:0.72rem;color:${colors[i]}">${u.initials}</div>
        <div class="podium-xp" style="color:${colors[i]}">${u.total_points.toLocaleString()} xp</div>
        <div class="podium-bar" style="height:${heights[i]};background:linear-gradient(180deg,${colors[i]}30,${colors[i]}0d);border-top:2px solid ${colors[i]};border-left:1px solid ${colors[i]}33;border-right:1px solid ${colors[i]}33;">
          <div class="podium-num" style="font-size:${fontSizes[i]};color:${colors[i]}">${labels[i]}</div>
        </div>
      </div>
    `).join('');

    if (!d.leaderboard || !lbEl) return;
    const rankColors = ['#c8a96e','#b8b8b8','#cd7f32'];
    lbEl.innerHTML = d.leaderboard.map((u, i) => {
      const rank = i + 1;
      const rc   = rank <= 3 ? rankColors[rank-1] : 'rgba(255,255,255,0.2)';
      const avBg = rank <= 3 ? `${rc}22` : 'rgba(255,255,255,0.05)';
      return `<div class="ranks-lb-row ${u.is_me ? 'is-me' : ''}">
        <div class="ranks-lb-rank" style="color:${rc}">${u.is_me ? 'YOU' : rank}</div>
        <div class="ranks-lb-av" style="background:${avBg};color:${rc};border:1px solid ${rc}44">${u.initials}</div>
        <div class="ranks-lb-name" style="${u.is_me ? 'color:#c8a96e' : ''}">${u.username}</div>
        <div class="ranks-lb-pts">${u.total_points.toLocaleString()} xp</div>
      </div>`;
    }).join('');

    const me     = d.leaderboard.find(u => u.is_me);
    const myRank = me ? d.leaderboard.indexOf(me) + 1 : null;
    if (me && youEl) {
      const toTop3 = myRank > 3 ? d.leaderboard[2].total_points - me.total_points : 0;
      youEl.innerHTML = `
        <div class="ranks-lb-rank" style="color:rgba(200,169,110,0.4);font-family:'Bebas Neue',sans-serif;font-size:1rem;">${myRank}</div>
        <div class="ranks-lb-av" style="width:32px;height:32px;background:rgba(200,169,110,0.15);border:1px solid rgba(200,169,110,0.35);color:#c8a96e;font-size:0.68rem;">${me.initials}</div>
        <div style="flex:1;">
          <div style="font-size:0.8rem;color:#c8a96e;font-weight:500;">You (${me.username})</div>
          ${toTop3 > 0 ? `<div style="font-size:0.62rem;color:rgba(255,255,255,0.3);margin-top:1px;">${toTop3.toLocaleString()} xp to top 3</div>` : '<div style="font-size:0.62rem;color:#c8a96e;margin-top:1px;">You\'re in the top 3!</div>'}
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.85rem;color:#c8a96e;font-weight:500;">${me.total_points.toLocaleString()} xp</div>
        </div>`;
    }
  },

  renderRewards(d) {
    const weeklyXp = d ? d.weekly_xp : 0;
    const userXp   = userProfile ? userProfile.total_points : 0;
    const rewards  = [
      { icon:'🎬', title:'$5 CINEMA GIFT CARD', sub:'AMC · Regal · Cinemark', cost:5000  },
      { icon:'🍿', title:'FREE POPCORN',         sub:'Partner cinemas only',  cost:2500  },
      { icon:'🏆', title:'DIRECTOR BADGE',       sub:'Exclusive profile flair', cost:10000 },
    ];
    const el = document.getElementById('ranksRewardsList');
    if (!el) return;
    el.innerHTML = rewards.map(r => {
      const pct       = Math.min(100, Math.round((userXp / r.cost) * 100));
      const remaining = Math.max(0, r.cost - userXp);
      const locked    = userXp < r.cost;
      return `<div class="ranks-reward-card ${locked ? 'locked' : ''}">
        <div class="ranks-reward-top">
          <div class="ranks-reward-icon">${r.icon}</div>
          <div>
            <div class="ranks-reward-title">${r.title}</div>
            <div class="ranks-reward-sub">${r.sub}</div>
          </div>
          <div class="ranks-reward-cost">
            <div class="ranks-reward-cost-num">${r.cost.toLocaleString()}</div>
            <div class="ranks-reward-cost-lbl">XP</div>
          </div>
        </div>
        <div class="ranks-progress-bar"><div class="ranks-progress-fill" style="width:${pct}%"></div></div>
        <div class="ranks-progress-labels">
          <span class="ranks-progress-cur">${userXp.toLocaleString()} / ${r.cost.toLocaleString()} XP</span>
          <span class="ranks-progress-rem">${remaining > 0 ? remaining.toLocaleString() + ' to go' : '✓ Unlocked'}</span>
        </div>
      </div>`;
    }).join('');
  },

  shareProgress() {
    const d = this.data;
    if (!d) return;
    const text = `I'm better than ${d.percentile}% of Mirror users! 🎬 ${d.scenes_completed} scenes completed · ${d.streak}-day streak · ${d.weekly_xp} XP this week. Practice English with iconic movie scenes → mirror-app-z8wr.onrender.com`;
    if (navigator.share) {
      navigator.share({ text });
    } else {
      navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('ranksShareBtn');
        if (btn) { btn.textContent = 'COPIED TO CLIPBOARD'; setTimeout(() => btn.textContent = 'SHARE MY PROGRESS', 2000); }
      });
    }
  },
};

window.RanksController = RanksController;

// ══════════════════════════════════════════════
// LEVEL 1 QUIZ — flashcard / multi-format quiz controller
// ══════════════════════════════════════════════
const QuizController = {
  questions: [], current: 0, score: 0, combo: 0, answered: false,
  typeScores: { movie_id: [0,0], word_order: [0,0], definition: [0,0], verb: [0,0] },
  wordSlots: [], wordBank: [],

  SCENES: [
    {id:'forrest_gump', movie:'Forrest Gump', quote:"Life is like a box of chocolates.", actor:'Tom Hanks', verb:{en:'to run',es:'correr'}, sentence:['Life','is','like','a','box','of','chocolates']},
    {id:'home_alone', movie:'Home Alone', quote:"Keep the change, ya filthy animal.", actor:'Macaulay Culkin', verb:{en:'to protect',es:'proteger'}, sentence:['Keep','the','change']},
    {id:'social_network', movie:'The Social Network', quote:"A million dollars isn't cool.", actor:'Justin Timberlake', verb:{en:'to create',es:'crear'}, sentence:['A','million','dollars','is','cool']},
    {id:'cast_away', movie:'Cast Away', quote:"I have made fire!", actor:'Tom Hanks', verb:{en:'to survive',es:'sobrevivir'}, sentence:['I','have','made','fire']},
    {id:'fight_club', movie:'Fight Club', quote:"The first rule of Fight Club is you do not talk about Fight Club.", actor:'Brad Pitt', verb:{en:'to fight',es:'pelear'}, sentence:['You','do','not','talk']},
    {id:'seven', movie:'Se7en', quote:"What's in the box?", actor:'Brad Pitt', verb:{en:'to discover',es:'descubrir'}, sentence:['What','is','in','the','box']},
    {id:'the_matrix', movie:'The Matrix', quote:"There is no spoon.", actor:'Keanu Reeves', verb:{en:'to choose',es:'elegir'}, sentence:['There','is','no','spoon']},
    {id:'men_in_black', movie:'Men in Black', quote:"I make this look good.", actor:'Will Smith', verb:{en:'to protect',es:'proteger'}, sentence:['I','make','this','look','good']},
    {id:'top_gun', movie:'Top Gun', quote:"I feel the need, the need for speed!", actor:'Tom Cruise', verb:{en:'to fly',es:'volar'}, sentence:['I','feel','the','need']},
    {id:'back_to_the_future', movie:'Back to the Future', quote:"Roads? Where we're going we don't need roads.", actor:'Christopher Lloyd', verb:{en:'to travel',es:'viajar'}, sentence:['We','do','not','need','roads']},
    {id:'the_blind_side', movie:'The Blind Side', quote:"You're changing that boy's life.", actor:'Sandra Bullock', verb:{en:'to change',es:'cambiar'}, sentence:['You','are','changing','his','life']},
    {id:'clueless', movie:'Clueless', quote:"As if!", actor:'Alicia Silverstone', verb:{en:'to argue',es:'discutir'}, sentence:['As','if']},
    {id:'the_intern', movie:'The Intern', quote:"Experience never gets old.", actor:'Robert De Niro', verb:{en:'to learn',es:'aprender'}, sentence:['Experience','never','gets','old']},
    {id:'mystic_river', movie:'Mystic River', quote:"Is that my daughter in there?", actor:'Sean Penn', verb:{en:'to lose',es:'perder'}, sentence:['Is','that','my','daughter']},
    {id:'mrs_doubtfire', movie:'Mrs. Doubtfire', quote:"Help is on the way, dear.", actor:'Robin Williams', verb:{en:'to help',es:'ayudar'}, sentence:['Help','is','on','the','way']},
    {id:'jerry_maguire', movie:'Jerry Maguire', quote:"You had me at hello.", actor:'Renée Zellweger', verb:{en:'to love',es:'amar'}, sentence:['You','had','me','at','hello']},
    {id:'apollo_13', movie:'Apollo 13', quote:"Houston, we have a problem.", actor:'Tom Hanks', verb:{en:'to solve',es:'resolver'}, sentence:['We','have','a','problem']},
    {id:'pursuit_of_happyness', movie:'The Pursuit of Happyness', quote:"Don't ever let somebody tell you you can't do something.", actor:'Will Smith', verb:{en:'to believe',es:'creer'}, sentence:['You','can','do','it']},
    {id:'fifth_element', movie:'The Fifth Element', quote:"Multipass!", actor:'Milla Jovovich', verb:{en:'to save',es:'salvar'}, sentence:['This','is','a','multipass']},
    {id:'devil_wears_prada', movie:'The Devil Wears Prada', quote:"That's all.", actor:'Meryl Streep', verb:{en:'to work',es:'trabajar'}, sentence:['That','is','all']},
  ],

  DEFS: [
    {en:'run',es_correct:'moverse rápido con las piernas',es_wrong:['cocinar con aceite caliente','escribir con un lápiz']},
    {en:'protect',es_correct:'defender a alguien de un peligro',es_wrong:['olvidar algo importante','cantar una canción']},
    {en:'survive',es_correct:'continuar viviendo después de un peligro',es_wrong:['comprar ropa nueva','hablar con un amigo']},
    {en:'choose',es_correct:'seleccionar una opción entre varias',es_wrong:['romper un objeto de vidrio','dormir hasta tarde']},
    {en:'believe',es_correct:'tener confianza en algo o alguien',es_wrong:['vender algo en el mercado','preparar la cena']},
    {en:'save',es_correct:'rescatar a alguien de un peligro',es_wrong:['perder el autobús','lavar los platos']},
    {en:'change',es_correct:'hacer que algo sea diferente',es_wrong:['subir a un árbol alto','beber agua fría']},
    {en:'learn',es_correct:'adquirir nuevos conocimientos',es_wrong:['romper una ventana','conducir muy rápido']},
  ],

  shuffle(arr) { return [...arr].sort(() => Math.random() - 0.5); },

  buildQuestions() {
    const q = [];
    const scenes = this.shuffle([...this.SCENES]);
    scenes.forEach((scene, i) => {
      const type = i % 4;
      if (type === 0) {
        const others = this.SCENES.filter(s => s.id !== scene.id);
        const wrongs = this.shuffle(others).slice(0,3).map(s => s.movie);
        q.push({ type:'movie_id', typeLabel:'¿De qué película es?', scene, question:`"${scene.quote}"`, options: this.shuffle([scene.movie,...wrongs]), answer: scene.movie });
      } else if (type === 1) {
        const words = this.shuffle([...scene.sentence]);
        q.push({ type:'word_order', typeLabel:'Ordena las palabras', scene, question:`Ordena estas palabras para formar la frase de ${scene.movie}:`, words, answer: scene.sentence.join(' ') });
      } else if (type === 2) {
        const def = this.DEFS.find(d => d.en === scene.verb.en) || this.DEFS[i % this.DEFS.length];
        const options = this.shuffle([def.es_correct, ...def.es_wrong.slice(0,2)]);
        q.push({ type:'definition', typeLabel:'Elige la definición', scene, question:`¿Qué significa "${scene.verb.en}" en español?`, options, answer: def.es_correct });
      } else {
        const coin = Math.random() > 0.5;
        const others = this.SCENES.filter(s => s.id !== scene.id);
        if (coin) {
          const wrongs = this.shuffle(others).slice(0,3).map(s => s.verb.es);
          q.push({ type:'verb', typeLabel:'Traducción', scene, question:`Traduce al español:\n\n"${scene.verb.en}"`, options: this.shuffle([scene.verb.es,...wrongs]), answer: scene.verb.es });
        } else {
          const wrongs = this.shuffle(others).slice(0,3).map(s => s.verb.en);
          q.push({ type:'verb', typeLabel:'Traducción', scene, question:`Traduce al inglés:\n\n"${scene.verb.es}"`, options: this.shuffle([scene.verb.en,...wrongs]), answer: scene.verb.en });
        }
      }
    });
    return q.slice(0,20);
  },

  open() {
    this.questions = this.buildQuestions();
    this.current = 0; this.score = 0; this.combo = 0; this.answered = false;
    this.typeScores = { movie_id:[0,0], word_order:[0,0], definition:[0,0], verb:[0,0] };
    document.getElementById('quizOverlay').style.display = 'block';
    document.body.style.overflow = 'hidden';
    document.getElementById('quizResults').style.display = 'none';
    ['quizQuestionNum','quizQuestionText','quizQuestionType','quizPosterWrap','quizOptions','quizWordOrder'].forEach(id => {
      const el = document.getElementById(id); if(el) el.style.display = '';
    });
    document.getElementById('quizNextBtn').style.display = 'none';
    document.getElementById('quizFeedback').style.display = 'none';
    document.getElementById('quizComboBadge').style.display = 'none';
    this.renderQuestion();
  },

  close() {
    document.getElementById('quizOverlay').style.display = 'none';
    document.body.style.overflow = '';
    setTimeout(() => { document.body.style.overflow = ''; document.documentElement.style.overflow = ''; }, 100);
  },

  renderQuestion() {
    const q = this.questions[this.current];
    const total = this.questions.length;
    document.getElementById('quizProgressFill').style.width = ((this.current/total)*100)+'%';
    document.getElementById('quizQuestionNum').textContent = 'PREGUNTA '+(this.current+1)+' DE '+total;
    document.getElementById('quizQuestionType').textContent = q.typeLabel;
    document.getElementById('quizQuestionText').textContent = q.question;
    document.getElementById('quizFeedback').style.display = 'none';
    document.getElementById('quizNextBtn').style.display = 'none';
    document.getElementById('quizComboBadge').style.display = 'none';
    this.answered = false;

    const img = document.getElementById('quizPosterImg');
    const movieLabel = document.getElementById('quizPosterMovie');
    if (img) { img.src = '/static/posters/'+q.scene.id+'.jpg'; img.onerror = () => { img.style.display='none'; }; }
    if (movieLabel) movieLabel.textContent = q.scene.movie;

    const optEl = document.getElementById('quizOptions');
    const woEl = document.getElementById('quizWordOrder');

    if (q.type === 'word_order') {
      optEl.style.display = 'none';
      woEl.style.display = 'block';
      this.wordSlots = [];
      this.wordBank = [...q.words];
      this.renderWordOrder(q);
    } else {
      optEl.style.display = 'flex';
      woEl.style.display = 'none';
      optEl.innerHTML = q.options.map(opt =>
        `<button class="quiz-opt" data-val="${opt}">${opt}</button>`
      ).join('');
      optEl.querySelectorAll('.quiz-opt').forEach(btn => {
        btn.onclick = () => this.selectAnswer(btn.dataset.val);
      });
    }
  },

  renderWordOrder(q) {
    const slotsEl = document.getElementById('quizWordSlots');
    const bankEl = document.getElementById('quizWordBank');
    slotsEl.innerHTML = this.wordSlots.map((w,i) =>
      `<span class="quiz-word-slot" data-i="${i}" onclick="QuizController.removeFromSlot(${i})">${w}</span>`
    ).join('') || '<span style="color:rgba(255,255,255,0.2);font-size:0.75rem;padding:0.3rem;">Toca las palabras para ordenarlas</span>';
    bankEl.innerHTML = this.wordBank.map((w,i) =>
      `<span class="quiz-word-tile" data-i="${i}" onclick="QuizController.addToSlot(${i})">${w}</span>`
    ).join('');
  },

  addToSlot(i) {
    if (this.answered) return;
    const word = this.wordBank[i];
    this.wordSlots.push(word);
    this.wordBank.splice(i,1);
    this.renderWordOrder(this.questions[this.current]);
    if (this.wordBank.length === 0) {
      this.checkWordOrder();
    }
  },

  removeFromSlot(i) {
    if (this.answered) return;
    const word = this.wordSlots[i];
    this.wordBank.push(word);
    this.wordSlots.splice(i,1);
    this.renderWordOrder(this.questions[this.current]);
  },

  checkWordOrder() {
    const q = this.questions[this.current];
    const userAnswer = this.wordSlots.join(' ');
    this.selectAnswer(userAnswer);
  },

  selectAnswer(selected) {
    if (this.answered) return;
    this.answered = true;
    const q = this.questions[this.current];
    const correct = selected.toLowerCase().trim() === q.answer.toLowerCase().trim();
    if (correct) {
      this.score++;
      this.combo++;
      this.typeScores[q.type][0]++;
    } else {
      this.combo = 0;
    }
    this.typeScores[q.type][1]++;

    if (this.combo >= 3) {
      const badge = document.getElementById('quizComboBadge');
      document.getElementById('quizComboNum').textContent = this.combo;
      badge.style.display = 'block';
    }

    if (q.type !== 'word_order') {
      document.querySelectorAll('.quiz-opt').forEach(btn => {
        btn.style.pointerEvents = 'none';
        if (btn.dataset.val === q.answer) { btn.style.background='rgba(200,169,110,0.15)'; btn.style.borderColor='rgba(200,169,110,0.5)'; btn.style.color='#c8a96e'; }
        else if (btn.dataset.val === selected && !correct) { btn.style.background='rgba(255,80,80,0.1)'; btn.style.borderColor='rgba(255,80,80,0.3)'; btn.style.color='rgba(255,100,100,0.8)'; }
      });
    }

    const fb = document.getElementById('quizFeedback');
    fb.style.display = 'block';
    fb.style.background = correct ? 'rgba(200,169,110,0.08)' : 'rgba(255,80,80,0.06)';
    fb.style.border = correct ? '0.5px solid rgba(200,169,110,0.2)' : '0.5px solid rgba(255,80,80,0.15)';
    fb.style.color = correct ? '#c8a96e' : 'rgba(255,120,120,0.8)';
    fb.textContent = correct ? '✓ ¡Correcto!' : '✗ La respuesta correcta es: '+q.answer;

    const nextBtn = document.getElementById('quizNextBtn');
    nextBtn.style.display = 'block';
    nextBtn.textContent = this.current < this.questions.length-1 ? 'SIGUIENTE →' : 'VER RESULTADOS →';
  },

  next() {
    this.current++;
    if (this.current >= this.questions.length) this.showResults();
    else this.renderQuestion();
  },

  async showResults() {
    const total = this.questions.length;
    const pct = Math.round((this.score/total)*100);
    const passed = pct >= 70;
    ['quizQuestionNum','quizQuestionText','quizQuestionType','quizPosterWrap','quizOptions','quizWordOrder','quizFeedback','quizNextBtn','quizComboBadge'].forEach(id => {
      const el = document.getElementById(id); if(el) el.style.display='none';
    });
    document.getElementById('quizProgressFill').style.width='100%';
    document.getElementById('quizResults').style.display='block';
    document.getElementById('quizResultIcon').textContent = passed ? '🏆' : '🎬';
    document.getElementById('quizResultTitle').textContent = passed ? '¡NIVEL 2 DESBLOQUEADO!' : 'SIGUE PRACTICANDO';
    document.getElementById('quizResultScore').textContent = pct+'% — '+this.score+'/'+total+' correctas'+(passed ? '' : ' · Necesitas 70% para pasar');

    const typeLabels = { movie_id:'Películas', word_order:'Ordenar', definition:'Definiciones', verb:'Verbos' };
    const bd = document.getElementById('quizBreakdown');
    bd.innerHTML = Object.entries(this.typeScores).map(([k,[c,t]]) =>
      `<div class="quiz-breakdown-card">
        <div class="quiz-breakdown-num">${t > 0 ? Math.round((c/t)*100) : 0}%</div>
        <div class="quiz-breakdown-lbl">${typeLabels[k] || k}</div>
      </div>`
    ).join('');

    if (passed) {
      try {
        const authToken = localStorage.getItem('mirror_token');
        const res = await fetch('/api/quiz-pass', {
          method: 'POST',
          headers: {
            'Authorization': 'Bearer ' + authToken,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ quiz: 'level1', score: pct })
        });
        if (res.ok) {
          if (typeof userProgress !== 'undefined') userProgress._quizPassed = true;
          if (typeof checkLevel2Unlock === 'function') checkLevel2Unlock();
        }
      } catch(e) { console.error('quiz-pass error', e); }
    }
  },

  init() {
    document.getElementById('quizNextBtn').onclick = () => this.next();
    document.getElementById('quizCloseBtn').onclick = () => this.close();
    document.getElementById('quizDoneBtn').onclick = () => this.close();
  }
};

function wireQuizControls() { QuizController.init(); }
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', wireQuizControls);
} else {
  wireQuizControls();
}

window.QuizController = QuizController;
function openQuiz() { QuizController.open(); }
window.openQuiz = openQuiz;

// ══════════════════════════════════════════════
// MISSIONS — daily quest, weekly challenge, active missions, XP toasts
// ══════════════════════════════════════════════
const MISSION_META = {
  daily:           { icon: '🎬', title: "Today's Daily Scene",         xp: 100 },
  pronunciation:   { icon: '⭐', title: 'Score 85%+ on 5 takes',       xp:  75 },
  genre_drama:     { icon: '🎭', title: 'Practice 5 drama scenes',     xp: 150 },
  sprint:          { icon: '⚡', title: 'Finish a scene under 4 min',  xp:  50 },
  weekly_thriller: { icon: '🔪', title: 'Practice 3 thriller scenes',  xp: 200 },
};

const MissionsController = {
  data:   null,
  loaded: false,

  init() {
    this.bindDailyCta();
    this.load();
  },

  async load() {
    const tok = localStorage.getItem('mirror_token');
    if (!tok) return;
    try {
      const r = await fetch(`${API}/api/missions`, {
        headers: { Authorization: 'Bearer ' + tok },
      });
      if (!r.ok) return;
      this.data = await r.json();
      this.loaded = true;
      this.render();
    } catch (e) {
      console.error('MissionsController load failed', e);
    }
  },

  render() {
    const d = this.data;
    if (!d) return;

    // Today's XP pill
    const xpEl   = document.getElementById('mpTodayXp');
    const goalEl = document.getElementById('mpTodayGoal');
    if (xpEl)   xpEl.textContent   = (d.today_xp || 0).toLocaleString();
    if (goalEl) goalEl.textContent = (d.daily_xp_goal || 1000).toLocaleString();

    // Streak banner
    const s = d.streak || {};
    const tag   = document.getElementById('mpStreakTag');
    const title = document.getElementById('mpStreakTitle');
    const sub   = document.getElementById('mpStreakSub');
    if (tag)   tag.textContent   = '🔥 ' + (s.current || 0) + '-Day Streak';
    if (title) title.textContent = (s.current || 0) > 0
        ? "Great work — keep your streak alive."
        : 'Complete a mission today to start your streak';
    if (sub)   sub.textContent   = 'Longest: ' + (s.longest || 0) + ' days · Total XP: ' + (s.total_xp || 0).toLocaleString();

    // Daily quest card
    const dq = d.daily_quest;
    const dailyMovie = document.getElementById('mpDailyMovie');
    const dailyQuote = document.getElementById('mpDailyQuote');
    const dailyBar   = document.getElementById('mpDailyBar');
    const dailyCount = document.getElementById('mpDailyCount');
    const dailyCta   = document.getElementById('mpDailyCta');
    const dailySid   = (typeof userProfile !== 'undefined' && userProfile && userProfile.daily_scene_id) || '';
    const dailyScene = (typeof scenes !== 'undefined' && scenes && dailySid) ? scenes[dailySid] : null;
    if (dailyMovie) dailyMovie.textContent = dailyScene ? (dailyScene.movie || dailyScene.title || dailySid) : '—';
    if (dailyQuote) {
      const q = dailyScene ? (dailyScene.quote || '') : '';
      dailyQuote.textContent = q ? '"' + (q.length > 90 ? q.slice(0, 90) + '…' : q) + '"' : '';
    }
    if (dq && dailyBar)   dailyBar.style.width = Math.min(100, (dq.progress / dq.goal) * 100) + '%';
    if (dq && dailyCount) dailyCount.textContent = dq.progress + '/' + dq.goal;
    if (dailyCta) {
      if (dq && dq.completed) {
        dailyCta.textContent = '✓ Completed';
        dailyCta.style.background = 'rgba(106,170,46,0.2)';
        dailyCta.style.color = '#6aaa2e';
        dailyCta.style.cursor = 'default';
      } else {
        dailyCta.textContent = 'Practice →';
        dailyCta.style.background = '#c8a96e';
        dailyCta.style.color = '#0d0d0d';
        dailyCta.style.cursor = 'pointer';
      }
    }

    // Weekly challenge card
    const wc       = d.weekly_challenge;
    const wBar     = document.getElementById('mpWeeklyBar');
    const wCount   = document.getElementById('mpWeeklyCount');
    if (wc && wBar)   wBar.style.width = Math.min(100, (wc.progress / wc.goal) * 100) + '%';
    if (wc && wCount) wCount.textContent = wc.progress + '/' + wc.goal;

    // Active missions list (excluding the two surfaced above)
    const list = document.getElementById('mpMissionList');
    if (!list) return;
    const surfaced = new Set(['daily', 'weekly_thriller']);
    const items    = (d.active_missions || []).filter(m => !surfaced.has(m.mission_id));
    list.innerHTML = items.map(m => {
      const meta = MISSION_META[m.mission_id] || { icon: '📌', title: m.mission_id, xp: 100 };
      const pct  = Math.min(100, (m.progress / m.goal) * 100);
      return (
        '<div class="mp-mission ' + (m.completed ? 'completed' : '') + '" data-mission-id="' + m.mission_id + '">' +
          '<div class="mp-mission-icon">' + meta.icon + '</div>' +
          '<div class="mp-mission-body">' +
            '<div class="mp-mission-title">' + meta.title + '</div>' +
            '<div class="mp-mission-bar-wrap">' +
              '<div class="mp-mission-bar"><div class="mp-mission-bar-fill" style="width:' + pct + '%"></div></div>' +
              '<div class="mp-mission-count">' + m.progress + '/' + m.goal + '</div>' +
            '</div>' +
          '</div>' +
          '<div class="mp-mission-xp">' + (m.completed ? '✓ ' : '+') + meta.xp + ' XP</div>' +
        '</div>'
      );
    }).join('');
  },

  bindDailyCta() {
    const cta = document.getElementById('mpDailyCta');
    if (!cta) return;
    cta.onclick = () => {
      if (this.data && this.data.daily_quest && this.data.daily_quest.completed) return;
      const sid = (typeof userProfile !== 'undefined' && userProfile && userProfile.daily_scene_id) || '';
      if (sid && typeof scenes !== 'undefined' && scenes && scenes[sid] && typeof openModal === 'function') {
        openModal(sid, scenes[sid]);
      }
    };
  },

  /**
   * Hook for /api/submit responses. Bumps mission progress bars in-place and
   * shows XP toasts for any newly-completed missions, without a full reload.
   */
  onSubmitResponse(resp) {
    if (!resp || !Array.isArray(resp.missions_updated) || !resp.missions_updated.length) return;

    // Update local cache so re-renders are accurate
    if (this.data && Array.isArray(this.data.active_missions)) {
      const byId = {};
      this.data.active_missions.forEach(m => { byId[m.mission_id] = m; });
      resp.missions_updated.forEach(u => {
        if (byId[u.mission_id]) {
          byId[u.mission_id].progress  = u.new_progress;
          byId[u.mission_id].completed = u.completed;
        }
      });
      // Bump today_xp + total_xp on the local snapshot
      if (resp.total_xp_earned) {
        this.data.today_xp = (this.data.today_xp || 0) + resp.total_xp_earned;
        if (this.data.streak) this.data.streak.total_xp = (this.data.streak.total_xp || 0) + resp.total_xp_earned;
      }
    }

    // Re-render so visible bars and counters reflect new state
    if (this.loaded) this.render();

    // Toasts for completions only (more meaningful than every increment)
    resp.missions_updated.forEach(u => {
      if (u.completed && u.xp_earned > 0) this.showXpToast(u.xp_earned, u.mission_id);
    });
    // If progress advanced but nothing completed, still show a single combined toast
    const completedCount = resp.missions_updated.filter(u => u.completed).length;
    if (completedCount === 0 && resp.total_xp_earned > 0) {
      this.showXpToast(resp.total_xp_earned, null);
    }
  },

  showXpToast(amount, missionId) {
    const wrap = document.getElementById('mpXpToastWrap');
    if (!wrap) return;
    const meta = missionId && MISSION_META[missionId];
    const label = meta ? (meta.icon + ' +' + amount + ' XP') : ('+' + amount + ' XP');
    const el = document.createElement('div');
    el.className = 'mp-xp-toast';
    el.textContent = label;
    wrap.appendChild(el);
    setTimeout(() => { if (el.parentNode) el.parentNode.removeChild(el); }, 3200);
  },
};

window.MissionsController = MissionsController;
