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

let mediaRecorder = null;
let audioChunks   = [];
let audioBlob     = null;
let audioEl       = null;
let micStream     = null;
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
  clearAuth();
  scenes      = {};
  activeLbTab = null;
  showAuthScreen();
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
}

function closeAuthModal() {
  setOverlayActive('authModalOverlay', false);
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
    if (r.ok) userProgress = await r.json();
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

  const poster = getScenePoster(sid);
  const posterImg = document.getElementById('heroPosterImg');
  if (posterImg && poster) {
    posterImg.src = poster;
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
