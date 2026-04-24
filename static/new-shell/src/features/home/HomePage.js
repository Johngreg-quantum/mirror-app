import { renderLeaderboardPanel } from '../../components/LeaderboardPanel.js';
import { renderProgressStatCard } from '../../components/ProgressStatCard.js';
import { renderStreakCard } from '../../components/StreakCard.js';
import { renderLoggedErrorState, renderLoadingState } from '../../components/AsyncState.js';
import { buttonLink, card, statusPill } from '../../components/primitives.js';
import { h } from '../../lib/helpers/dom.js';
import { getFreshPostScoreReadCache } from '../../lib/api/post-score-refresh.js';
import { loadPersonalReadData, loadPublicHomeData } from '../../lib/api/read-data.js';
import { adaptLeaderboard } from '../../lib/adapters/leaderboard-adapter.js';
import { adaptProgressSummary, adaptProfile } from '../../lib/adapters/progress-adapter.js';
import { adaptSceneConfig } from '../../lib/adapters/scene-adapter.js';
import { createAppHref } from '../../lib/routing/navigation.js';

let nsCfCenter = 0;
let nsCfTotal = 0;

function nsCfRotate(dir) {
  const wrap = document.getElementById('nsCoverflowWrap');
  const titleEl = document.getElementById('nsCoverflowTitle');
  const actorEl = document.getElementById('nsCoverflowActor');
  if (!wrap) return;

  const cards = wrap.querySelectorAll('.ns-cf-card');
  nsCfTotal = cards.length;
  nsCfCenter = (nsCfCenter + dir + nsCfTotal) % nsCfTotal;
  wrap.dataset.center = String(nsCfCenter);

  const slots = [
    { x: -440, z: -320, ry: 62,  b: 0.15, zi: 1 },
    { x: -320, z: -230, ry: 56,  b: 0.3,  zi: 3 },
    { x: -185, z: -130, ry: 46,  b: 0.55, zi: 6 },
    { x: 0,    z: 0,    ry: 0,   b: 1,    zi: 10 },
    { x: 185,  z: -130, ry: -46, b: 0.55, zi: 6 },
    { x: 320,  z: -230, ry: -56, b: 0.3,  zi: 3 },
    { x: 440,  z: -320, ry: -62, b: 0.15, zi: 1 },
  ];

  cards.forEach((card, i) => {
    let offset = i - nsCfCenter;
    if (offset > nsCfTotal / 2) offset -= nsCfTotal;
    if (offset < -nsCfTotal / 2) offset += nsCfTotal;
    const slotIdx = offset + 3;
    const slot = slotIdx >= 0 && slotIdx <= 6
      ? slots[slotIdx]
      : { x: (offset > 0 ? 600 : -600), z: -400, ry: offset > 0 ? -65 : 65, b: 0, zi: 0 };

    card.style.transform = `translateX(${slot.x}px) translateZ(${slot.z}px) rotateY(${slot.ry}deg)`;
    card.style.filter = `brightness(${slot.b})`;
    card.style.zIndex = slot.zi;

    const poster = card.querySelector('.ns-cf-poster');
    if (slot.zi === 10 && poster) {
      poster.style.boxShadow = '0 30px 80px rgba(0,0,0,0.85), 0 0 0 0.5px rgba(200,169,110,0.35), 0 0 50px rgba(200,169,110,0.12)';
      poster.style.borderColor = 'rgba(200,169,110,0.4)';
    } else if (poster) {
      poster.style.boxShadow = '';
      poster.style.borderColor = 'rgba(200,169,110,0.15)';
    }
  });

  if (titleEl && actorEl) {
    const centerCard = cards[nsCfCenter];
    if (centerCard) {
      titleEl.textContent = centerCard.querySelector('.ns-cf-poster img')?.alt || '';
    }
  }
}

export function renderHomePage({ appState, actions }) {
  const page = h('div', {}, [renderLoadingState('Loading scene browser')]);

  loadHomeViewModel({ appState, actions })
    .then((viewModel) => {
      page.replaceChildren(renderHomeSurface({ appState, ...viewModel }));
      requestAnimationFrame(() => requestAnimationFrame(() => {
        nsCfTotal = document.querySelectorAll('.ns-cf-card').length;
        nsCfRotate(0);
      }));
    })
    .catch((error) => {
      page.replaceChildren(renderLoggedErrorState(error, {
        title: 'Scene browser could not load',
        surface: 'home',
      }));
    });

  return page;
}

async function loadHomeViewModel({ appState, actions }) {
  const postScoreCache = getFreshPostScoreReadCache(appState);
  const publicData = (
    postScoreCache?.sceneConfig
    && !postScoreCache?.errors?.sceneConfig
    && postScoreCache?.daily
    && !postScoreCache?.errors?.daily
    && postScoreCache?.leaderboard
    && !postScoreCache?.errors?.leaderboard
  )
    ? {
        sceneConfig: postScoreCache.sceneConfig,
        daily: postScoreCache.daily,
        leaderboard: postScoreCache.leaderboard,
      }
    : await loadPublicHomeData();
  await actions.session?.waitForInitialSession?.();
  const session = appState.session;
  let personalData = null;
  let personalError = null;

  if (session?.status === 'authenticated') {
    if (
      postScoreCache?.progress
      && !postScoreCache?.errors?.progress
      && postScoreCache?.profile
      && !postScoreCache?.errors?.profile
      && postScoreCache?.history
      && !postScoreCache?.errors?.history
    ) {
      personalData = {
        progress: postScoreCache.progress,
        profile: postScoreCache.profile,
        history: postScoreCache.history,
      };
    } else {
      try {
        personalData = await loadPersonalReadData();
      } catch (error) {
        personalError = error;
      }
    }
  }

  const { scenes } = adaptSceneConfig(publicData.sceneConfig, {
    progress: personalData?.progress || null,
    daily: publicData.daily,
  });
  const leaderboard = adaptLeaderboard(publicData.leaderboard, scenes, publicData.daily.scene_id);
  const profile = adaptProfile(personalData?.profile);
  const progressSummary = personalData
    ? adaptProgressSummary(personalData)
    : {
        scoreAverage: scenes.length ? '--' : 0,
        scenesCompleted: scenes.length,
        personalBests: '--',
        unlockedScenes: scenes.length,
      };

  return {
    scenes,
    leaderboard,
    profile,
    progressSummary,
    personalError,
  };
}

function renderHomeSurface({ appState, scenes, leaderboard, profile, progressSummary, personalError }) {
  return h('article', { className: 'ns-page' }, [
    h('section', { className: 'ns-home-hero' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Scene browser' }),
        h('h2', { text: profile ? `Welcome back, ${profile.displayName}` : 'Choose a scene' }),
        h('p', {
          text: 'Browse scenes, check today\'s challenge, and keep your practice streak moving.',
        }),
        h('div', { className: 'ns-action-row' }, [
          buttonLink({ href: createAppHref('/daily'), text: 'Daily challenge' }),
          buttonLink({ href: createAppHref('/progress'), text: 'View progress', variant: 'secondary' }),
        ]),
      ]),
      profile
        ? renderStreakCard({ profile })
        : card({
            title: 'Personal data paused',
            body: personalError?.message || 'Sign in to personalize streak, locks, and progress here.',
            children: [statusPill(personalError?.rateLimited ? 'Rate limited' : 'Session auth')],
          }),
    ]),
    h('div', { className: 'ns-grid ns-grid--four' }, [
      renderProgressStatCard({ label: 'Average', value: progressSummary.scoreAverage, detail: 'practice average' }),
      renderProgressStatCard({ label: 'Completed', value: progressSummary.scenesCompleted, detail: 'scenes finished' }),
      renderProgressStatCard({ label: 'PBs', value: progressSummary.personalBests, detail: 'personal bests set' }),
      renderProgressStatCard({ label: 'Visible', value: progressSummary.unlockedScenes, detail: 'available scenes' }),
    ]),
    h('section', { className: 'ns-stack' }, [
      h('div', { className: 'ns-section-heading' }, [
        h('div', {}, [
          h('p', { className: 'ns-eyebrow', text: 'Scenes' }),
          h('h2', { text: 'Scene browser' }),
        ]),
        statusPill('Live scene config'),
      ]),
      h('div', { className: 'ns-coverflow-section' }, [
        h('div', { className: 'ns-coverflow-wrap', id: 'nsCoverflowWrap' }, [
          h('div', { className: 'ns-coverflow-stage' }, [
            h('div', { className: 'ns-coverflow-track', id: 'nsCoverflowTrack' },
              scenes.map((scene, i) => h('div', {
                className: 'ns-cf-card',
                attrs: { 'data-index': String(i), 'data-scene-id': scene.id },
                on: { click: () => {
                  const wrap = document.getElementById('nsCoverflowWrap');
                  if (!wrap) return;
                  const currentCenter = parseInt(wrap.dataset.center || '0');
                  if (i === currentCenter) {
                    window.location.href = `/app/scene/${scene.id}`;
                  } else {
                    const diff = i - currentCenter;
                    nsCfRotate(diff > 0 ? 1 : -1);
                  }
                } },
              }, [
                h('div', { className: 'ns-cf-poster' }, [
                  h('img', { src: scene.imageUrl, alt: scene.film }),
                ]),
                h('div', { className: 'ns-cf-reflection' }, [
                  h('img', { src: scene.imageUrl, alt: '' }),
                ]),
              ]))
            ),
            h('div', { className: 'ns-coverflow-floor' }),
          ]),
          h('div', { className: 'ns-coverflow-info', id: 'nsCoverflowInfo' }, [
            h('div', { className: 'ns-coverflow-title', id: 'nsCoverflowTitle',
              text: scenes[0] ? (scenes[0].film || scenes[0].title || '') : '' }),
            h('div', { className: 'ns-coverflow-actor', id: 'nsCoverflowActor',
              text: scenes[0] ? (scenes[0].actor || '') : '' }),
          ]),
          h('div', { className: 'ns-coverflow-arrows' }, [
            h('button', { className: 'ns-cf-arrow ns-cf-prev', on: { click: () => nsCfRotate(-1) } }, ['‹']),
            h('button', { className: 'ns-cf-arrow ns-cf-next', on: { click: () => nsCfRotate(1) } }, ['›']),
          ]),
        ]),
      ]),
    ]),
    h('div', { className: 'ns-grid ns-grid--two' }, [
      leaderboard.rows.length
        ? renderLeaderboardPanel({ leaderboard, entrySource: 'leaderboard' })
        : card({ title: 'Leaderboard is empty', body: 'Scores will appear here after scored takes are submitted.' }),
      card({
        title: 'Practice sync',
        body: 'Scene availability, leaderboard rows, progress, and profile data stay in sync with your latest scored takes.',
      }),
    ]),
  ]);
}
