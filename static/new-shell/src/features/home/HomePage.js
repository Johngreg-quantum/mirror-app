import { renderLeaderboardPanel } from '../../components/LeaderboardPanel.js';
import { renderProgressStatCard } from '../../components/ProgressStatCard.js';
import { renderSceneCard } from '../../components/SceneCard.js';
import { renderStreakCard } from '../../components/StreakCard.js';
import { renderLoggedErrorState, renderLoadingState } from '../../components/AsyncState.js';
import { buttonLink, card, statusPill } from '../../components/primitives.js';
import { h } from '../../lib/helpers/dom.js';
import { getFreshPostScoreReadCache } from '../../lib/api/post-score-refresh.js';
import { loadPersonalReadData, loadPublicHomeData } from '../../lib/api/read-data.js';
import { adaptLeaderboard } from '../../lib/adapters/leaderboard-adapter.js';
import { adaptDailyChallenge } from '../../lib/adapters/daily-adapter.js';
import { adaptProgressSummary, adaptProfile } from '../../lib/adapters/progress-adapter.js';
import { adaptSceneConfig } from '../../lib/adapters/scene-adapter.js';
import { createAppHref } from '../../lib/routing/navigation.js';
import { sceneHref } from '../../lib/routing/scene-routes.js';

export function renderHomePage({ appState, actions }) {
  const page = h('div', {}, [renderLoadingState('Loading scene browser')]);

  loadHomeViewModel({ appState, actions })
    .then((viewModel) => {
      page.replaceChildren(renderHomeSurface({ appState, ...viewModel }));
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
  const daily = adaptDailyChallenge(publicData.daily, scenes, profile);
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
    daily,
    leaderboard,
    profile,
    progressSummary,
    personalError,
  };
}

function renderHomeActionPanel({ daily, scenes }) {
  const firstOpenScene = scenes.find((scene) => !scene.locked) || scenes[0] || daily?.scene;
  const practiceHref = firstOpenScene
    ? sceneHref(firstOpenScene.id, { from: 'home' })
    : createAppHref('/daily');

  return h('section', { className: 'ns-home-action-panel' }, [
    h('div', { className: 'ns-home-action-panel__primary' }, [
      h('p', { className: 'ns-eyebrow', text: 'Start here' }),
      h('h3', { text: daily?.scene?.title || 'Today\'s practice' }),
      h('p', {
        text: daily?.scene
          ? `${daily.scene.film} is live for the daily. One recorded take starts your streak, score, and next challenge.`
          : 'Choose one scene, record one take, and get your first score on the board.',
      }),
      h('div', { className: 'ns-action-row' }, [
        buttonLink({
          href: daily?.scene ? sceneHref(daily.scene.id, { from: 'daily' }) : practiceHref,
          text: daily?.scene ? 'Start daily take' : 'Start practice',
        }),
        buttonLink({ href: practiceHref, text: 'Pick a scene', variant: 'secondary' }),
      ]),
    ]),
    h('div', { className: 'ns-home-action-panel__challenge' }, [
      h('span', { text: 'Challenge loop' }),
      h('strong', { text: 'Score. Improve. Challenge.' }),
      h('p', { text: 'A saved score gives you a benchmark to beat again or turn into a friend challenge later.' }),
    ]),
  ]);
}

function renderHomeSurface({ appState, scenes, daily, leaderboard, profile, progressSummary, personalError }) {
  return h('article', { className: 'ns-page ns-home-page' }, [
    h('section', { className: 'ns-home-hero' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Practice room' }),
        h('h2', { text: profile ? `Record the next take, ${profile.displayName}` : 'Get your first score' }),
        h('p', {
          text: 'Pick a scene, record one take, analyze it, then let Mirror point you to the next move.',
        }),
        h('div', { className: 'ns-action-row' }, [
          buttonLink({
            href: daily?.scene ? sceneHref(daily.scene.id, { from: 'daily' }) : createAppHref('/daily'),
            text: 'Start practicing now',
          }),
          buttonLink({ href: createAppHref('/daily'), text: 'Keep daily streak', variant: 'secondary' }),
          buttonLink({ href: createAppHref('/progress'), text: 'View progress', variant: 'secondary' }),
        ]),
      ]),
      daily
        ? renderHomeActionPanel({ daily, scenes })
        : profile
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
          h('p', { className: 'ns-eyebrow', text: 'Choose fast' }),
          h('h2', { text: 'Pick a first take' }),
        ]),
        statusPill('Record / analyze / continue'),
      ]),
      scenes.length
        ? h('div', { className: 'ns-carousel-wrap' }, [
            h('button', {
              className: 'ns-carousel-prev',
              on: { click: () => { const c = document.querySelector('.ns-scene-carousel'); if (c) c.scrollBy({ left: -240, behavior: 'smooth' }); } },
              text: '←',
            }),
            h('div', { className: 'ns-scene-carousel' }, scenes.map((scene) => renderSceneCard({ scene, entrySource: 'home' }))),
            h('button', {
              className: 'ns-carousel-next',
              on: { click: () => { const c = document.querySelector('.ns-scene-carousel'); if (c) c.scrollBy({ left: 240, behavior: 'smooth' }); } },
              text: '→',
            }),
          ])
        : card({ title: 'No scenes found', body: 'Scenes will appear here when the catalog is ready.' }),
    ]),
    h('div', { className: 'ns-grid ns-grid--two' }, [
      leaderboard.rows.length
        ? renderLeaderboardPanel({ leaderboard, entrySource: 'leaderboard' })
        : card({ title: 'Leaderboard is empty', body: 'Scores will appear here after scored takes are submitted.' }),
      card({
        eyebrow: 'Momentum',
        title: 'Your next action is the product',
        body: 'A score should never be a dead end. Each take can lead into a retry, a next scene, the daily loop, progress, or a challenge.',
        className: 'ns-support-card',
      }),
    ]),
  ]);
}
