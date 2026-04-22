import { renderLoggedErrorState, renderLoadingState } from '../../components/AsyncState.js';
import { renderProgressStatCard } from '../../components/ProgressStatCard.js';
import { buttonLink, card, statusPill } from '../../components/primitives.js';
import { h } from '../../lib/helpers/dom.js';
import { getFreshPostScoreReadCache } from '../../lib/api/post-score-refresh.js';
import { fetchHistory, fetchProfile, fetchProgress, fetchSceneConfig } from '../../lib/api/read-data.js';
import {
  adaptFocusAreas,
  adaptPersonalBests,
  adaptProgressSummary,
  adaptProfile,
  adaptRecentHistory,
} from '../../lib/adapters/progress-adapter.js';
import { adaptSceneConfig } from '../../lib/adapters/scene-adapter.js';
import { createAppHref } from '../../lib/routing/navigation.js';
import { sceneHref } from '../../lib/routing/scene-routes.js';

export function renderProgressDashboardPage({ appState }) {
  const page = h('div', {}, [renderLoadingState('Loading progress dashboard')]);

  loadProgressViewModel(appState)
    .then((viewModel) => {
      page.replaceChildren(renderProgressSurface(viewModel));
    })
    .catch((error) => {
      page.replaceChildren(renderLoggedErrorState(error, {
        title: 'Progress dashboard needs sign-in',
        surface: 'progress',
      }));
    });

  return page;
}

async function loadProgressViewModel(appState) {
  const postScoreCache = getFreshPostScoreReadCache(appState);
  const [sceneConfig, progress, profile, history] = await Promise.all([
    postScoreCache?.sceneConfig && !postScoreCache?.errors?.sceneConfig
      ? Promise.resolve(postScoreCache.sceneConfig)
      : fetchSceneConfig(),
    postScoreCache?.progress && !postScoreCache?.errors?.progress
      ? Promise.resolve(postScoreCache.progress)
      : fetchProgress(),
    postScoreCache?.profile && !postScoreCache?.errors?.profile
      ? Promise.resolve(postScoreCache.profile)
      : fetchProfile(),
    postScoreCache?.history && !postScoreCache?.errors?.history
      ? Promise.resolve(postScoreCache.history)
      : fetchHistory(),
  ]);
  const { scenes } = adaptSceneConfig(sceneConfig, { progress });

  return {
    recommendedScene: scenes.find((scene) => !scene.locked) || scenes[0] || null,
    profile: adaptProfile(profile),
    progressSummary: adaptProgressSummary({ progress, profile, history }),
    personalBests: adaptPersonalBests({ progress, scenes }),
    recentHistory: adaptRecentHistory(history),
    focusAreas: adaptFocusAreas(history),
  };
}

function renderProgressSurface({ recommendedScene, profile, progressSummary, personalBests, recentHistory, focusAreas }) {
  const hasScores = recentHistory.length > 0 || personalBests.length > 0;

  return h('article', { className: 'ns-page ns-progress-page' }, [
    h('header', { className: 'ns-page__header' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Progress' }),
        h('h2', { text: hasScores ? 'Progress dashboard' : 'Start your progress story' }),
        h('p', {
          className: 'ns-page__summary',
          text: hasScores
            ? `${profile.displayName} is in ${profile.division} with ${profile.points.toLocaleString()} points. Use the next take to move one signal forward.`
            : `${profile.displayName} is ready to build a baseline. One scored take unlocks history, personal bests, and focus areas.`,
        }),
      ]),
      h('div', { className: 'ns-inline-list' }, [
        statusPill('Synced'),
        hasScores ? statusPill('Practice history active') : statusPill('First score needed'),
      ]),
    ]),
    h('div', { className: 'ns-grid ns-grid--four' }, [
      renderProgressStatCard({ label: 'Average', value: progressSummary.scoreAverage, detail: 'all scored takes' }),
      renderProgressStatCard({ label: 'Scenes', value: progressSummary.scenesCompleted, detail: 'completed' }),
      renderProgressStatCard({ label: 'PBs', value: progressSummary.personalBests, detail: 'set so far' }),
      renderProgressStatCard({ label: 'Unlocked', value: progressSummary.unlockedScenes, detail: 'ready scenes' }),
    ]),
    h('div', { className: 'ns-grid ns-grid--three' }, [
      card({
        eyebrow: 'Best marks',
        title: 'Personal bests',
        body: personalBests.length
          ? 'Best-scoring scenes from your saved progress data.'
          : 'Your first scored take becomes the baseline Mirror can help you beat.',
        className: 'ns-support-card',
        children: [
          personalBests.length
            ? h('ul', {}, personalBests.map((best) => h('li', { text: `${best.sceneTitle} - ${best.score}` })))
            : h('div', { className: 'ns-empty-cta' }, [
                h('p', { className: 'ns-muted', text: 'Start with one short scene and come back here after analyze.' }),
                recommendedScene
                  ? buttonLink({ href: sceneHref(recommendedScene.id, { from: 'progress' }), text: 'Record first score', variant: 'secondary' })
                  : buttonLink({ href: createAppHref('/'), text: 'Choose a scene', variant: 'secondary' }),
              ]),
        ],
      }),
      card({
        eyebrow: 'Recent reps',
        title: 'Recent history',
        body: recentHistory.length
          ? 'Recent scored takes from your saved history.'
          : 'Recent scores appear here after analyze, turning practice into a visible streak of reps.',
        className: 'ns-support-card',
        children: [
          recentHistory.length
            ? h('ul', {}, recentHistory.map((item) => h('li', { text: `${item.sceneTitle}: ${item.score} (${item.result})` })))
            : h('div', { className: 'ns-empty-cta' }, [
                h('p', { className: 'ns-muted', text: 'No reps yet. Do the daily or pick one scene to start the timeline.' }),
                buttonLink({ href: createAppHref('/daily'), text: 'Do today\'s daily', variant: 'secondary' }),
              ]),
        ],
      }),
      card({
        eyebrow: 'Next focus',
        title: 'Focus areas',
        body: hasScores
          ? 'Patterns from recent scored takes, shaped into simple practice focus.'
          : 'Once you have a few takes, this becomes your lightweight coaching prompt.',
        className: 'ns-support-card',
        children: [
          h('ul', {}, focusAreas.map((area) => h('li', { text: area }))),
        ],
      }),
    ]),
    card({
      eyebrow: 'Next step',
      title: hasScores ? 'Use progress to choose the next rep' : 'Get one score on the board',
      body: hasScores
        ? 'Progress should tell you where to practice next: retry a lower mark, protect the daily, or move to a fresh scene.'
        : 'The dashboard gets warmer after the first analyze result. Start with one scene and let the score create your baseline.',
      className: 'ns-support-card',
      children: [
        h('div', { className: 'ns-action-row' }, [
          recommendedScene
            ? buttonLink({ href: sceneHref(recommendedScene.id, { from: 'progress' }), text: hasScores ? 'Practice recommended scene' : 'Record first score' })
            : buttonLink({ href: createAppHref('/'), text: 'Choose a scene' }),
          buttonLink({ href: createAppHref('/daily'), text: 'Open daily', variant: 'secondary' }),
          buttonLink({ href: createAppHref('/'), text: 'Back home', variant: 'secondary' }),
        ]),
      ],
    }),
  ]);
}
