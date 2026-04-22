import { renderLoggedErrorState, renderLoadingState } from '../../components/AsyncState.js';
import { renderDailyChallengeCard } from '../../components/DailyChallengeCard.js';
import { renderSceneCard } from '../../components/SceneCard.js';
import { renderSessionPrompt } from '../../components/SessionState.js';
import { renderStreakCard } from '../../components/StreakCard.js';
import { buttonLink, card, statusPill } from '../../components/primitives.js';
import { h } from '../../lib/helpers/dom.js';
import { getFreshPostScoreReadCache } from '../../lib/api/post-score-refresh.js';
import { fetchDailyChallenge, fetchProfile, fetchSceneConfig } from '../../lib/api/read-data.js';
import { adaptDailyChallenge } from '../../lib/adapters/daily-adapter.js';
import { adaptProfile } from '../../lib/adapters/progress-adapter.js';
import { adaptSceneConfig } from '../../lib/adapters/scene-adapter.js';
import { createAppHref } from '../../lib/routing/navigation.js';

export function renderDailyChallengePage({ appState }) {
  const page = h('div', {}, [renderLoadingState('Loading daily challenge')]);

  loadDailyViewModel(appState)
    .then((viewModel) => {
      page.replaceChildren(renderDailySurface(viewModel));
    })
    .catch((error) => {
      page.replaceChildren(renderLoggedErrorState(error, {
        title: 'Daily challenge could not load',
        surface: 'daily',
      }));
    });

  return page;
}

async function loadDailyViewModel(appState) {
  const session = appState.session;
  const postScoreCache = getFreshPostScoreReadCache(appState);
  const [sceneConfig, rawDaily] = await Promise.all([
    postScoreCache?.sceneConfig && !postScoreCache?.errors?.sceneConfig
      ? Promise.resolve(postScoreCache.sceneConfig)
      : fetchSceneConfig(),
    postScoreCache?.daily && !postScoreCache?.errors?.daily
      ? Promise.resolve(postScoreCache.daily)
      : fetchDailyChallenge(),
  ]);
  let rawProfile = null;
  let profileError = null;

  if (session?.status === 'authenticated') {
    if (postScoreCache?.profile && !postScoreCache?.errors?.profile) {
      rawProfile = postScoreCache.profile;
    } else {
      try {
        rawProfile = await fetchProfile();
      } catch (error) {
        profileError = error;
      }
    }
  }

  const { scenes } = adaptSceneConfig(sceneConfig, { daily: rawDaily });
  const profile = adaptProfile(rawProfile);
  const daily = adaptDailyChallenge(rawDaily, scenes, profile);

  return {
    daily,
    profile,
    profileError,
    session,
  };
}

function renderDailySurface({ daily, profile, profileError, session }) {
  const dailyCompleted = profile?.dailyStatus && !/ready|not/i.test(profile.dailyStatus);

  return h('article', { className: 'ns-page ns-daily-page' }, [
    h('header', { className: 'ns-page__header' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Daily challenge' }),
        h('h2', { text: dailyCompleted ? 'Daily locked in' : 'Keep the streak alive' }),
        h('p', {
          className: 'ns-page__summary',
          text: dailyCompleted
            ? `${daily.scene.title} is banked for today. Come back after reset to keep the habit warm.`
            : `Today is worth ${daily.rewardPoints} points plus ${daily.streakBonus}. Record one take before reset.`,
        }),
      ]),
      statusPill(daily.resetLabel),
    ]),
    renderSessionPrompt({
      session,
      title: 'Streak data needs sign-in',
      body: 'The daily scene is public. Streak status appears after your session is verified.',
    }),
    renderDailyChallengeCard({ daily }),
    h('div', { className: 'ns-grid ns-grid--two' }, [
      profile
        ? renderStreakCard({ profile })
        : card({
            eyebrow: 'Habit state',
            title: 'Start tracking your streak',
            body: profileError?.message || 'Sign in to attach daily completions, streak status, and points to your profile.',
            className: 'ns-support-card',
            children: [
              statusPill(profileError?.rateLimited ? 'Rate limited' : 'Session'),
              buttonLink({ href: createAppHref('/auth'), text: 'Sign in', variant: 'secondary' }),
            ],
          }),
      renderSceneCard({ scene: daily.scene, entrySource: 'daily' }),
    ]),
    h('div', { className: 'ns-grid ns-grid--two' }, [
      card({
        eyebrow: 'Reward loop',
        title: dailyCompleted ? 'Come back tomorrow' : 'One take protects the habit',
        body: dailyCompleted
          ? 'Your daily state is reflected here. Tomorrow brings a fresh scene, fresh points, and another streak moment.'
          : 'Score the daily scene to update points, streak status, and the next best action across Mirror.',
        className: 'ns-support-card',
        children: [
          h('div', { className: 'ns-inline-list' }, [
            statusPill(`${daily.rewardPoints} points`),
            statusPill(daily.streakBonus),
            statusPill(daily.resetLabel),
          ]),
          h('div', { className: 'ns-action-row' }, [
            buttonLink({ href: createAppHref('/progress'), text: 'Open progress', variant: 'secondary' }),
            buttonLink({ href: createAppHref('/'), text: 'Pick another scene', variant: 'secondary' }),
          ]),
        ],
      }),
      card({
        eyebrow: 'Why daily matters',
        title: 'Make Mirror easy to return to',
        body: 'The daily scene gives you a short, repeatable loop: record, analyze, see progress, then return tomorrow with momentum.',
        className: 'ns-support-card',
      }),
    ]),
  ]);
}
