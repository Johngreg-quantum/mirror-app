import { renderLoggedErrorState, renderLoadingState } from '../../components/AsyncState.js';
import { renderDailyChallengeCard } from '../../components/DailyChallengeCard.js';
import { renderSceneCard } from '../../components/SceneCard.js';
import { renderSessionPrompt } from '../../components/SessionState.js';
import { renderStreakCard } from '../../components/StreakCard.js';
import { card, statusPill } from '../../components/primitives.js';
import { h } from '../../lib/helpers/dom.js';
import { getFreshPostScoreReadCache } from '../../lib/api/post-score-refresh.js';
import { fetchDailyChallenge, fetchProfile, fetchSceneConfig } from '../../lib/api/read-data.js';
import { adaptDailyChallenge } from '../../lib/adapters/daily-adapter.js';
import { adaptProfile } from '../../lib/adapters/progress-adapter.js';
import { adaptSceneConfig } from '../../lib/adapters/scene-adapter.js';

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
  const lineTotal = daily.lineTotal || 1;

  return h('article', { className: 'ns-page' }, [
    h('header', { className: 'ns-page__header' }, [
      h('div', {}, [
        h('p', { className: 'ns-eyebrow', text: 'Daily Take' }),
        h('h2', { text: 'Daily Take' }),
        h('p', {
          className: 'ns-page__summary',
          text: lineTotal > 1
            ? `${lineTotal} short lines, one sitting. Finish all ${lineTotal} to count the day towards your streak.`
            : 'Practice today\'s scene and keep your streak moving when your session is active.',
        }),
      ]),
      statusPill(daily.progressLabel),
    ]),
    renderSessionPrompt({
      session,
      title: 'Streak data needs sign-in',
      body: 'Today\'s lines are public. Which ones you have recorded appears after your session is verified.',
    }),
    renderDailyChallengeCard({ daily }),
    h('div', { className: 'ns-grid ns-grid--two' }, [
      profile
        ? renderStreakCard({ profile })
        : card({
            title: 'Streak data needs sign-in',
            body: profileError?.message || 'Sign in to show streak status here.',
            children: [statusPill(profileError?.rateLimited ? 'Rate limited' : 'Session')],
          }),
      // The next line to record, as a full scene card. Once the set is done
      // there is no next line, so this falls back to the first.
      renderSceneCard({ scene: daily.scene, entrySource: 'daily' }),
    ]),
    h('div', { className: 'ns-grid ns-grid--two' }, [
      card({
        title: 'How the set is scored',
        body: lineTotal > 1
          ? `Each line pays its own points as you record it. Finishing all ${lineTotal} pays a completion award on the average of the set, doubled, plus a bonus when every line clears 70%.`
          : 'Points, streak bonus, and reset timing update after a scored daily take.',
        children: [statusPill(daily.streakBonus)],
      }),
      card({
        title: 'When the lines change',
        body: 'A new set appears at midnight where you are, so an evening session is never interrupted by the rotation. Progress is read from your recorded takes, so it survives a reload or a switch of device — and a set left unfinished at midnight is replaced rather than carried over.',
        children: [statusPill(daily.resetLabel)],
      }),
    ]),
  ]);
}
