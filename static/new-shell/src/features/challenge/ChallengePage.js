import { renderLoggedErrorState, renderLoadingState } from '../../components/AsyncState.js';
import { renderChallengeResultCard } from '../../components/ChallengeResultCard.js';
import { renderSessionPrompt } from '../../components/SessionState.js';
import { buttonLink, card, statusPill } from '../../components/primitives.js';
import { h } from '../../lib/helpers/dom.js';
import { fetchChallengeEntry } from '../../lib/api/challenge.js';
import { adaptChallengeEntry, adaptChallengeResult } from '../../lib/adapters/challenge-adapter.js';
import { createAppHref } from '../../lib/routing/navigation.js';
import { scenePath } from '../../lib/routing/scene-routes.js';
import { trackEvent } from '../../lib/observability.js';
import { getStoredChallengeEntry, getStoredChallengeResult, storeChallengeEntry } from '../../state/app-state.js';

function buildChallengeScenePath(challengeEntry) {
  return scenePath(challengeEntry.sceneId, {
    from: 'challenge',
    challengeId: challengeEntry.id,
  });
}

function buildChallengeAuthPath(challengeEntry) {
  const challengeScenePath = buildChallengeScenePath(challengeEntry);
  return createAppHref(`/auth?redirect=${encodeURIComponent(challengeScenePath)}`);
}

async function loadChallengeViewModel(appState, challengeId) {
  const storedResult = getStoredChallengeResult(appState, challengeId);
  const cachedEntry = getStoredChallengeEntry(appState, challengeId)?.raw
    || storedResult?.challengeEntry
    || null;
  const rawChallenge = cachedEntry || await fetchChallengeEntry(challengeId);
  storeChallengeEntry(appState, challengeId, rawChallenge);

  const challengeEntry = adaptChallengeEntry(rawChallenge);
  const challengeResult = adaptChallengeResult({
    challengeEntry,
    analyzeResult: storedResult?.analyzeResult || null,
  });

  return {
    challengeEntry,
    challengeResult,
    storedResult,
  };
}

function renderChallengeRouteError(challengeId, error = null) {
  return h('article', { className: 'ns-page' }, [
    renderLoggedErrorState(error || new Error(`Challenge ${challengeId} could not load right now.`), {
      title: 'Challenge could not load',
      surface: 'challenge',
    }),
    card({
      title: 'Rollback',
      body: 'Use the rollback challenge link while this invite is being checked.',
      children: [
        buttonLink({
          href: `/legacy/challenge/${encodeURIComponent(challengeId)}`,
          text: 'Open rollback challenge',
          variant: 'secondary',
        }),
      ],
    }),
  ]);
}

function renderChallengeEntryCard({ challengeEntry, isAuthenticated }) {
  const primaryHref = isAuthenticated
    ? createAppHref(buildChallengeScenePath(challengeEntry))
    : buildChallengeAuthPath(challengeEntry);

  return h('section', { className: 'ns-challenge-entry ns-challenge-entry--hero' }, [
    h('div', { className: 'ns-challenge-entry__copy' }, [
      h('p', { className: 'ns-eyebrow', text: 'Incoming challenge' }),
      h('h3', { text: `${challengeEntry.challengerName} challenged you` }),
      h('p', { text: `${challengeEntry.sceneTitle} from ${challengeEntry.film}. Beat the score, bank the points, then decide whether to run it back.` }),
    ]),
    h('div', { className: 'ns-challenge-entry__benchmark' }, [
      h('span', { text: 'Score to beat' }),
      h('strong', { text: challengeEntry.targetScoreLabel }),
    ]),
    h('div', { className: 'ns-inline-list' }, [
      statusPill(challengeEntry.createdLabel),
      statusPill(isAuthenticated ? 'Ready to record' : 'Sign-in handoff'),
    ]),
    h('div', { className: 'ns-action-row ns-challenge-entry__actions' }, [
      buttonLink({
        href: primaryHref,
        text: isAuthenticated ? 'Record your take' : 'Sign in to accept',
      }),
      buttonLink({
        href: createAppHref(buildChallengeScenePath(challengeEntry)),
        text: 'Open challenge scene',
        variant: 'secondary',
      }),
    ]),
  ]);
}

function renderChallengeResultSummary({ challengeEntry, challengeResult, challengeSceneHref }) {
  if (!challengeResult) {
    return card({
      eyebrow: 'Aftermath',
      title: 'The board is waiting',
      body: 'Start the attempt and your score will land here with a clear win or retry path.',
      className: 'ns-challenge-aftermath ns-support-card',
      children: [
        h('div', { className: 'ns-inline-list' }, [
          statusPill('Awaiting scored take'),
          statusPill(`Beat ${challengeEntry.targetScoreLabel}`),
        ]),
        buttonLink({
          href: challengeSceneHref,
          text: 'Start the attempt',
          variant: 'secondary',
        }),
      ],
    });
  }

  const isWin = challengeResult.outcome === 'won';

  return card({
    eyebrow: 'Aftermath',
    title: isWin ? 'Make the win count' : 'Run it back',
    body: challengeResult.message,
    className: `ns-challenge-aftermath ns-challenge-aftermath--${isWin ? 'win' : 'loss'}`,
    children: [
      h('div', { className: 'ns-inline-list' }, [
        statusPill(challengeResult.comparisonLabel),
        statusPill(`${challengeResult.yourScore} yours`),
        statusPill(`${challengeResult.opponentScore} to beat`),
        statusPill(`${challengeResult.pointsEarned} points`),
        statusPill(challengeResult.streakLabel),
      ]),
      h('div', { className: 'ns-action-row ns-challenge-aftermath__actions' }, [
        buttonLink({
          href: challengeSceneHref,
          text: isWin ? 'Defend with another take' : 'Try again',
          variant: 'secondary',
        }),
        buttonLink({
          href: createAppHref('/progress'),
          text: 'View progress',
          variant: 'secondary',
        }),
      ]),
    ],
  });
}

export function renderChallengePage({ appState, params }) {
  const challengeId = params.challengeId || 'pending';
  const page = h('div', {}, [renderLoadingState('Loading challenge')]);

  loadChallengeViewModel(appState, challengeId)
    .then(({ challengeEntry, challengeResult }) => {
      const isAuthenticated = appState.session.status === 'authenticated';
      const challengeSceneHref = createAppHref(buildChallengeScenePath(challengeEntry));

      trackEvent('challenge_opened', {
        challengeId: challengeEntry.id,
        sceneId: challengeEntry.sceneId,
        hasResult: Boolean(challengeResult),
      });

      if (challengeResult) {
        trackEvent('challenge_completed', {
          challengeId: challengeEntry.id,
          sceneId: challengeEntry.sceneId,
          outcome: challengeResult.outcome,
          yourScore: challengeResult.yourScore,
          opponentScore: challengeResult.opponentScore,
        });
      }

      page.replaceChildren(h('article', { className: 'ns-page ns-challenge-page' }, [
        h('header', { className: 'ns-page__header' }, [
          h('div', {}, [
            h('p', { className: 'ns-eyebrow', text: 'Challenge' }),
            h('h2', { text: `Beat ${challengeEntry.targetScoreLabel}` }),
            h('p', {
              className: 'ns-page__summary',
            text: `${challengeEntry.challengerName} set the benchmark on ${challengeEntry.sceneTitle}. Record a take, beat the score, then keep the rivalry moving.`,
            }),
          ]),
          h('div', { className: 'ns-inline-list' }, [
            statusPill(appState.session.status),
            statusPill(challengeEntry.targetScoreLabel),
          ]),
        ]),
        renderSessionPrompt({
          session: appState.session,
          title: isAuthenticated
            ? `Signed in as ${appState.session.user?.displayName || 'performer'}`
            : 'Sign in to accept this challenge',
          body: isAuthenticated
            ? 'Launch the scene with this benchmark attached, then analyze a take to settle the board.'
            : 'After sign-in, Mirror sends you into the challenge scene to record your take.',
        }),
        renderChallengeEntryCard({ challengeEntry, isAuthenticated }),
        h('div', { className: 'ns-grid ns-grid--two' }, [
          renderChallengeResultCard({ entry: challengeEntry, result: challengeResult }),
          renderChallengeResultSummary({ challengeEntry, challengeResult, challengeSceneHref }),
        ]),
        card({
          eyebrow: 'Next action',
          title: challengeResult ? 'Keep the rivalry moving' : 'Challenge scene launch',
          body: isAuthenticated
            ? challengeResult
              ? 'Try again from the same scene, improve the benchmark, or use the score as fuel for the next invite.'
              : 'Open the linked scene with challenge context preserved and score the take there.'
            : 'Sign in first, then Mirror brings this challenge context into the scene.',
          className: 'ns-challenge-launch-card ns-support-card',
          children: [
            h('div', { className: 'ns-inline-list' }, [
              statusPill(challengeEntry.sceneTitle),
              statusPill('Challenge context saved'),
              buttonLink({
                href: challengeSceneHref,
                text: 'Open challenge scene',
                variant: 'secondary',
              }),
            ]),
          ],
        }),
      ]));
    })
    .catch((error) => {
      page.replaceChildren(renderChallengeRouteError(challengeId, error));
    });

  return page;
}
