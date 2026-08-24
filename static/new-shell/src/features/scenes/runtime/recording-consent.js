import { h } from '../../../lib/helpers/dom.js';
import { acceptRecordingConsent } from '../../../lib/api/consent.js';
import { logFrontendError, trackEvent } from '../../../lib/observability.js';

// First-run recording consent (privacy policy §6).
//
// The new shell has its own recorder (runtime/media-recorder.js) and its own
// entry point, so it needs its own gate — the legacy shell's gate in
// static/app.js does not cover /app/*, /scene/:id or /challenge/:id, which this
// shell serves. Both gates hit the same POST /api/consent/recording and read the
// same users.recording_consent_at, so accepting in either shell counts.
//
// Acceptance is server-side state, not localStorage, so it survives a device
// change and is asked once per account.

const ACCEPT_LABEL = 'Got it — start recording';

function buildNotice({ onAccept, onDismiss }) {
  const errorEl = h('div', { className: 'ns-consent__error' });
  const acceptButton = h('button', {
    className: 'ns-button ns-button--primary ns-consent__accept',
    type: 'button',
    text: ACCEPT_LABEL,
    on: { click: () => onAccept({ acceptButton, errorEl }) },
  });

  const box = h('div', {
    className: 'ns-consent__box',
    attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'nsConsentTitle' },
  }, [
    h('h2', {
      className: 'ns-consent__title',
      id: 'nsConsentTitle',
      text: 'Before you record',
    }),
    h('p', { className: 'ns-consent__body' }, [
      'When you submit a take, your audio is sent to ',
      h('strong', { text: 'OpenAI' }),
      ' to be turned into text. They delete it within 30 days and never use it for training.',
    ]),
    h('p', { className: 'ns-consent__body' }, [
      "We don't keep the audio. We do keep the ",
      h('strong', { text: 'transcript and your score' }),
      ', saved to your account so you can see your progress. Deleting your account erases both.',
    ]),
    errorEl,
    h('div', { className: 'ns-consent__actions' }, [
      h('a', {
        className: 'ns-consent__legal',
        href: '/privacy',
        target: '_blank',
        rel: 'noopener',
        text: 'Privacy Policy',
      }),
      acceptButton,
    ]),
  ]);

  const overlay = h('div', {
    className: 'ns-consent',
    on: {
      click: (event) => {
        if (event.target === overlay) {
          onDismiss();
        }
      },
    },
  }, [box]);

  return { overlay, acceptButton, errorEl };
}

/**
 * Returns an async gate: `await gate()` resolves true only once the user has
 * accepted and the server has stored it. Declining, dismissing, or a failed
 * save all resolve false, and the caller must not record.
 */
export function createRecordingConsentGate({ hasConsented = false, onConsented } = {}) {
  let consented = Boolean(hasConsented);
  let pending = null;

  return function requireRecordingConsent() {
    if (consented) {
      return Promise.resolve(true);
    }

    // A second tap while the notice is open joins the same decision rather than
    // stacking another overlay.
    if (pending) {
      return pending;
    }

    pending = new Promise((resolve) => {
      let settled = false;
      let notice = null;

      function close(result) {
        if (settled) {
          return;
        }

        settled = true;
        document.removeEventListener('keydown', onKey);
        notice?.overlay.remove();
        pending = null;
        resolve(result);
      }

      function onKey(event) {
        if (event.key === 'Escape') {
          close(false);
        }
      }

      async function onAccept({ acceptButton, errorEl }) {
        acceptButton.disabled = true;
        acceptButton.textContent = 'Saving…';
        errorEl.textContent = '';

        try {
          await acceptRecordingConsent();
          // Only treat as consented once the server confirms, so a failed save
          // shows the notice again instead of silently letting recording start.
          consented = true;
          onConsented?.();
          trackEvent('recording_consent_accepted', {});
          close(true);
        } catch (error) {
          logFrontendError(error, {
            phase: 'recording-consent',
            surface: 'scene-runtime',
          });
          acceptButton.disabled = false;
          acceptButton.textContent = ACCEPT_LABEL;
          errorEl.textContent = 'Could not save your choice. Please try again.';
        }
      }

      notice = buildNotice({ onAccept, onDismiss: () => close(false) });
      document.body.appendChild(notice.overlay);
      document.addEventListener('keydown', onKey);
      // Next frame, so the opacity transition runs from the initial state.
      window.requestAnimationFrame(() => notice.overlay.classList.add('is-open'));
      notice.acceptButton.focus();
    });

    return pending;
  };
}
