import { h } from '../../../lib/helpers/dom.js';
import { acceptRecordingConsent } from '../../../lib/api/consent.js';
import { logFrontendError, trackEvent } from '../../../lib/observability.js';

// First-run recording consent (privacy policy §6).
//
// The new shell has its own recorder (runtime/media-recorder.js) and its own
// entry point, so it needs its own notice — the legacy shell's gate in
// static/app.js does not cover /app/*, /scene/:id or /challenge/:id, which this
// shell serves. Both hit the same POST /api/consent/recording and read the same
// users.recording_consent_at, so accepting in either shell counts.
//
// SHAPE: this is NOT a promise the record button awaits. getUserMedia needs a
// user gesture, and on iOS Safari that activation does not reliably survive a
// network round-trip. So the unconsented tap shows the notice and stops, and
// the notice's own Accept click — a fresh gesture — is what starts recording.
// Nothing is awaited between a tap and getUserMedia.

const ACCEPT_LABEL = 'Got it — start recording';

export function createRecordingConsentNotice({ hasConsented = false, onAccept } = {}) {
  let consented = Boolean(hasConsented);
  let overlay = null;

  function close() {
    document.removeEventListener('keydown', onKey);
    overlay?.remove();
    overlay = null;
  }

  function onKey(event) {
    if (event.key === 'Escape') {
      close();
    }
  }

  function handleAccept() {
    // ORDER IS LOAD-BEARING. onAccept() issues getUserMedia and must be the
    // first thing this handler does, so it runs inside the Accept click's
    // gesture. Persisting consent happens after, and never blocks it.
    consented = true;
    onAccept?.();
    close();

    acceptRecordingConsent()
      .then(() => {
        trackEvent('recording_consent_accepted', {});
      })
      .catch((error) => {
        // Deliberately not surfaced: the take is already running and the user
        // did consent. Not persisting means they are asked again next session.
        consented = false;
        logFrontendError(error, {
          phase: 'recording-consent-persist',
          surface: 'scene-runtime',
        });
      });
  }

  return {
    hasConsented() {
      return consented;
    },
    show() {
      if (overlay) {
        return;
      }

      const box = h('div', {
        className: 'ns-consent__box',
        attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'nsConsentTitle' },
      }, [
        h('h2', { className: 'ns-consent__title', id: 'nsConsentTitle', text: 'Before you record' }),
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
        h('div', { className: 'ns-consent__actions' }, [
          h('a', {
            className: 'ns-consent__legal',
            href: '/privacy',
            target: '_blank',
            rel: 'noopener',
            text: 'Privacy Policy',
          }),
          h('button', {
            className: 'ns-button ns-button--primary ns-consent__accept',
            type: 'button',
            text: ACCEPT_LABEL,
            on: { click: handleAccept },
          }),
        ]),
      ]);

      overlay = h('div', {
        className: 'ns-consent',
        on: {
          click: (event) => {
            // Backdrop tap declines. No mic was acquired, nothing to release.
            if (event.target === overlay) {
              close();
            }
          },
        },
      }, [box]);

      document.body.appendChild(overlay);
      document.addEventListener('keydown', onKey);
      window.requestAnimationFrame(() => overlay?.classList.add('is-open'));
      box.querySelector('.ns-consent__accept')?.focus();
    },
    cleanup: close,
  };
}
