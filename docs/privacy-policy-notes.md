# Privacy policy — internal notes

Working notes for `privacy-policy.md`. **Not published** — only
`privacy-policy.md` is routed (`GET /privacy` in `main.py`); nothing else in
`docs/` is served, and `/static` is mounted from `static/`, not from here.

Keep notes-to-self in this file. The published policy should contain only
things a user needs to know — including honest disclosures of what the app does
*not* yet do. There are currently **no open items**: both the §4 erasure gap and
the §6 consent gap have shipped, and the policy describes real behaviour
throughout. If a future gap opens, disclose it in the policy as a blockquote and
track it under a heading here, the way those two were.

## Needs review by someone qualified

The policy is framed throughout around **UK/EU GDPR** — Art. 6 lawful bases,
the ICO named as supervisory authority, Standard Contractual Clauses in §9 —
while the data controller is a **Florida sole trader** (see the header and
§11). That combination deserves a professional read:

- GDPR can reach a US business serving EU users via Art. 3(2), but the SCC
  mechanism in §9 assumes a data exporter established in the EEA. Whether SCCs
  are the right instrument here, or whether something else applies, is not a
  question to answer by editing wording.
- A US-based operator serving US users also has Florida and US federal
  obligations that this document does not address at all.
- §8 sets the age floor at 13, which is the COPPA line rather than the
  GDPR Art. 8 range (13–16 depending on member state). Fine if intentional;
  worth confirming.

## Closed: §6 consent to recording

The first-run recording notice shipped. §6 describes the real flow and §3 no
longer points at an open item. This one broke production on its first attempt,
so the notes are specific:

- **There are two shells and therefore two notices.** `static/app.js` serves `/`
  and `/legacy`; `static/new-shell/` serves `/app/*`, `/scene/:id` and
  `/challenge/:id`. Each has its own recorder and its own `getUserMedia` call.
  **Any third recording entry point needs its own gate**, or `/scene/:id` — the
  shareable link format — records with no notice while §6 claims otherwise.
- **NEVER await anything between a tap and `getUserMedia`.** iOS Safari requires
  a user gesture, and the transient activation does not reliably survive a
  network round-trip. The first attempt awaited a consent promise before
  `getUserMedia`; the shape now is: an unconsented tap shows the notice and
  returns, and the notice's own Accept click issues `getUserMedia` as its first
  statement. Both shells split `startRecording` / `beginRecording` for exactly
  this reason — do not merge them back.
- **The consent POST is fire-and-forget, on purpose.** The user consented by
  clicking Accept; persisting it is bookkeeping. Awaiting it before recording
  would put a network round-trip inside the gesture. A failed POST means they
  are asked again next session, which is the harmless direction.
- **The mic is never opened before consent.** An earlier design held the stream
  open while the user read the notice; that lights the iOS recording indicator
  during a privacy notice and puts the browser's permission prompt ahead of the
  explanation. §6 promises this does not happen.
- **z-index must stay above 99999.** The notice opens from *inside* the scene
  modal, whose `.overlay` is at 99999 (as is `.quiz-overlay`). It first shipped
  at 9600, rendered behind an opaque backdrop, and hung the record button
  forever on an Accept nobody could reach — with no console error, because
  nothing threw. Both notices are now at 100000.
- **Escape ordering.** `handleGlobalEscape()` in `static/app.js` has an
  intentional priority chain; the consent overlay is first. Without that, Escape
  dismissed the notice *and* closed the scene modal underneath.
- **New shell: do not refresh the session on accept.** `refreshSession({force})`
  re-renders `SceneDetailPage`, disposing the runtime the Accept click just
  started recording on. The gate tracks acceptance locally; `/api/auth/me`
  reports it on the next navigation.
- **`consentGate` is a required option with no default** on
  `createSceneRuntimeStore`. A default would fail open or fail closed silently.
  Do not "fix" it by adding one.
- **Consent is server state** — `users.recording_consent_at` plus
  `recording_consent_version`, set by `POST /api/consent/recording` and read via
  `/api/auth/me`, which both shells already call at boot. A timestamp rather
  than a boolean because Art. 7(1) requires being able to *demonstrate* consent.
- **`RECORDING_CONSENT_VERSION` must equal the policy's effective date.**
  Bumping it does not re-prompt anyone; the gate checks only whether
  `recording_consent_at` is set.
- **Existing users are prompted on their next recording**, because the migration
  leaves their columns NULL. That is the point. Do not backfill them as
  consented — that fabricates a record of consent that was never given.

## Closed: §4 right to erasure

Self-service deletion shipped — `DELETE /api/account` plus the **Delete my
account** control in the Profile panel. The §4 open-item blockquote is gone and
§4/§7 now describe the real mechanism. Notes for whoever maintains this:

- **Erasure spans six tables and there are no foreign keys**, so nothing
  cascades. `_USER_DATA_TABLES` in `main.py` is the authoritative list. **Any
  new table holding user data must be added there** or that data survives
  deletion and the policy becomes untrue.
- **Two tables key on username, not user_id** (`user_missions`, `user_streak`).
  Deleting a user frees the username for re-registration, so those rows must go
  in the same transaction or the next person to claim the name inherits them.
  This is why §4 tells users the username becomes available again.
- **Tokens are stateless.** `require_live_user()` checks the user row still
  exists on every authed request; without it a stale token re-creates the
  username-keyed rows via `seed_user_missions()`.
- **Subscriptions are cancelled as part of deletion, and the ordering is the
  whole safety property.** `DELETE /api/account` runs in three phases:
  authenticate, cancel, erase. The cancel happens **outside and before** the
  deletion transaction, because a database transaction cannot roll back an HTTP
  call to Lemon Squeezy — deleting first would risk an erased account whose
  card keeps being charged. Every cancel failure returns 502 with nothing
  deleted. **Do not "tidy" this into one transaction, and do not move the
  cancel after the deletes.**
  - A 404 from Lemon Squeezy counts as success: nothing is left to bill, and
    refusing would trap the user forever.
  - `is_pro` set with no `ls_subscription_id` still yields a 409 — that is the
    pre-webhook case. It hands over `ls_customer_portal_url` when we have one.
  - Cancelling stops future billing but does not end the subscription: status
    becomes `cancelled` and access runs to `ends_at`. Lemon Squeezy has no
    immediate-expiry API. §4 states the current period is not refunded; that
    wording is deliberate, and the alternative — blocking deletion until the
    period ends — was rejected as trapping people in an account they asked to
    erase.
  - No backfill was written: there were no existing subscribers when this
    shipped, and the webhook covers everyone from then on.
- **Apple** requires an in-app deletion path for App Store approval
  (guideline 5.1.1(v)); the Profile-tab control is what satisfies it. Note that
  if billing ever moves to Apple IAP, we cannot cancel those subscriptions at
  all and the copy must tell users to manage it in App Store settings instead.

## Maintenance

- **New processors:** update §5 *before* the processor goes live — e.g. if
  Azure Pronunciation Assessment is added, it processes voice and needs a row,
  a §4 mention, and possibly a consent change.
- **Effective date:** §10 promises the date is updated when the policy changes.
  Bump the header date for substantive changes (what is collected, who
  processes it, retention); typography and formatting do not count.
- **Cited sources:** the OpenAI row in §5 quotes terms checked on 22 August
  2026 and links OpenAI's developer docs, because `openai.com/policies/*`
  blocks automated fetches. Re-check periodically and confirm the canonical
  policy page by hand.
- **Rendering:** the page is converted from this markdown at request time and
  cached on (mtime, size), so an edit here is live on the next request after
  deploy. Tables need the `tables` extension — keep table syntax simple.
