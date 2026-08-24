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

The first-run recording notice shipped. §6 now describes the real flow and §3 no
longer points at an open item. Notes for whoever maintains this:

- **There are two shells and therefore two gates.** `static/app.js` serves `/`
  and `/legacy`; `static/new-shell/` serves `/app/*`, `/scene/:id` and
  `/challenge/:id`. Each has its own recorder and its own `getUserMedia` call,
  so each needs its own gate:
  - legacy — `ensureRecordingConsent()` awaited at the top of `startRec()`
  - new shell — `createRecordingConsentGate()` passed into
    `createSceneRuntimeStore({ requireConsent })`, awaited in `startRecording()`
  **Any third recording entry point must be gated too.** Gating one shell and
  not the other would leave `/scene/:id` — the shareable link format — able to
  record with no notice, while §6 claims otherwise.
- **`requireConsent` is a required option with no default.** A default would
  either fail open (recording without the notice) or fail closed silently
  (recording mysteriously dead), so `createSceneRuntimeStore` throws when it is
  missing. That is deliberate; do not "fix" it by adding a default.
- **Consent is server state, not localStorage** — `users.recording_consent_at`
  plus `recording_consent_version`, set by `POST /api/consent/recording` and
  read via `/api/auth/me`, which both shells already call at boot. A timestamp
  rather than a boolean because Art. 7(1) requires being able to *demonstrate*
  consent; the version pins which text was shown.
- **`RECORDING_CONSENT_VERSION` in `main.py` must equal the policy's effective
  date.** Bumping it does **not** re-prompt anyone — the gate checks only
  whether `recording_consent_at` is set. If a future change to §4/§5 is material
  enough to need fresh consent, that re-prompt is a separate, deliberate change
  to the consent check.
- **Unknown consent state always reads as "not consented"** on both the server
  (`/api/auth/me` error path) and the clients. Showing the notice twice is
  harmless; skipping it is not.
- **Existing users are prompted on their next recording**, because the migration
  leaves their columns NULL. That is the point, not a side effect — everyone who
  recorded before this shipped did so without an informed consent step. Do not
  backfill them as consented; that would fabricate a record of consent that was
  never given.

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
- **Subscriptions block deletion with a 409.** We store `is_pro` but never the
  Lemon Squeezy subscription id, so we cannot cancel on the user's behalf. This
  is a deliberate, disclosed limitation, not a permanent refusal of erasure —
  the email route in §4/§7 is the fallback. Storing the subscription id so the
  cancel can happen server-side is the follow-up that removes the gate; when it
  lands, update the §4 paragraph that tells users to cancel first.
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
