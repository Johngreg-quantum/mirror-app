# Privacy policy — internal notes

Working notes for `privacy-policy.md`. **Not published** — only
`privacy-policy.md` is routed (`GET /privacy` in `main.py`); nothing else in
`docs/` is served, and `/static` is mounted from `static/`, not from here.

Keep notes-to-self in this file. The published policy should contain only
things a user needs to know — including honest disclosures of what the app
does *not* yet do, which is why the §6 open item lives there and not here.

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

## Open items tracked in the published policy

Disclosed to users on purpose; remove from the policy only when the
corresponding feature actually ships:

- **§6 — consent notice.** No first-run consent screen exists, and the
  recording screen does not link to the policy. §3 leans on Art. 6(1)(a)
  consent, so this gap is now the weakest point in the document.

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
