# Privacy policy — internal notes

Working notes for `privacy-policy.md`. **Not published** — only
`privacy-policy.md` is routed (`GET /privacy` in `main.py`); nothing else in
`docs/` is served, and `/static` is mounted from `static/`, not from here.

Keep notes-to-self in this file. The published policy should contain only
things a user needs to know — including honest disclosures of what the app
does *not* yet do, which is why the §4 and §6 open items live there and not
here.

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

Both are disclosed to users on purpose and should be removed from the policy
only when the corresponding feature actually ships:

- **§4 — right to erasure.** No self-service account deletion exists. Deletion
  requests to contact@mirrorspeak.app must be actioned by hand against the
  database until an in-app path is built.
- **§6 — consent notice.** No first-run consent screen exists, and the
  recording screen does not link to the policy. §3 leans on Art. 6(1)(a)
  consent, so this gap is the weakest point in the document.

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
