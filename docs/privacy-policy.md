# MIRROR — Privacy Policy (DRAFT for review)

> **Status: DRAFT.** Placeholders in `[BRACKETS]` must be filled before publishing.
> This reflects what the app does today (transcription via OpenAI; transcripts
> persisted). Update it whenever a new processor — e.g. Azure Pronunciation
> Assessment — is added, *before* that processor goes live.

**Effective date:** [DATE]
**Data controller:** [LEGAL ENTITY / SOLE TRADER NAME], [ADDRESS], contact
[privacy@yourdomain].

---

## 1. Who we are and what this covers

MIRROR ("we", "us") is a language-learning app in which you record yourself
speaking lines from movie scenes and receive a pronunciation/accuracy score.
This policy explains what personal data we collect when you use the app, why,
how long we keep it, who processes it on our behalf, and your rights.

## 2. What we collect

| Category | Data | Source |
|---|---|---|
| **Account** | Username, email address, password (stored only as a bcrypt hash — we never store or see your plaintext password) | You, at registration |
| **Voice recordings** | The audio you record when practising a scene | Your device microphone, with your permission |
| **Transcripts** | The text transcription of each recording | Generated from your audio (see §4) |
| **Practice & progress data** | Per-attempt scores, best scores, points, division/rank, daily streak, level progress, chosen avatar scene | Generated as you use the app |
| **Technical** | IP address (used transiently for rate-limiting), basic request logs | Automatically, when you use the service |

We do **not** collect payment card details directly — subscriptions are handled
by our payment processor (see §5).

## 3. Why we process it, and our lawful basis (UK/EU GDPR)

- **To provide the service** (score your speech, track progress, run
  leaderboards): lawful basis **performance of a contract** (Art. 6(1)(b)).
- **To process your voice recordings** specifically: lawful basis **your
  consent** (Art. 6(1)(a)), which you give before recording for the first time
  (see §6). You can withdraw consent at any time by stopping recording and/or
  deleting your account; withdrawal does not affect processing already carried
  out.
- **To secure the service** (rate-limiting, abuse prevention): **legitimate
  interests** (Art. 6(1)(f)).

We do **not** sell your data, and we do **not** use your recordings or
transcripts to train our own or anyone else's models. [Confirm this remains true
of every processor's contract — see §5.]

## 4. How your voice is processed, and retention

When you submit a recording:

1. The audio is uploaded to our server over an encrypted connection.
2. It is written to a temporary file and sent to **OpenAI** (see §5) for
   speech-to-text transcription.
3. **The audio is deleted from our server immediately after transcription.** We
   do not retain a copy of your recording. [Verify OpenAI's own retention — see
   §5.]
4. The resulting **transcript text and your score are stored** and associated
   with your account, so we can show your history and progress.

**Retention periods:**

- **Voice audio:** not retained by us — deleted within the same request after
  transcription.
- **Transcripts, scores, progress, account data:** retained for as long as your
  account exists. [Define a maximum inactivity period, e.g. "and deleted after
  [N] months of inactivity," once you decide one.]
- **Request/IP logs:** [define, e.g. rotated after 30 days].

> **Open item — right to erasure.** The app currently has **no self-service
> account-deletion mechanism**. Under GDPR you must be able to honour deletion
> requests. Until an in-app "delete my account" feature exists, deletion
> requests to [privacy@yourdomain] must be actioned manually against the
> database. Build the in-app path before wide release.

## 5. Processors and third parties

We share data only with the service providers needed to run MIRROR, each bound
by a data-processing agreement:

| Processor | What they receive | Purpose | Notes |
|---|---|---|---|
| **OpenAI** | Your voice recording (audio) | Speech-to-text transcription | Audio may be retained by OpenAI for a limited period for abuse monitoring, then deleted; not used for training under the API terms. **[Confirm against OpenAI's current API Data Usage / DPA and link it.]** |
| **Anthropic** | The scene identifier only (no audio, no personal data) | Generating study vocabulary for a scene | No user personal data is sent. |
| **[Payment processor — e.g. Lemon Squeezy]** | Name, email, payment details you enter with them | Processing subscriptions | They act as controller for payment data under their own policy: [link]. |
| **[Hosting — e.g. Render]** | All data above, at rest and in transit | Application and database hosting | [Region: specify.] |
| **TMDB** | No personal data | Movie artwork/metadata only | Data flows to us, not from you. |

If we add a new provider that processes your voice (for example, a
pronunciation-assessment service), we will update this policy and, where
required, ask for your consent **before** enabling it.

## 6. Consent to recording

Before your **first** recording, the app shows a consent notice explaining that
your audio will be sent to our transcription provider and that a transcript will
be stored, with a link to this policy. You must accept to record. See the app's
first-run recording screen.

## 7. Your rights

Subject to applicable law, you can request to: access your data; correct it;
delete it ("right to be forgotten"); export it; restrict or object to
processing; and withdraw consent. Contact [privacy@yourdomain]. You may also
complain to your local data-protection authority (in the UK, the ICO).

## 8. Children

MIRROR is not directed to children under [13/16 — pick per jurisdiction]. We do
not knowingly collect data from them.

## 9. International transfers

Your data may be processed outside your country (e.g. by OpenAI in the US). Where
required, transfers rely on [Standard Contractual Clauses / adequacy decisions].

## 10. Changes

We will post changes here and update the effective date. Material changes
affecting how we process your voice will be notified in-app.

## 11. Contact

[privacy@yourdomain] · [LEGAL ENTITY] · [ADDRESS]
