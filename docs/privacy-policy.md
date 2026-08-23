# MIRROR — Privacy Policy (DRAFT for review)

> **Status: DRAFT — no placeholders left, but not yet reviewed.** Before
> publishing: have it read by someone qualified (see §9 — the controller is
> US-based while the policy is framed around UK/EU GDPR), and close the erasure
> open item in §4. This reflects what the app does today (transcription via
> OpenAI; transcripts persisted). Update it whenever a new processor — e.g.
> Azure Pronunciation Assessment — is added, *before* that processor goes live.

**Effective date:** August 22, 2026
**Data controller:** John Greg (sole trader), 14610 Bull Run Road, Miami Lakes,
FL 33014, USA, contact contact@mirrorspeak.app.

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
transcripts to train our own or anyone else's models. Our transcription
provider's API terms state that data sent to their API is not used to train
their models unless the customer opts in, and we have not opted in (see §5).

## 4. How your voice is processed, and retention

When you submit a recording:

1. The audio is uploaded to our server over an encrypted connection.
2. It is written to a temporary file and sent to **OpenAI** (see §5) for
   speech-to-text transcription.
3. **The audio is deleted from our server immediately after transcription.** We
   do not retain a copy of your recording. OpenAI may retain it for up to 30
   days for abuse monitoring before deleting it (see §5).
4. The resulting **transcript text and your score are stored** and associated
   with your account, so we can show your history and progress.

**Retention periods:**

- **Voice audio:** not retained by us — deleted within the same request after
  transcription.
- **Transcripts, scores, progress, account data:** retained for as long as your
  account exists.
- **Request/IP logs:** retained for 30 days.

> **Open item — right to erasure.** The app currently has **no self-service
> account-deletion mechanism**. Under GDPR you must be able to honour deletion
> requests. Until an in-app "delete my account" feature exists, deletion
> requests to contact@mirrorspeak.app must be actioned manually against the
> database. Build the in-app path before wide release.

## 5. Processors and third parties

We share data only with the service providers needed to run MIRROR, each bound
by a data-processing agreement:

| Processor | What they receive | Purpose | Notes |
|---|---|---|---|
| **OpenAI** | Your voice recording (audio) | Speech-to-text transcription | Not used for training: "As of March 1, 2023, data sent to the OpenAI API is not used to train or improve OpenAI models (unless you explicitly opt in to share data with us)." Abuse-monitoring logs are retained "for up to 30 days, unless longer retention is required by law", then deleted. Source: [OpenAI — data controls in the OpenAI platform](https://developers.openai.com/api/docs/guides/your-data) (checked 22 August 2026). |
| **Anthropic** | The scene identifier only (no audio, no personal data) | Generating study vocabulary for a scene | Called once per scene, on a cache miss only; the request contains the scene id and nothing else, and the result is stored and reused for all users. No user personal data is sent. |
| **Lemon Squeezy** | Name, email, payment details you enter with them | Processing subscriptions | They act as controller for payment data under their own policy: [Privacy](https://www.lemonsqueezy.com/privacy) · [DPA](https://www.lemonsqueezy.com/dpa). |
| **Render** | All data above, at rest and in transit | Application and database hosting | Region: Oregon, USA (US West). |
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
processing; and withdraw consent. Contact contact@mirrorspeak.app. You may also
complain to your local data-protection authority (in the UK, the ICO).

## 8. Children

MIRROR is not directed to children under 13. We do not knowingly collect data
from them.

## 9. International transfers

Your data may be processed outside your country: we are based in the USA, our
hosting is in Oregon, USA, and OpenAI processes your audio in the US. Where
required — for example for users in the UK or EEA — these transfers rely on
Standard Contractual Clauses.

## 10. Changes

We will post changes here and update the effective date. Material changes
affecting how we process your voice will be notified in-app.

## 11. Contact

contact@mirrorspeak.app · John Greg (sole trader) · 14610 Bull Run Road, Miami
Lakes, FL 33014, USA
