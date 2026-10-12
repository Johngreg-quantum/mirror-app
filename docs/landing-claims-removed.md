# Restore when the pronunciation engine ships

The landing page claimed Mirror scores pronunciation, rhythm, timing and
emotion. It does not. The scoring engine sends the recording to Whisper and
compares the returned transcript to the original line:

- `sync_score` is `difflib.SequenceMatcher` over the normalised transcript and
  the normalised expected quote — a measure of **which words you said**.
- The result screen's Word Breakdown labels each expected word `Matched`,
  `Close` or `Missed` from a per-word character-sequence similarity
  (`wordBreakdown()` in `static/app-helpers.js`: >= 0.9, >= 0.55, below).
- Nothing in the pipeline measures phonemes, stress, intonation, duration,
  pacing or affect. There is no audio-to-audio comparison with the actor's take
  at any point.

So the copy below was replaced with claims the current engine can support. Every
entry is the **exact original wording**, with where it lived. When the Azure
pronunciation engine ships — which does return per-phoneme accuracy, fluency and
prosody — these are the lines to put back, verbatim or close to it, because they
were written for exactly that feature.

Two cautions for whoever restores them:

1. **Check the numbers against what Azure actually returns.** The figures below
   (96 / 91 / 88) were invented for the demo animation. Azure's
   `PronunciationAssessment` gives `AccuracyScore`, `FluencyScore`,
   `ProsodyScore` and `CompletenessScore` — close to, but not the same as,
   "Pronunciation / Rhythm / Emotion". Emotion in particular is **not** one of
   them: nothing in Azure's assessment scores emotional match, so that label
   should not come back unless something else is shipped to earn it.
2. **The word breakdown that replaced these is real and should stay.** Restoring
   the old copy means adding sub-scores alongside it, not reverting to it.

---

## 1. "The Mirror Method" card body

`index.html`, the `.mm-card-body` paragraph in the Mirror Method section.

```html
<p class="mm-card-body">Pick a scene. Hear the line. Speak it back. Our engine scores your delivery against the original performance &mdash; word by word, beat by beat &mdash; until it sounds like you own it.</p>
```

Two separate claims here, both untrue today: "beat by beat" (rhythm) and
"against the original performance" (implies comparing your audio to the actor's
audio; it compares your transcript to the written line).

## 2. Screen Test 03 sub-scores

`index.html`, inside `#mlsBeat3` (the pinned Screen Test sequence). The numbers
animate up from 0 via `data-target`, driven by the generic count-up over
`.mls-subscores b`.

```html
    <div class="mls-subscores">
      <div>PRONUNCIATION <b data-target="96">0</b></div>
      <div>RHYTHM <b data-target="91">0</b></div>
      <div>EMOTION <b data-target="88">0</b></div>
    </div>
```

## 3. Act III — "THE AI NEVER LIES." meters

`index.html`, section `#actIV`. The sub-label and all four meter rows, including
the Sync Score row that duplicated the hero number above it.

```html
  <div class="actIV-score-sub act-anim" data-delay="200">Mirror sync score &middot; last take</div>
  <div class="actIV-meters">
    <div class="actIV-meter-row act-anim" data-delay="300">
      <span class="actIV-meter-label">Pronunciation</span>
      <div class="actIV-meter-bar"><div class="actIV-meter-fill" style="width:96%"></div></div>
      <span class="actIV-meter-val">96</span>
    </div>
    <div class="actIV-meter-row act-anim" data-delay="400">
      <span class="actIV-meter-label">Rhythm</span>
      <div class="actIV-meter-bar"><div class="actIV-meter-fill" style="width:91%"></div></div>
      <span class="actIV-meter-val">91</span>
    </div>
    <div class="actIV-meter-row act-anim" data-delay="500">
      <span class="actIV-meter-label">Emotional Match</span>
      <div class="actIV-meter-bar"><div class="actIV-meter-fill" style="width:88%"></div></div>
      <span class="actIV-meter-val">88</span>
    </div>
    <div class="actIV-meter-row act-anim" data-delay="500">
      <span class="actIV-meter-label">Sync Score</span>
      <div class="actIV-meter-bar"><div class="actIV-meter-fill" style="width:94%"></div></div>
      <span class="actIV-meter-val">94</span>
    </div>
  </div>
```

Note: the original `actIV-score-sub` used a literal `·` character rather than
`&middot;`. It is written as the entity here only so this file renders cleanly;
restore whichever matches the surrounding file.

## 4. "What you get" — Live Scoring card

`index.html`, the `.mw-card-desc` under the `Live Scoring` card title.

```html
        <p class="mw-card-desc">Word-by-word pronunciation and timing scores against the original take, the instant you finish.</p>
```

## 5. "What is Mirror?" — step ii, Act it out

`index.html`, the `.what-card-body` under the `Act it out` heading.

```html
              <p class="what-card-body">
                Speak the line in real time. Mirror listens, scores your pronunciation, rhythm, and emotion.
              </p>
```

## 6. Act I ticker

`index.html`, the `.actI-ticker-track` marquee. `PRONUNCIATION` appeared twice,
once per duplicated run of the loop.

```html
      <span class="actI-ticker-item">PRONUNCIATION</span><span class="actI-ticker-dot">•</span>
```

Both occurrences became `WORD BREAKDOWN`. The other three items in the loop
(`SYNC SCORE`, `FLUENCY RANK`, `47+ SCENES`) were left alone.

## 7. Level card descriptions

`index.html`, the `.level-card-desc` lines on the Intermediate and Advanced
cards in the Levels path.

```html
        <p class="level-card-desc">Rhythm and emotion start to matter.</p>
```

```html
        <p class="level-card-desc">Accent precision and raw delivery.</p>
```

Both described the engine weighting things it does not measure. Replaced with
descriptions of the scenes themselves, which is what actually changes between
levels: longer lines, faster delivery, full monologues.

---

# Not pronunciation-related — do not restore these

The same pass corrected four claims that were simply false and will not become
true when the pronunciation engine ships. They are recorded here so the change
is traceable, **not** so they can be put back.

## Daily Challenge card

`index.html`, the `HABIT` card. The daily became the Daily Take — three short
lines in the user's own day — and there has never been an on-time bonus.

```html
        <div class="mw-card-title">Daily Challenge</div>
        <p class="mw-card-desc">One fresh scene every day. Keep your streak alive and earn bonus XP for showing up on time.</p>
```

## "Learners in 20+ countries" — three places

We have no country data of any kind: nothing in the schema records location, and
the privacy policy states we do not derive it from the IP address. The figure was
invented. The `50+ learners` count is real and was kept.

```html
    <div class="ml-topright">LEARNERS IN<br>20+ COUNTRIES</div>
```

```html
  <p class="actVII-sub act-anim" data-delay="200">Learners in 20+ countries are already speaking cinema. The next great performance is yours.</p>
```

```html
    <div class="actVII-proof-text">50+ learners · 20+ countries</div>
```

## "Bronze to Platinum"

There is no Platinum division. `DIVISIONS` in `main.py` is Bronze, Silver, Gold,
Diamond, Director. Corrected to "Bronze to Director"; if the ladder is ever
renamed, this line has to move with it.

```html
        <p class="mw-card-desc">Climb from Bronze to Platinum as you master scenes. Compete on weekly leaderboards with other performers.</p>
```
