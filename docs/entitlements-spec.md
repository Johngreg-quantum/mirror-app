# Entitlements, one-time purchases, and the first real access gate

Status: **decisions settled; Deploy 1 built, awaiting review**
Decided 2026-09-12. Deploy 2 (enforcement) not yet built.
Scope: the entitlements table, `order_created` handling, and one enforced gate,
landing together.

---

## 1. Why these three land together

An entitlements table that nothing reads is `is_pro` with more columns. The
deliverable is the *access check*; the table is only where it looks. So this
spec covers the table, the webhook that writes to it, and the first endpoint
that consults it — as one piece of work.

---

## 2. Current state, verified

Established by reading and probing `main.py` on 2026-09-09:

- **`is_pro` gates nothing.** It appears in exactly two read sites:
  `/api/auth/me` (returns it) and account deletion (`if is_pro:` — cancels the
  subscription). No endpoint restricts anything on it.
- **Scene access is not enforced at all.** The only `HTTPException(403,…)` in
  `main.py` is the *translation* gate (`Scene not yet unlocked`, at the
  translation endpoint). `/api/submit` validates that `scene_id` exists and
  nothing more. `/api/scenes` returns every scene, unauthenticated.
- **Levels are presentation.** `/api/progress` computes
  `unlocked = [s for lvl in LEVELS if lvl["level"] <= current_level …]` and the
  client draws padlocks. The API will score a Level 3 scene for a Level 1 user.
- **The webhook had never once run.** Found in the Lemon Squeezy dashboard
  on 2026-09-12, not in the code: it was pointed at
  `mirror-app-z8wr.onrender.com/api/lemonsqueezy/webhook`, while the route
  that exists is `/api/billing/webhook`. Every delivery had returned 404.
  The earlier audit's 401 probe was right that the *handler* was sound; the
  dashboard was aimed at a path that does not exist. Now corrected to the
  canonical domain, verified by resending an old delivery (200), and
  subscribed events raised 5 to 8 (`order_created`, `order_refunded`,
  `subscription_resumed` — the last was already handled in code but had never
  been subscribed to). Signing secret rotated in both places; **the Lemon
  Squeezy field caps at 40 characters**, so a longer secret is silently
  truncated and every signature then fails.
- **Consequence for grandfathering:** since the webhook never ran, no user can
  hold `ls_subscription_id`, and any `is_pro = TRUE` was set by hand. The
  backfill therefore sources pro rows as `manual` unless an id is genuinely
  present, so a later cancellation looking up by subscription id is not misled
  by a NULL. A real subscription webhook upserts over the row and corrects it.
- **Only subscription events were handled.** The webhook processes
  `subscription_created|updated|resumed|cancelled|expired`, treats
  `subscription_payment_failed` as a no-op, and ignores everything else —
  including orders and refunds.

**Consequence for this work:** the "one real gate" is *net-new enforcement*, not
a redirect of an existing check. That is a larger and riskier change than
swapping what a gate consults, and it is the reason for the two-deploy
sequencing in §8.

---

## 3. Schema

```sql
CREATE TABLE entitlements (
  id           <SERIAL | INTEGER PRIMARY KEY AUTOINCREMENT>,
  user_id      INTEGER NOT NULL,
  kind         TEXT    NOT NULL,        -- 'pro' | 'level'
  ref          TEXT    NOT NULL,        -- '' for pro; '1','2','3' for level
  granted_at   TEXT    NOT NULL,        -- ISO8601 UTC
  expires_at   TEXT,                    -- NULL = permanent
  source_type  TEXT    NOT NULL,        -- 'subscription' | 'order' | 'manual'
  source_id    TEXT,                    -- LS subscription/order id; NULL if manual
  revoked_at   TEXT,                    -- set on refund; row is kept
  revoke_reason TEXT,                   -- 'refund' | 'chargeback' | 'manual'
  UNIQUE (user_id, kind, ref)
);
CREATE INDEX idx_entitlements_user   ON entitlements (user_id);
CREATE INDEX idx_entitlements_source ON entitlements (source_type, source_id);
```

Must be added through `_USERS_MIGRATION_COLUMNS`' sibling pattern — a
non-destructive `CREATE TABLE IF NOT EXISTS` run for both backends, using `PH`
for placeholders and honouring `USE_PG`, so SQLite and PostgreSQL cannot drift.

### Decisions already made

| field | decision |
|---|---|
| `expires_at` | nullable. `NULL` = permanent (one-time level purchases). A timestamp = pro's paid-through date. |
| `ref` | `''` for pro, never `NULL` — `NULL` behaves differently in unique indexes across SQLite and PostgreSQL, and this project runs both. |
| `UNIQUE(user_id, kind, ref)` | gives idempotency for free: webhook retries become upserts. |
| `source_type` + `source_id` | subscription ids and order ids are different namespaces and can collide numerically. Indexed together, because refund lookup is by source. |
| cancellation | **updates** `expires_at`; never deletes. |

### Decided: keep `revoked_at` and `revoke_reason`

Confirmed 2026-09-12 — "a refund and a lapse are different facts and the whole
point of the table is that it remembers which happened."

**These were not in the original shape.** They exist because cancellation and refund are different events:

- *Cancellation* — "paid through the 30th, access continues until then."
  Expressed as `expires_at = period_end`.
- *Refund* — "this purchase never counted, revoke now."

Both could be expressed with `expires_at = now()`, which is two fewer columns —
but then a refund is indistinguishable from a lapse, and the table stops being
the history you wanted it for.

### One limitation to accept knowingly

`UNIQUE(user_id, kind, ref)` means a user who subscribes, cancels, and
re-subscribes gets their existing row **updated** — the first subscription's
`granted_at` and `source_id` are overwritten. One row per (user, kind, ref) is
current state, not a ledger.

Full history needs an append-only event table with a derived current-state view.
That is the right shape at a few thousand paying users. At ~50 users and a first
paid product it is overhead, and the LS dashboard remains the record of every
transaction. **Decided: accepted knowingly.** Current state, not a ledger; the Lemon Squeezy
dashboard is the transaction record at this size, and moving to a ledger later is
additive (this table becomes the projection).

---

## 4. The access check

```python
def load_entitlements(user_id, cur) -> set[str]
    # {'pro'} or {'level:1', 'level:2'} — expired and revoked rows excluded

def can_access_scene(user_id, scene_id, cur) -> bool
    # 'pro' satisfies everything.
    # 'level:N' satisfies every scene in LEVELS[N].
    # The free tier satisfies whatever §5 decides.
```

Active means: `revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now)`.

**One enforcement point to start: `/api/submit`.** It is the endpoint that costs
money to serve (OpenAI transcription) and the one that produces a score, so it
is where unauthorised access actually matters. `/api/scenes` stays open — scene
*metadata* is marketing, and locking it would break the landing page.

Returns `403` with a body naming what would unlock it, so the client can offer
the right purchase rather than a generic paywall.

---

## 5. Decided: what is free

**Decision (c), taken 2026-09-12: free = 5 scenes for new users, all of Level 1
for existing users, purely by grandfathering with no special case in the
check.**

Recorded because it will be re-litigated later: Mythos had earlier been told
"one scene free". Five wins because the pricing card already promises five, it
is the better number for TikTok traffic, and it means the marketing copy does
not change. `FREE_SCENE_COUNT = 5`.

The reasoning that led there:

The pricing card promises **"5 Beginner scenes"** free. Today, reaching Level 1
unlocks all **20** Level 1 scenes, and every user starts at Level 1 — so all 20
are effectively free, and `scene_config.json` has 20 / 20 / 7 across three
levels.

Selling "Mirror Level 1" for $1 therefore contradicts the current behaviour. Two
coherent readings:

**(a) Free = 5 scenes; "Mirror Level 1" = the other 15.**
Matches the pricing card. Requires choosing *which* 5, and it takes something
away from existing users — §7 grandfathering becomes mandatory, not optional.

**(b) Free = all of Level 1; the $1 product is misnamed and should be "Level 2".**
Takes nothing away, so grandfathering is a formality. But the $1 entry offer then
sits behind a skill gate (you must finish Level 1 first), which weakens it as an
entry offer.

There is a third option worth naming: **(c) free = 5 scenes for new users, all of
Level 1 for existing users** — implemented purely by grandfathering, no special
casing in the check. It gets the entry offer without taking anything from anyone.
**This is the option taken**, and it is only available because grandfathering runs
first — see §8.

---

## 6. `order_created`

The checkout already sets `custom_data.user_id`, so orders identify their buyer
the same way subscriptions do.

- Event carries the order id at `data.id` and the purchased variant at
  `data.attributes.first_order_item.variant_id`.
- Map variant → grant through the plan registry added in `b729031`: extend
  `_configured_plans()` rows with a `grants` field, e.g.
  `("level1", LS_LEVEL1_ID, "Mirror Level 1", ("level", "1"))`. The mapping then
  lives beside the ids, and adding Level 2 stays an env var plus one row.
- Write is an upsert on `UNIQUE(user_id, kind, ref)` with
  `source_type='order'`, `source_id=<order id>`, `expires_at=NULL`.
- Unknown variant → log and return 200. Do not guess a grant.

**Also fix while here:** the handler currently swallows database errors and
returns `{"ok": True}` regardless, so Lemon Squeezy never retries a failed write.
It must return non-200 when the write fails. Without that, an order can be paid
and silently never granted — which matters far more once something is actually
gated.

---

## 7. Refund flow — write the test first

Agreed: the test lands before the implementation.

**Event name confirmed as `order_refunded`** in the Lemon Squeezy dashboard,
2026-09-12. It is one of the three events added when the URL was fixed.

Flow:

1. Verify signature (existing path).
2. Read the order id from the payload.
3. Look up by `(source_type='order', source_id=<order id>)` — the reason that
   index exists.
4. Set `revoked_at = now`, `revoke_reason = 'refund'`. **Do not delete.**
5. Return 200 only if the write succeeded.

Required properties, each its own test:

- A refunded **level** purchase revokes that level and **leaves a pro row
  untouched**. This is the flow most likely to be got wrong, because the naive
  implementation revokes by `user_id` instead of by source.
- A refund for an order with **no matching entitlement** is a no-op returning
  200, not a 500 (LS retries on 5xx and would loop).
- A **duplicate** refund event is idempotent.
- A user with both pro and a refunded level **keeps full access** via pro.
- Revocation is **immediately visible** to `can_access_scene` — no cache.

---

## 8. Grandfathering ~50 existing users

The expensive failure is locking out existing users for the window between the
gate shipping and the backfill running. So they are separated:

**Deploy 1 — table, backfill, webhook writes. No gate.**
1. Create the table.
2. Backfill, idempotent (`ON CONFLICT DO NOTHING` / `INSERT OR IGNORE`), so it is
   safe to re-run:
   - Every existing user gets `kind='level'`, `ref='1'…ref=<their current level>`,
     `source_type='manual'`, `source_id=NULL`, `expires_at=NULL`,
     `revoke_reason=NULL`.
   - Every user with `is_pro = TRUE` also gets `kind='pro'`, `ref=''`,
     `source_type='subscription'`, `source_id=ls_subscription_id`,
     `expires_at=NULL` — permanent until a cancellation event sets it. Reusing
     the stored subscription id is what lets a later cancellation find the row.
3. `order_created` and refund handling go live.
4. **Verify before proceeding**: row counts match user counts, and spot-check
   that a known Level 3 user holds three level rows.

**Deploy 2 — the gate.**
`/api/submit` starts consulting `can_access_scene`. Only after Deploy 1 is
verified in production.

`is_pro` stays as a column and keeps being written, redundantly, for one release
— so Deploy 2 can be reverted without users losing access. Remove it in a third
change, once the table is proven.

**Do not backfill from `LEVELS` alone.** A user's level comes from their scores;
compute it the same way `/api/progress` does, or the backfill will disagree with
what the app has been telling them.

---

## 9. The upgrade surface — agreed as written

Asked for explicitly: what should the in-app card present, and in what order,
once the $1 tier is real.

Today it is a single button reading **"Upgrade — $47.88/year"**. That is honest —
it was `$3.99/mo` while starting an annual checkout until `b729031` — but it is
the largest number in the catalogue, and it is the first price a signed-in user
ever sees. A $47.88 annual commitment is the wrong opening ask for someone who
has not yet paid anything.

**Recommended order, top to bottom:**

1. **What they get** — the value line, unchanged ("Unlock all 47 scenes" or
   whatever §5 settles).
2. **The entry offer, primary and visually dominant** —
   `Unlock Level 1 — $1` with `one-time, no subscription` beneath it in small
   type. The "no subscription" is doing real work: it removes the objection that
   stops people who would otherwise pay.
3. **The upsell, secondary and visually lighter** — a text link rather than a
   second filled button: `Or go Pro — everything, $3.99/mo billed yearly`. It
   must state the cadence; the annual total belongs on the checkout page, not
   here.

**After a Level 1 purchase the card must not disappear.** It becomes the next
offer — Level 2 when it exists, otherwise Pro. A card that vanishes on first
purchase ends the ladder at step one, which defeats the reason for building the
ladder.

**Also needs deciding:** a signed-in user still cannot choose monthly. The
monthly/yearly pill lives on the landing page, hidden once authenticated, so
yearly always wins by default. Either surface the choice in-app or state the
cadence explicitly wherever Pro is offered.

---

## 10. Constraints and non-goals

- **Test mode only.** The store is in test mode with identity verification In
  Review, so no real purchase can be tested. Everything is built and verified
  against the test store; going live is the env swap `b729031` made possible.
- **Variant `2109931` is `status=pending`** and still issues a working checkout
  URL — verified. It is usable for test purchases now.
- **Not in scope:** Level 2/3 products, proration, upgrade/downgrade between
  plans, gifting, team accounts, or a ledger-shaped history table.

---

## 11. Decision log

All five settled 2026-09-12.

| # | decision |
|---|---|
| 1 | §5 — free = 5 scenes for new users, all of Level 1 for existing, by grandfathering. Option (c). |
| 2 | §3 — keep `revoked_at` and `revoke_reason`. |
| 3 | §3 — accept current-state-not-ledger. |
| 4 | §7 — refund event is `order_refunded`. |
| 5 | §9 — offer ordering agreed as written. |

## 12. Build status

**Deploy 1 — built, awaiting review.** Entitlements table, `app_meta`, the
one-time backfill, `order_created`, `order_refunded`, entitlement writes on the
subscription events, and the webhook now returning non-200 so Lemon Squeezy
retries a failed write. `can_access_scene()` is defined and tested but wired to
no endpoint.

Two test scripts, following the repo's `scripts/` convention since there is no
test framework:

- `scripts/test-entitlements.py` — 23 assertions over real HMAC-signed webhook
  deliveries, asserting on rows. The refund cases were written before the refund
  implementation, per §7.
- `scripts/test-backfill.py` — 9 assertions running the real `init_db()` against
  a throwaway SQLite file, including the negative property that a user who
  registers after the backfill is **not** grandfathered.

**Deploy 2 — not built.** `/api/submit` starts calling `can_access_scene()`, only
after the backfill is verified in production.
