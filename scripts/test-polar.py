"""Tests for the Polar provider: signatures, event mapping, refunds, cancel.

    python scripts/test-polar.py [sqlitePath]

Default: ./mirror.db

Runs in-process with BILLING_PROVIDER=polar set before the import, because the
provider is read at module load: without it every branch under test is the
Lemon Squeezy one and the suite would pass while proving nothing.

Webhooks are posted through the real FastAPI app with real Standard Webhooks
signatures, so the signature check, the event mapping and the SQL are exercised
together rather than as isolated helpers.

The cases that drove the design:

  1. A forged or stale signature is REJECTED. This endpoint can grant Pro to
     anyone, so verification failing open would be the worst bug in the app.
  2. subscription.canceled does NOT remove access -- it is an intention, and
     the user keeps what they paid for until the period ends.
     subscription.revoked is what ends it.
  3. A refund revokes ONLY the matching purchase. A refunded $1 level must not
     take away a subscription the same user holds.
  4. A refunded SUBSCRIPTION loses access. The pro row is sourced to the
     subscription, so revoking by order id alone matched nothing -- the bug
     this release fixes, in both providers.
"""
import base64
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import time
import uuid

# Namespaces this run's webhook delivery ids. See the note on post().
RUN = "run_" + uuid.uuid4().hex[:8]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = sys.argv[1] if len(sys.argv) > 1 else "mirror.db"

# Set BEFORE importing main: the provider, the gate and the secrets are all
# read at module scope.
SECRET_RAW = b"polar-test-secret-key-32-bytes-ok!!"
SECRET = "whsec_" + base64.b64encode(SECRET_RAW).decode()
os.environ["BILLING_PROVIDER"] = "polar"
os.environ["POLAR_WEBHOOK_SECRET"] = SECRET
os.environ["POLAR_SERVER"] = "sandbox"
os.environ["POLAR_MONTHLY_PRODUCT_ID"] = "39ce5af3-a866-49fa-beb6-6bcafdcf94b6"
os.environ["POLAR_YEARLY_PRODUCT_ID"] = "4b015299-dacc-441a-8790-a1ebb417914e"
os.environ["POLAR_LEVEL1_PRODUCT_ID"] = "8970ad64-ec03-494a-b18f-7a5547f48d14"
os.environ["ENFORCE_ENTITLEMENTS"] = "true"

import main as app   # noqa: E402

MONTHLY = os.environ["POLAR_MONTHLY_PRODUCT_ID"]
YEARLY  = os.environ["POLAR_YEARLY_PRODUCT_ID"]
LEVEL1  = os.environ["POLAR_LEVEL1_PRODUCT_ID"]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))


def make_user(suffix):
    con = sqlite3.connect(DB)
    try:
        name = "polartest_" + suffix
        con.execute("DELETE FROM users WHERE username = ?", (name,))
        cur = con.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (name, "polartest_%s@example.com" % suffix, "x"),
        )
        uid = cur.lastrowid
        con.commit()
        return uid, name
    finally:
        con.close()


def sign(body: bytes, msg_id="msg_1", timestamp=None, secret=SECRET):
    """A real Standard Webhooks signature: HMAC-SHA256 over id.timestamp.body,
    base64, with the secret base64-decoded after its whsec_ prefix."""
    timestamp = str(timestamp if timestamp is not None else int(time.time()))
    raw = secret[len("whsec_"):] if secret.startswith("whsec_") else secret
    key = base64.b64decode(raw)
    signed = msg_id.encode() + b"." + timestamp.encode() + b"." + body
    sig = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    return {
        "webhook-id": msg_id,
        "webhook-timestamp": timestamp,
        "webhook-signature": "v1," + sig,
    }


def entitlements(uid):
    con = sqlite3.connect(DB)
    try:
        return con.execute(
            "SELECT kind, ref, source_type, source_id, revoked_at, expires_at "
            "FROM entitlements WHERE user_id = ? ORDER BY kind", (uid,)
        ).fetchall()
    finally:
        con.close()


def is_pro(uid):
    con = sqlite3.connect(DB)
    try:
        row = con.execute("SELECT is_pro FROM users WHERE id = ?", (uid,)).fetchone()
        return bool(row[0]) if row else False
    finally:
        con.close()


def held(uid):
    """What the gate would actually honour right now."""
    con = sqlite3.connect(DB)
    try:
        return app.load_entitlements(con.cursor(), uid)
    finally:
        con.close()


def main():
    if not os.path.isfile(DB):
        print("SQLite file not found: %s" % DB)
        sys.exit(2)

    app.DB_PATH = DB
    app.init_db()

    from fastapi.testclient import TestClient
    client = TestClient(app.app)

    # Each delivery gets its own webhook-id unless a test names one. Sharing an
    # id across posts would make every event after the first look like a retry
    # to the dedupe table -- which is the behaviour the duplicate tests below
    # assert deliberately, by passing the SAME msg_id twice on purpose.
    # Delivery ids are also namespaced per RUN. The dedupe table is persistent
    # by design, so a fixed id would be "already handled" on the second run of
    # this suite and every grant after it would silently no-op.
    counter = {"n": 0}

    def post(event, data, headers=None, msg_id=None, timestamp=None):
        counter["n"] += 1
        msg_id = "%s_%s" % (RUN, msg_id or ("auto_%d" % counter["n"]))
        body = json.dumps({"type": event, "data": data}).encode()
        hdrs = headers if headers is not None else sign(body, msg_id, timestamp)
        return client.post("/api/billing/webhook", content=body, headers=hdrs)

    print("provider %s   server %s\n" % (app.BILLING_PROVIDER, app.POLAR_SERVER))

    # ── CONFIG ────────────────────────────────────────────────────────────────
    print("provider wiring")
    check("BILLING_PROVIDER selects polar", app.BILLING_PROVIDER == "polar")
    check("sandbox picks the sandbox host",
          app.POLAR_API_BASE == "https://sandbox-api.polar.sh", app.POLAR_API_BASE)
    keys = {p["key"]: p["product_id"] for p in app._configured_plans()}
    check("all three plans are configured from env",
          keys == {"monthly": MONTHLY, "yearly": YEARLY, "level1": LEVEL1}, str(keys))
    check("a product id maps to the right grant",
          app._grant_for_variant(LEVEL1) == ("level", "1")
          and app._grant_for_variant(MONTHLY) == ("pro", ""), "")
    check("an unknown product grants nothing",
          app._grant_for_variant("00000000-0000-0000-0000-000000000000") is None)
    check("plan_key resolves, and variant_id still resolves as an alias",
          app._resolve_plan("level1", None)["product_id"] == LEVEL1
          and app._resolve_plan(None, LEVEL1)["key"] == "level1")

    # ── 1. SIGNATURES ─────────────────────────────────────────────────────────
    print("\nwebhook signature")
    uid, _ = make_user("sig")
    good = {"id": "ord_sig", "product_id": LEVEL1, "metadata": {"user_id": str(uid)}}

    r = post("order.paid", good, headers={})
    check("no signature headers is rejected", r.status_code == 401, str(r.status_code))

    body = json.dumps({"type": "order.paid", "data": good}).encode()
    bad = sign(body)
    bad["webhook-signature"] = "v1,AAAA" + bad["webhook-signature"][7:]
    r = post("order.paid", good, headers=bad)
    check("a FORGED signature is rejected", r.status_code == 401, str(r.status_code))

    r = post("order.paid", good, timestamp=int(time.time()) - 3600)
    check("a replayed hour-old timestamp is rejected", r.status_code == 401,
          str(r.status_code))

    wrong_secret = "whsec_" + base64.b64encode(b"a-different-secret-entirely-32b!").decode()
    body = json.dumps({"type": "order.paid", "data": good}).encode()
    r = post("order.paid", good, headers=sign(body, secret=wrong_secret))
    check("a signature from the wrong secret is rejected", r.status_code == 401,
          str(r.status_code))
    check("...and none of those granted anything", not entitlements(uid), str(entitlements(uid)))

    r = post("order.paid", good)
    check("a VALID signature is accepted", r.status_code == 200, str(r.status_code))
    check("...and the grant landed", "level:1" in held(uid), str(held(uid)))

    # ── 2. ONE-TIME PURCHASE ──────────────────────────────────────────────────
    print("\n$1 Level 1, bought and refunded")
    uid, _ = make_user("level1")
    post("order.paid", {"id": "ord_l1", "product_id": LEVEL1,
                        "metadata": {"user_id": str(uid)}})
    rows = entitlements(uid)
    check("level:1 granted, sourced to the ORDER",
          len(rows) == 1 and rows[0][0] == "level" and rows[0][2] == "order"
          and rows[0][3] == "ord_l1", str(rows))
    check("the gate honours it", "level:1" in held(uid))

    post("order.refunded", {"id": "ord_l1", "metadata": {"user_id": str(uid)}})
    check("REFUND removes access", "level:1" not in held(uid), str(held(uid)))
    rows = entitlements(uid)
    check("...by revoking the row rather than deleting it",
          rows and rows[0][4] is not None, str(rows))

    # ── 3. SUBSCRIPTION LIFECYCLE ─────────────────────────────────────────────
    print("\nsubscription lifecycle")
    uid, _ = make_user("sub")
    post("subscription.active", {"id": "sub_1", "customer_id": "cus_1",
                                 "metadata": {"user_id": str(uid)}})
    check("subscription.active grants pro", "pro" in held(uid), str(held(uid)))
    check("...sets is_pro", is_pro(uid) is True)
    con = sqlite3.connect(DB)
    ids = con.execute("SELECT polar_subscription_id, polar_customer_id FROM users "
                      "WHERE id = ?", (uid,)).fetchone()
    con.close()
    check("...and stores the subscription and customer ids",
          ids == ("sub_1", "cus_1"), str(ids))

    # Canceled is an INTENTION. The period is still paid for.
    future = "2099-01-01T00:00:00Z"
    post("subscription.canceled", {"id": "sub_1", "ends_at": future,
                                   "metadata": {"user_id": str(uid)}})
    check("subscription.canceled does NOT remove access", "pro" in held(uid), str(held(uid)))
    check("...it sets the paid-through date instead",
          any(r[0] == "pro" and r[5] == future for r in entitlements(uid)),
          str(entitlements(uid)))

    post("subscription.uncanceled", {"id": "sub_1", "customer_id": "cus_1",
                                     "metadata": {"user_id": str(uid)}})
    check("subscription.uncanceled restores it", "pro" in held(uid))
    check("...and clears the end date",
          all(r[5] is None for r in entitlements(uid) if r[0] == "pro"),
          str(entitlements(uid)))

    post("subscription.past_due", {"id": "sub_1", "metadata": {"user_id": str(uid)}})
    check("past_due does NOT cut access off on one failed charge",
          "pro" in held(uid), str(held(uid)))

    post("subscription.revoked", {"id": "sub_1", "metadata": {"user_id": str(uid)}})
    check("subscription.REVOKED is what ends access", "pro" not in held(uid), str(held(uid)))
    check("...and clears is_pro", is_pro(uid) is False)
    con = sqlite3.connect(DB)
    left = con.execute("SELECT polar_subscription_id FROM users WHERE id = ?",
                       (uid,)).fetchone()[0]
    con.close()
    check("...and drops the subscription id, so deletion will not call a dead one",
          left is None, str(left))

    # ── 4. A REFUND TOUCHES ONLY ITS OWN PURCHASE ─────────────────────────────
    print("\na refund revokes only the matching purchase")
    uid, _ = make_user("both")
    post("subscription.active", {"id": "sub_2", "customer_id": "cus_2",
                                 "metadata": {"user_id": str(uid)}})
    post("order.paid", {"id": "ord_both", "product_id": LEVEL1,
                        "metadata": {"user_id": str(uid)}})
    check("the user holds both pro and level:1",
          {"pro", "level:1"} <= held(uid), str(held(uid)))

    post("order.refunded", {"id": "ord_both", "metadata": {"user_id": str(uid)}})
    check("refunding the $1 LEVEL takes only the level",
          "level:1" not in held(uid) and "pro" in held(uid), str(held(uid)))
    check("...and leaves is_pro alone", is_pro(uid) is True)

    # The bug this release fixes: a subscription's pro row is sourced to the
    # SUBSCRIPTION, so a refund that only revoked by order id matched nothing.
    print("\nrefunding the SUBSCRIPTION itself")
    uid, _ = make_user("subrefund")
    post("subscription.active", {"id": "sub_3", "customer_id": "cus_3",
                                 "metadata": {"user_id": str(uid)}})
    post("order.paid", {"id": "ord_sub3", "product_id": MONTHLY,
                        "subscription_id": "sub_3",
                        "metadata": {"user_id": str(uid)}})
    rows = [r for r in entitlements(uid) if r[0] == "pro"]
    check("a renewal order keeps the pro row sourced to the subscription",
          len(rows) == 1 and rows[0][2] == "subscription" and rows[0][3] == "sub_3",
          str(rows))

    post("order.refunded", {"id": "ord_sub3", "subscription_id": "sub_3",
                            "metadata": {"user_id": str(uid)}})
    check("REFUNDING A SUBSCRIPTION REMOVES ACCESS", "pro" not in held(uid), str(held(uid)))
    check("...and clears is_pro, so /api/auth/me cannot disagree with the gate",
          is_pro(uid) is False)

    # ── 4b. PARTIAL REFUNDS DO NOT REMOVE ACCESS ──────────────────────────────
    print("\npartial refunds")
    uid, _ = make_user("partial")
    post("subscription.active", {"id": "sub_p", "customer_id": "cus_p",
                                 "metadata": {"user_id": str(uid)}})
    post("order.paid", {"id": "ord_p", "product_id": YEARLY,
                        "subscription_id": "sub_p", "total_amount": 4788,
                        "metadata": {"user_id": str(uid)}})
    check("a yearly subscriber holds pro", "pro" in held(uid))

    post("order.refunded", {"id": "ord_p", "subscription_id": "sub_p",
                            "status": "partially_refunded",
                            "refunded_amount": 500, "total_amount": 4788,
                            "metadata": {"user_id": str(uid)}}, msg_id="msg_partial")
    check("A PARTIAL REFUND KEEPS ACCESS", "pro" in held(uid), str(held(uid)))
    check("...and leaves is_pro alone", is_pro(uid) is True)

    # Amounts alone, with no status field, must reach the same conclusion.
    post("order.refunded", {"id": "ord_p", "subscription_id": "sub_p",
                            "refunded_amount": 500, "total_amount": 4788,
                            "metadata": {"user_id": str(uid)}}, msg_id="msg_partial2")
    check("...same when only the amounts say so", "pro" in held(uid), str(held(uid)))

    post("order.refunded", {"id": "ord_p", "subscription_id": "sub_p",
                            "status": "refunded",
                            "refunded_amount": 4788, "total_amount": 4788,
                            "metadata": {"user_id": str(uid)}}, msg_id="msg_full")
    check("a FULL refund does remove it", "pro" not in held(uid), str(held(uid)))

    # ── 4c. RETRIES AND LATE EVENTS CANNOT RESURRECT ACCESS ───────────────────
    print("\nduplicate and late deliveries")
    uid, _ = make_user("dupe")
    post("order.paid", {"id": "ord_d", "product_id": LEVEL1,
                        "metadata": {"user_id": str(uid)}}, msg_id="msg_dupe_1")
    check("the first delivery grants", "level:1" in held(uid))

    con = sqlite3.connect(DB)
    before = con.execute("SELECT granted_at FROM entitlements WHERE user_id = ?",
                         (uid,)).fetchone()[0]
    con.close()
    r = post("order.paid", {"id": "ord_d", "product_id": LEVEL1,
                            "metadata": {"user_id": str(uid)}}, msg_id="msg_dupe_1")
    con = sqlite3.connect(DB)
    after = con.execute("SELECT granted_at FROM entitlements WHERE user_id = ?",
                        (uid,)).fetchone()[0]
    rows = con.execute("SELECT COUNT(*) FROM webhook_events WHERE event_id = ?",
                       ("polar:%s_msg_dupe_1" % RUN,)).fetchone()[0]
    con.close()
    check("A DUPLICATE webhook-id is accepted but not re-processed",
          r.status_code == 200 and before == after, "granted_at %s -> %s" % (before, after))
    check("...and it is recorded exactly once", rows == 1, str(rows))

    # The dangerous one: refund, then a retry of the ORIGINAL purchase lands.
    post("order.refunded", {"id": "ord_d", "status": "refunded",
                            "metadata": {"user_id": str(uid)}}, msg_id="msg_dupe_ref")
    check("the refund removes access", "level:1" not in held(uid))
    post("order.paid", {"id": "ord_d", "product_id": LEVEL1,
                        "modified_at": "2020-01-01T00:00:00Z",
                        "metadata": {"user_id": str(uid)}}, msg_id="msg_dupe_late")
    check("A LATE order.paid DOES NOT RESURRECT a refunded entitlement",
          "level:1" not in held(uid), str(held(uid)))

    # Same shape for a subscription, which is the one that costs real money.
    uid, _ = make_user("lateactive")
    post("subscription.active", {"id": "sub_l", "customer_id": "cus_l",
                                 "modified_at": "2026-01-01T00:00:00Z",
                                 "metadata": {"user_id": str(uid)}}, msg_id="msg_la_1")
    post("order.paid", {"id": "ord_l", "product_id": MONTHLY,
                        "subscription_id": "sub_l", "total_amount": 899,
                        "metadata": {"user_id": str(uid)}}, msg_id="msg_la_2")
    post("order.refunded", {"id": "ord_l", "subscription_id": "sub_l",
                            "status": "refunded", "refunded_amount": 899,
                            "total_amount": 899,
                            "metadata": {"user_id": str(uid)}}, msg_id="msg_la_3")
    check("the subscription refund removes access", "pro" not in held(uid))
    check("...and clears is_pro", is_pro(uid) is False)

    post("subscription.active", {"id": "sub_l", "customer_id": "cus_l",
                                 "status": "active",
                                 "modified_at": "2026-01-01T00:00:00Z",
                                 "metadata": {"user_id": str(uid)}}, msg_id="msg_la_4")
    check("A LATE subscription.active DOES NOT RESTORE PRO",
          "pro" not in held(uid), str(held(uid)))
    check("...and does not set is_pro either", is_pro(uid) is False)

    # A genuine repurchase AFTER the refund must still work.
    post("subscription.active", {"id": "sub_l2", "customer_id": "cus_l",
                                 "status": "active",
                                 "modified_at": "2099-01-01T00:00:00Z",
                                 "metadata": {"user_id": str(uid)}}, msg_id="msg_la_5")
    check("...but a genuine later resubscribe DOES grant again",
          "pro" in held(uid), str(held(uid)))

    # A status the event name contradicts is not trusted.
    uid, _ = make_user("badstatus")
    post("subscription.active", {"id": "sub_b", "status": "canceled",
                                 "metadata": {"user_id": str(uid)}}, msg_id="msg_bad")
    check("subscription.active carrying status=canceled grants nothing",
          "pro" not in held(uid), str(held(uid)))

    # ── 4d. DELETION CANCELS A REFUNDED-BUT-LIVE SUBSCRIPTION ─────────────────
    # The hole the refund fix opened: clearing is_pro meant a deletion keyed on
    # is_pro skipped the cancel, erasing the account while the card kept being
    # charged -- and taking the subscription id with it.
    print("\naccount deletion after a refund")
    uid, name = make_user("deleterefund")
    con = sqlite3.connect(DB)
    con.execute("UPDATE users SET password_hash = ?, is_pro = 0, "
                "polar_subscription_id = 'sub_live', polar_customer_id = 'cus_live' "
                "WHERE id = ?", (app.hash_pw("pw123456"), uid))
    con.commit()
    con.close()

    cancelled = {}

    async def fake_cancel(subscription_id):
        cancelled["id"] = subscription_id

    real_cancel = app._cancel_subscription
    app._cancel_subscription = fake_cancel
    try:
        token = app.make_token(uid, name)
        r = client.request(
            "DELETE", "/api/account",
            headers={"Authorization": "Bearer %s" % token},
            json={"password": "pw123456", "confirm_username": name},
        )
    finally:
        app._cancel_subscription = real_cancel

    check("deleting a REFUNDED user still cancels the live subscription",
          cancelled.get("id") == "sub_live",
          "cancelled=%r is_pro was false" % cancelled.get("id"))
    check("...and the account is then erased", r.status_code == 200, str(r.status_code))

    # And the other direction still refuses: believed pro, no id to cancel.
    uid2, name2 = make_user("deletenoid")
    con = sqlite3.connect(DB)
    con.execute("UPDATE users SET password_hash = ?, is_pro = 1, "
                "polar_subscription_id = NULL WHERE id = ?",
                (app.hash_pw("pw123456"), uid2))
    con.commit()
    con.close()
    r2 = client.request(
        "DELETE", "/api/account",
        headers={"Authorization": "Bearer %s" % app.make_token(uid2, name2)},
        json={"password": "pw123456", "confirm_username": name2},
    )
    check("a believed-subscriber with no id is still refused, not erased",
          r2.status_code == 409, str(r2.status_code))

    # ── 4e. THE SERVER REFUSES TO SELL WHAT YOU OWN ───────────────────────────
    print("\nbuying something you already have")
    uid, name = make_user("owns")
    tok = {"Authorization": "Bearer %s" % app.make_token(uid, name)}
    post("order.paid", {"id": "ord_own", "product_id": LEVEL1,
                        "metadata": {"user_id": str(uid)}}, msg_id="msg_own")
    check("the user owns level:1", "level:1" in held(uid))
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "level1"})
    check("BUYING LEVEL 1 TWICE IS REFUSED BY THE SERVER",
          r.status_code == 409, "%s %s" % (r.status_code, r.text[:70]))

    uid, name = make_user("ownspro")
    tok = {"Authorization": "Bearer %s" % app.make_token(uid, name)}
    post("subscription.active", {"id": "sub_own", "customer_id": "cus_own",
                                 "metadata": {"user_id": str(uid)}}, msg_id="msg_ownpro")
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "level1"})
    check("a Pro subscriber is refused a level purchase", r.status_code == 409,
          "%s %s" % (r.status_code, r.text[:70]))
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "monthly"})
    check("...and is refused a second subscription", r.status_code == 409,
          str(r.status_code))

    uid, name = make_user("ownsnothing")
    tok = {"Authorization": "Bearer %s" % app.make_token(uid, name)}
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "level1"})
    check("a user who owns nothing is NOT blocked (checkout reaches Polar)",
          r.status_code in (200, 502), str(r.status_code))

    # ── 5. ATTRIBUTION ────────────────────────────────────────────────────────
    print("\nattribution")
    uid, _ = make_user("attrib")
    r = post("order.paid", {"id": "ord_nouser", "product_id": LEVEL1})
    check("an event with no user metadata is accepted but grants nothing",
          r.status_code == 200 and not entitlements(uid), str(r.status_code))
    post("order.paid", {"id": "ord_extid", "product_id": LEVEL1,
                        "external_customer_id": str(uid)})
    check("external_customer_id is used when metadata is absent",
          "level:1" in held(uid), str(held(uid)))

    uid2, _ = make_user("nested")
    post("order.paid", {"id": "ord_nested", "product_id": LEVEL1,
                        "subscription": {"metadata": {"user_id": str(uid2)}}})
    check("metadata on a nested subscription is found too",
          "level:1" in held(uid2), str(held(uid2)))

    # ── 6. THE GATE IS UNCHANGED BY ANY OF THIS ───────────────────────────────
    print("\nthe gate never learns who sold the scene")
    uid, _ = make_user("gate")
    post("order.paid", {"id": "ord_gate", "product_id": LEVEL1,
                        "metadata": {"user_id": str(uid)}})
    con = sqlite3.connect(DB)
    cur = con.cursor()
    paid_l1 = next(s for s in app.LEVELS[0]["scenes"] if s not in app._free_scene_ids())
    check("a polar-granted level:1 opens a paid Level 1 scene",
          app.can_access_scene(cur, uid, paid_l1) is True, paid_l1)
    lvl2 = app.LEVELS[1]["scenes"][0] if len(app.LEVELS) > 1 else None
    if lvl2:
        check("...and still does not open Level 2",
              app.can_access_scene(cur, uid, lvl2) is False, lvl2)
    con.close()

    # ── 7. THE DEFAULT IS STILL LEMON SQUEEZY ─────────────────────────────────
    print("\nthe default provider")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "main.py"), encoding="utf-8").read()
    check("BILLING_PROVIDER defaults to lemonsqueezy when unset",
          'os.getenv("BILLING_PROVIDER", "lemonsqueezy")' in src)
    check("the Lemon Squeezy handler is kept, not deleted",
          "_lemonsqueezy_webhook" in src and "_cancel_ls_subscription" in src)
    check("the products path has the trailing slash Polar redirects to",
          '/v1/products/' in src)
    # The Lemon Squeezy Order object has no subscription_id, so an earlier
    # version of the refund fix read a field that never exists and was a
    # permanent no-op. The limitation is documented rather than faked.
    check("the dead LS subscription_id read is gone",
          'payload["data"]["attributes"].get("subscription_id")' not in src)
    check("...and the limitation is written down where the code is",
          "KNOWN LIMITATION" in src and "has no subscription_id field" in src)
    # The plain upsert should appear exactly twice: its own definition, and the
    # single call inside grant_entitlement_guarded(). Any third occurrence is a
    # webhook branch granting without the refund guard.
    check("every webhook grant goes through the guarded path",
          src.count("grant_entitlement(cur, user_id") == 2,
          "%d occurrences" % src.count("grant_entitlement(cur, user_id"))

    # Clean up
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM entitlements WHERE user_id IN "
                "(SELECT id FROM users WHERE username LIKE 'polartest_%')")
    con.execute("DELETE FROM user_missions WHERE username LIKE 'polartest_%'")
    con.execute("DELETE FROM user_streak WHERE username LIKE 'polartest_%'")
    con.execute("DELETE FROM users WHERE username LIKE 'polartest_%'")
    con.execute("DELETE FROM webhook_events WHERE event_id LIKE ?", ("polar:run_%",))
    con.commit()
    con.close()

    failed = [n for n, ok, _ in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(failed), len(results)))
    if failed:
        print("FAILED:")
        for n in failed:
            print("  - " + n)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
