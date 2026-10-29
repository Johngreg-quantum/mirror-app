"""Tests for the Stripe provider: signatures, checkout, events, refunds, cancel.

    python scripts/test-stripe.py [sqlitePath]

Default: ./mirror.db

Runs in-process with BILLING_PROVIDER=stripe set before the import, for the
reason test-polar.py does: the provider is read at module load.

Webhooks are posted through the real FastAPI app with real Stripe-Signature
headers. Stripe's API is replaced by an in-memory fake (FakeStripe below) so
the line-item, invoice and charge lookups the handler makes are exercised
without a network or a key -- and so a test can make one of them fail.

The cases that drove the design:

  1. A forged, stale or wrong-secret signature is REJECTED, and a missing
     secret fails closed.
  2. cancel_at_period_end does NOT remove access -- it dates the row. Only
     customer.subscription.deleted ends it.
  3. A refund or dispute revokes BY ORIGIN: refunding the $1 pack leaves the
     same user's Pro alone, and refunding a Pro charge finds the subscription
     through the invoice, in both the pre- and post-basil API shapes.
  4. The same event twice is processed once, and a handler that fails does
     not burn the event id.
  5. Account deletion cancels at Stripe first, and only a real "gone" counts
     as success.
"""
import asyncio
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import time
import uuid

RUN = "run_" + uuid.uuid4().hex[:8]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB = sys.argv[1] if len(sys.argv) > 1 else "mirror.db"

SECRET = "whsec_test_" + "s3cr3t" * 6
os.environ["BILLING_PROVIDER"] = "stripe"
os.environ["STRIPE_SECRET_KEY"] = "sk_test_fake_for_suite"
os.environ["STRIPE_WEBHOOK_SECRET"] = SECRET
os.environ["STRIPE_MONTHLY_PRICE_ID"] = "price_monthly_test"
os.environ["STRIPE_YEARLY_PRICE_ID"] = "price_yearly_test"
os.environ["STRIPE_STARTER_PRICE_ID"] = "price_starter_test"
os.environ["STRIPE_PORTAL_CONFIGURATION_ID"] = ""
os.environ["ENFORCE_ENTITLEMENTS"] = "true"
os.environ["BILLING_OPEN"] = "true"

import main as app   # noqa: E402

MONTHLY, YEARLY, STARTER = "price_monthly_test", "price_yearly_test", "price_starter_test"

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))


# ── A fake Stripe API ────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, code, body):
        self.status_code = code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeStripe:
    """Routes (METHOD, path) to a handler. Unrouted calls are a 404 with
    Stripe's resource_missing shape, so a test that forgets a route sees a
    realistic "not found" rather than a crash."""

    def __init__(self):
        self.routes = {}
        self.calls = []
        self.customers = 0

    def on(self, method, path, fn):
        self.routes[(method, path)] = fn

    def handle(self, method, url, kw):
        path = url.replace(app.STRIPE_API_BASE, "")
        self.calls.append((method, path, kw.get("data") or kw.get("params")))
        fn = self.routes.get((method, path))
        if fn is None:
            return _Resp(404, {"error": {"code": "resource_missing",
                                         "message": "No such thing: %s" % path}})
        out = fn(kw)
        if isinstance(out, Exception):
            raise out
        return _Resp(*out)


FAKE = FakeStripe()


class _Client:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        return FAKE.handle("GET", url, kw)

    async def post(self, url, **kw):
        return FAKE.handle("POST", url, kw)

    async def delete(self, url, **kw):
        return FAKE.handle("DELETE", url, kw)


app.httpx.AsyncClient = lambda *a, **kw: _Client()


def new_customer(kw):
    FAKE.customers += 1
    return 200, {"id": "cus_%s_%d" % (RUN, FAKE.customers)}


FAKE.on("POST", "/v1/customers", new_customer)
FAKE.on("POST", "/v1/checkout/sessions",
        lambda kw: (200, {"id": "cs_x", "url": "https://checkout.stripe.com/c/pay/cs_x"}))


def line_items(session_id, price_id):
    FAKE.on("GET", "/v1/checkout/sessions/%s/line_items" % session_id,
            lambda kw: (200, {"data": [{"price": {"id": price_id}}]}))


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_user(suffix):
    con = sqlite3.connect(DB)
    try:
        name = "stripetest_" + suffix
        con.execute("DELETE FROM users WHERE username = ?", (name,))
        cur = con.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (name, "stripetest_%s@example.com" % suffix, "x"),
        )
        uid = cur.lastrowid
        con.commit()
        return uid, name
    finally:
        con.close()


def sign(body: bytes, timestamp=None, secret=SECRET):
    t = str(timestamp if timestamp is not None else int(time.time()))
    sig = hmac.new(secret.encode(), t.encode() + b"." + body, hashlib.sha256).hexdigest()
    return {"Stripe-Signature": "t=%s,v1=%s" % (t, sig)}


def entitlements(uid):
    con = sqlite3.connect(DB)
    try:
        return con.execute(
            "SELECT kind, ref, source_type, source_id, revoked_at, expires_at "
            "FROM entitlements WHERE user_id = ? ORDER BY kind", (uid,)
        ).fetchall()
    finally:
        con.close()


def user_row(uid):
    con = sqlite3.connect(DB)
    try:
        r = con.execute("SELECT is_pro, stripe_customer_id, stripe_subscription_id "
                        "FROM users WHERE id = ?", (uid,)).fetchone()
        return (bool(r[0]), r[1], r[2])
    finally:
        con.close()


def set_user(uid, **cols):
    con = sqlite3.connect(DB)
    try:
        for k, v in cols.items():
            con.execute("UPDATE users SET %s = ? WHERE id = ?" % k, (v, uid))
        con.commit()
    finally:
        con.close()


def held(uid):
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

    counter = {"n": 0}

    def post(event, obj, event_id=None, created=None, headers=None, timestamp=None):
        """One signed delivery. Event ids are namespaced per RUN, because the
        dedupe table is persistent by design."""
        counter["n"] += 1
        eid = "evt_%s_%s" % (RUN, event_id or ("auto_%d" % counter["n"]))
        body = json.dumps({
            "id": eid, "object": "event", "type": event,
            "created": created if created is not None else int(time.time()),
            "data": {"object": obj},
        }).encode()
        hdrs = headers if headers is not None else sign(body, timestamp)
        return client.post("/api/billing/webhook", content=body, headers=hdrs)

    def session(uid, sid, price, mode, **extra):
        line_items(sid, price)
        obj = {"id": sid, "object": "checkout.session", "mode": mode,
               "payment_status": "paid", "status": "complete",
               "client_reference_id": str(uid), "metadata": {"user_id": str(uid)},
               "customer": "cus_%s_%d" % (RUN, uid)}
        obj.update(extra)
        return obj

    def subscription(uid, sub_id, price=MONTHLY, status="active", **extra):
        obj = {"id": sub_id, "object": "subscription", "status": status,
               "customer": "cus_%s_%d" % (RUN, uid),
               "metadata": {"user_id": str(uid)},
               "cancel_at_period_end": False, "cancel_at": None,
               "items": {"data": [{"price": {"id": price},
                                   "current_period_end": 4102444800}]}}
        obj.update(extra)
        return obj

    def subscribe(uid, sub_id, price=MONTHLY, created=None):
        return post("checkout.session.completed",
                    session(uid, "cs_" + sub_id, price, "subscription",
                            subscription=sub_id, payment_intent=None),
                    created=created)

    print("provider %s\n" % app.BILLING_PROVIDER)

    # ── CONFIG ────────────────────────────────────────────────────────────────
    print("provider wiring")
    check("BILLING_PROVIDER selects stripe", app.BILLING_PROVIDER == "stripe")
    keys = {p["key"]: p["product_id"] for p in app._configured_plans()}
    check("all three plans are configured from the STRIPE_*_PRICE_ID vars",
          keys == {"monthly": MONTHLY, "yearly": YEARLY, "starter": STARTER}, str(keys))
    check("a price id maps to the right grant",
          app._grant_for_variant(STARTER) == ("pack", "starter")
          and app._grant_for_variant(MONTHLY) == ("pro", "")
          and app._grant_for_variant(YEARLY) == ("pro", ""))
    check("an unknown price grants nothing", app._grant_for_variant("price_other") is None)

    # ── 1. SIGNATURES ─────────────────────────────────────────────────────────
    print("\nwebhook signature")
    uid, _ = make_user("sig")
    obj = session(uid, "cs_sig", STARTER, "payment", payment_intent="pi_sig")

    r = post("checkout.session.completed", obj, headers={})
    check("no Stripe-Signature is rejected", r.status_code == 401, str(r.status_code))

    body = json.dumps({"id": "evt_forged", "type": "checkout.session.completed",
                       "created": int(time.time()), "data": {"object": obj}}).encode()
    good = sign(body)["Stripe-Signature"]
    forged = good[:-8] + ("0" * 8 if not good.endswith("0" * 8) else "1" * 8)
    r = client.post("/api/billing/webhook", content=body,
                    headers={"Stripe-Signature": forged})
    check("a FORGED signature is rejected", r.status_code == 401, str(r.status_code))

    r = client.post("/api/billing/webhook", content=body + b" ",
                    headers={"Stripe-Signature": good})
    check("a valid signature over a DIFFERENT body is rejected",
          r.status_code == 401, str(r.status_code))

    r = post("checkout.session.completed", obj, timestamp=int(time.time()) - 3600)
    check("a replayed hour-old timestamp is rejected", r.status_code == 401,
          str(r.status_code))

    r = client.post("/api/billing/webhook", content=body,
                    headers=sign(body, secret="whsec_some_other_endpoint"))
    check("a signature from the wrong secret is rejected", r.status_code == 401,
          str(r.status_code))

    # Standard Webhooks base64-decodes its secret; Stripe does not. Signing
    # with the decoded form must fail, or the verifier is the wrong scheme.
    check("...and none of those granted anything", not entitlements(uid),
          str(entitlements(uid)))

    was = app.STRIPE_WEBHOOK_SECRET
    app.STRIPE_WEBHOOK_SECRET = ""
    try:
        r = client.post("/api/billing/webhook", content=body, headers=sign(body))
        check("NO SECRET CONFIGURED FAILS CLOSED (503)", r.status_code == 503,
              str(r.status_code))
    finally:
        app.STRIPE_WEBHOOK_SECRET = was

    # During a secret roll Stripe sends one v1 per active secret.
    body2 = json.dumps({"id": "evt_%s_roll" % RUN, "type": "checkout.session.completed",
                        "created": int(time.time()), "data": {"object": obj}}).encode()
    t = str(int(time.time()))
    old = hmac.new(b"whsec_old", t.encode() + b"." + body2, hashlib.sha256).hexdigest()
    new = hmac.new(SECRET.encode(), t.encode() + b"." + body2, hashlib.sha256).hexdigest()
    r = client.post("/api/billing/webhook", content=body2,
                    headers={"Stripe-Signature": "t=%s,v1=%s,v1=%s" % (t, old, new)})
    check("a VALID signature is accepted (also among several v1 during a roll)",
          r.status_code == 200, str(r.status_code))
    check("...and the grant landed", "pack:starter" in held(uid), str(held(uid)))

    # ── 2. CHECKOUT ───────────────────────────────────────────────────────────
    print("\ncheckout")
    uid, name = make_user("checkout")
    user = {"id": uid, "username": name}
    starter = app._plan_by_key("starter")
    monthly = app._plan_by_key("monthly")

    form = app._stripe_checkout_form(starter, user, "cus_abc")
    check("Starter Pack opens a one-time PAYMENT checkout", form["mode"] == "payment")
    check("...for the starter price, quantity 1",
          form["line_items[0][price]"] == STARTER and form["line_items[0][quantity]"] == "1")
    check("...with client_reference_id and metadata.user_id",
          form["client_reference_id"] == str(uid)
          and form["metadata[user_id]"] == str(uid))
    check("...copied onto the payment intent too",
          form["payment_intent_data[metadata][user_id]"] == str(uid))
    check("...on the user's own customer", form["customer"] == "cus_abc")
    check("...with no payment_method_types (Stripe rejects it; the dashboard sets methods)",
          not any(k.startswith("payment_method_types") for k in form))
    check("...and tells the buyer all sales are final, before they pay",
          "final" in form["custom_text[submit][message]"].lower()
          and "non-refundable" in form["custom_text[submit][message]"].lower())

    form = app._stripe_checkout_form(monthly, user, "cus_abc")
    check("Pro opens a SUBSCRIPTION checkout", form["mode"] == "subscription")
    check("...with the user id on the subscription, so its events attribute",
          form["subscription_data[metadata][user_id]"] == str(uid)
          and "payment_intent_data[metadata][user_id]" not in form)
    check("...and no payment_method_types on Pro either",
          not any(k.startswith("payment_method_types") for k in form))

    tok = {"Authorization": "Bearer %s" % app.make_token(uid, name)}
    FAKE.calls.clear()
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "starter"})
    check("checkout returns Stripe's hosted URL",
          r.status_code == 200 and r.json().get("checkout_url", "").startswith(
              "https://checkout.stripe.com/"), "%s %s" % (r.status_code, r.text[:80]))
    made = [c for c in FAKE.calls if c[:2] == ("POST", "/v1/customers")]
    check("...after creating ONE Stripe customer for the user", len(made) == 1, str(len(made)))
    check("...with the user id and email on it",
          made and made[0][2].get("metadata[user_id]") == str(uid)
          and made[0][2].get("email") == "stripetest_checkout@example.com")
    stored = user_row(uid)[1]
    check("...and stores stripe_customer_id", bool(stored) and stored.startswith("cus_"),
          str(stored))

    FAKE.calls.clear()
    client.post("/api/billing/checkout", headers=tok, json={"plan_key": "monthly"})
    made = [c for c in FAKE.calls if c[:2] == ("POST", "/v1/customers")]
    sess = [c for c in FAKE.calls if c[:2] == ("POST", "/v1/checkout/sessions")]
    check("a second checkout REUSES the customer", not made and
          sess and sess[0][2]["customer"] == stored, str(made))
    check("the session actually POSTed carries no payment_method_types",
          sess and not any(k.startswith("payment_method_types") for k in sess[0][2]))

    # Test-mode customers do not exist under a live key.
    tries = {"n": 0}

    def stale_then_ok(kw):
        tries["n"] += 1
        if tries["n"] == 1:
            return 400, {"error": {"code": "resource_missing", "param": "customer",
                                   "message": "No such customer: '%s'" % kw["data"]["customer"]}}
        return 200, {"id": "cs_y", "url": "https://checkout.stripe.com/c/pay/cs_y"}

    FAKE.on("POST", "/v1/checkout/sessions", stale_then_ok)
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "starter"})
    check("a customer unknown to this key (test->live) is REPLACED, once",
          r.status_code == 200 and tries["n"] == 2 and user_row(uid)[1] != stored,
          "%s tries=%s" % (r.status_code, tries["n"]))
    FAKE.on("POST", "/v1/checkout/sessions",
            lambda kw: (500, {"error": {"message": "boom"}}))
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "starter"})
    check("a Stripe failure is a 502, not a crash", r.status_code == 502, str(r.status_code))
    FAKE.on("POST", "/v1/checkout/sessions",
            lambda kw: (200, {"id": "cs_x", "url": "https://checkout.stripe.com/c/pay/cs_x"}))

    # ── 3. ONE-TIME PURCHASE ──────────────────────────────────────────────────
    print("\n$1 Starter Pack")
    uid, _ = make_user("starter")
    r = post("checkout.session.completed",
             session(uid, "cs_st", STARTER, "payment", payment_intent="pi_st"))
    rows = entitlements(uid)
    check("checkout.session.completed grants pack:starter",
          r.status_code == 200 and "pack:starter" in held(uid), str(rows))
    check("...sourced to the PAYMENT INTENT, which is what a refund names",
          rows and rows[0][2] == "order" and rows[0][3] == "pi_st", str(rows))
    check("...from the line items, not from our own metadata",
          ("GET", "/v1/checkout/sessions/cs_st/line_items") in
          [(c[0], c[1]) for c in FAKE.calls])
    check("...and records the customer", user_row(uid)[1] == "cus_%s_%d" % (RUN, uid))
    check("...and does not make the buyer Pro", user_row(uid)[0] is False)

    uid, _ = make_user("unpaid")
    post("checkout.session.completed",
         session(uid, "cs_unpaid", STARTER, "payment", payment_intent="pi_unpaid",
                 payment_status="unpaid"))
    check("a completed but UNPAID session grants nothing", not entitlements(uid))
    post("checkout.session.async_payment_succeeded",
         session(uid, "cs_unpaid", STARTER, "payment", payment_intent="pi_unpaid"))
    check("...until async_payment_succeeded", "pack:starter" in held(uid), str(held(uid)))

    uid, _ = make_user("foreign")
    r = post("checkout.session.completed",
             session(uid, "cs_foreign", "price_not_ours", "payment", payment_intent="pi_f"))
    check("a price that is not ours grants nothing, and answers 200",
          r.status_code == 200 and not entitlements(uid), str(entitlements(uid)))

    # ── 4. SUBSCRIPTION LIFECYCLE ─────────────────────────────────────────────
    print("\nsubscription lifecycle")
    uid, _ = make_user("sub")
    subscribe(uid, "sub_1", MONTHLY)
    check("subscription checkout grants pro", "pro" in held(uid), str(held(uid)))
    check("...sets is_pro and stores both ids",
          user_row(uid) == (True, "cus_%s_%d" % (RUN, uid), "sub_1"), str(user_row(uid)))
    rows = [r for r in entitlements(uid) if r[0] == "pro"]
    check("...sourced to the subscription",
          rows and rows[0][2] == "subscription" and rows[0][3] == "sub_1", str(rows))

    # basil shape: no top-level current_period_end, it is on the item.
    post("customer.subscription.updated",
         subscription(uid, "sub_1", cancel_at_period_end=True))
    rows = [r for r in entitlements(uid) if r[0] == "pro"]
    check("cancel_at_period_end does NOT remove access", "pro" in held(uid), str(held(uid)))
    check("...it sets expires_at to the period end (read off the item)",
          rows and rows[0][5] == "2100-01-01T00:00:00Z", str(rows))

    post("customer.subscription.updated",
         subscription(uid, "sub_1", cancel_at=4070908800,
                      cancel_at_period_end=False))
    rows = [r for r in entitlements(uid) if r[0] == "pro"]
    check("a dated cancel_at is honoured as the end too",
          rows and rows[0][5] == "2099-01-01T00:00:00Z", str(rows))

    post("customer.subscription.updated", subscription(uid, "sub_1"))
    rows = [r for r in entitlements(uid) if r[0] == "pro"]
    check("withdrawing the cancellation clears the end date",
          rows and rows[0][5] is None and "pro" in held(uid), str(rows))

    post("customer.subscription.updated", subscription(uid, "sub_1", status="past_due"))
    check("past_due does NOT cut access off on one failed charge", "pro" in held(uid))

    r = post("invoice.payment_failed",
             {"id": "in_fail", "customer": "cus_%s_%d" % (RUN, uid),
              "parent": {"subscription_details": {"subscription": "sub_1"}},
              "attempt_count": 1})
    check("invoice.payment_failed is logged only", r.status_code == 200 and
          "pro" in held(uid) and user_row(uid)[0] is True)

    t_deleted = int(time.time())
    post("customer.subscription.deleted", subscription(uid, "sub_1", status="canceled"),
         created=t_deleted)
    rows = [r for r in entitlements(uid) if r[0] == "pro"]
    check("customer.subscription.deleted is what ends access", "pro" not in held(uid))
    check("...by REVOKING the row", rows and rows[0][4] is not None, str(rows))
    check("...and clears is_pro and the subscription id",
          user_row(uid)[0] is False and user_row(uid)[2] is None, str(user_row(uid)))

    post("customer.subscription.updated", subscription(uid, "sub_1"),
         created=t_deleted - 600)
    check("A LATE subscription.updated DOES NOT RESURRECT a deleted subscription",
          "pro" not in held(uid) and user_row(uid)[0] is False, str(held(uid)))

    subscribe(uid, "sub_2", YEARLY, created=t_deleted + 60)
    check("...but a genuine resubscription does grant again",
          "pro" in held(uid) and user_row(uid)[2] == "sub_2", str(user_row(uid)))
    post("customer.subscription.deleted", subscription(uid, "sub_1", status="canceled"))
    check("a late deletion of the OLD subscription leaves the new one alone",
          "pro" in held(uid) and user_row(uid) [0] is True and user_row(uid)[2] == "sub_2",
          str(user_row(uid)))

    uid, _ = make_user("unpaidsub")
    subscribe(uid, "sub_u")
    post("customer.subscription.updated", subscription(uid, "sub_u", status="unpaid"))
    check("a subscription gone UNPAID loses access", "pro" not in held(uid)
          and user_row(uid)[0] is False, str(held(uid)))

    # Attribution by customer when metadata is absent.
    uid, _ = make_user("bycustomer")
    set_user(uid, stripe_customer_id="cus_only_%s" % RUN)
    sub = subscription(uid, "sub_c", metadata={}, customer="cus_only_%s" % RUN)
    post("customer.subscription.updated", sub)
    check("an event with no metadata is attributed by stripe_customer_id",
          "pro" in held(uid), str(held(uid)))

    # ── 5. REFUNDS AND DISPUTES, BY ORIGIN ────────────────────────────────────
    print("\nrefunds and disputes revoke by origin")
    no_invoices = lambda kw: (200, {"data": []})  # noqa: E731
    FAKE.on("GET", "/v1/invoice_payments", no_invoices)

    uid, _ = make_user("both")
    subscribe(uid, "sub_b")
    post("checkout.session.completed",
         session(uid, "cs_b", STARTER, "payment", payment_intent="pi_b"))
    check("the user holds both pro and pack:starter",
          {"pro", "pack:starter"} <= held(uid), str(held(uid)))

    post("charge.refunded", {"id": "ch_b", "payment_intent": "pi_b", "amount": 100,
                             "amount_refunded": 100, "refunded": True,
                             "customer": "cus_%s_%d" % (RUN, uid)})
    check("refunding the $1 PACK takes only the pack",
          "pack:starter" not in held(uid) and "pro" in held(uid), str(held(uid)))
    check("...and leaves is_pro alone", user_row(uid)[0] is True)

    # A Pro charge, post-basil: charge -> invoice_payments -> invoice.parent.
    FAKE.on("GET", "/v1/invoice_payments", lambda kw: (200, {"data": [
        {"invoice": "in_b"}]} if kw["params"].get("payment[payment_intent]") == "pi_sub_b"
        else {"data": []}))
    FAKE.on("GET", "/v1/invoices/in_b", lambda kw: (200, {
        "id": "in_b", "parent": {"subscription_details": {"subscription": "sub_b"}}}))
    post("charge.refunded", {"id": "ch_sb", "payment_intent": "pi_sub_b", "amount": 899,
                             "amount_refunded": 899, "refunded": True})
    check("REFUNDING A PRO CHARGE REMOVES PRO (basil: via invoice payments)",
          "pro" not in held(uid), str(held(uid)))
    check("...and clears is_pro", user_row(uid)[0] is False)
    FAKE.on("GET", "/v1/invoice_payments", no_invoices)

    # Pre-basil: the charge names its invoice, the invoice its subscription.
    uid, _ = make_user("oldapi")
    subscribe(uid, "sub_o")
    FAKE.on("GET", "/v1/invoices/in_o", lambda kw: (200, {"id": "in_o",
                                                          "subscription": "sub_o"}))
    post("charge.refunded", {"id": "ch_o", "payment_intent": "pi_o", "invoice": "in_o",
                             "amount": 899, "amount_refunded": 899, "refunded": True})
    check("...and via charge.invoice on an older API version",
          "pro" not in held(uid) and user_row(uid)[0] is False, str(held(uid)))

    uid, _ = make_user("partial")
    subscribe(uid, "sub_p", YEARLY)
    FAKE.on("GET", "/v1/invoices/in_p", lambda kw: (200, {"id": "in_p",
                                                          "subscription": "sub_p"}))
    post("charge.refunded", {"id": "ch_p", "payment_intent": "pi_p", "invoice": "in_p",
                             "amount": 4788, "amount_refunded": 500, "refunded": False})
    check("A PARTIAL REFUND KEEPS ACCESS", "pro" in held(uid), str(held(uid)))
    post("charge.refunded", {"id": "ch_p", "payment_intent": "pi_p", "invoice": "in_p",
                             "amount": 4788, "amount_refunded": 4788, "refunded": True})
    check("...the full refund then removes it", "pro" not in held(uid), str(held(uid)))

    uid, _ = make_user("dispute")
    post("checkout.session.completed",
         session(uid, "cs_d", STARTER, "payment", payment_intent="pi_d"))
    FAKE.on("GET", "/v1/charges/ch_d", lambda kw: (200, {"id": "ch_d",
                                                         "payment_intent": "pi_d"}))
    post("charge.dispute.created", {"id": "dp_1", "charge": "ch_d", "payment_intent": "pi_d",
                                    "amount": 100, "reason": "fraudulent"})
    rows = entitlements(uid)
    check("charge.dispute.created revokes the disputed purchase",
          "pack:starter" not in held(uid), str(held(uid)))
    con = sqlite3.connect(DB)
    reason = con.execute("SELECT revoke_reason FROM entitlements WHERE user_id = ?",
                         (uid,)).fetchone()[0]
    con.close()
    check("...recorded as a dispute, not a refund", reason == "dispute", str(reason))

    uid, _ = make_user("subdispute")
    subscribe(uid, "sub_sd")
    FAKE.on("GET", "/v1/charges/ch_sd", lambda kw: (200, {"id": "ch_sd",
                                                          "payment_intent": "pi_sd",
                                                          "invoice": "in_sd"}))
    FAKE.on("GET", "/v1/invoices/in_sd", lambda kw: (200, {"id": "in_sd",
                                                           "subscription": "sub_sd"}))
    post("charge.dispute.created", {"id": "dp_2", "charge": "ch_sd",
                                    "payment_intent": "pi_sd"})
    check("a disputed Pro charge removes Pro",
          "pro" not in held(uid) and user_row(uid)[0] is False, str(held(uid)))

    # ── 6. DUPLICATES AND FAILURES ────────────────────────────────────────────
    print("\nduplicate deliveries and failed handlers")
    uid, _ = make_user("dupe")
    obj = session(uid, "cs_dupe", STARTER, "payment", payment_intent="pi_dupe")
    post("checkout.session.completed", obj, event_id="dupe")
    con = sqlite3.connect(DB)
    before = con.execute("SELECT granted_at FROM entitlements WHERE user_id = ?",
                         (uid,)).fetchone()[0]
    con.close()
    time.sleep(1.1)
    r = post("checkout.session.completed", obj, event_id="dupe")
    con = sqlite3.connect(DB)
    after = con.execute("SELECT granted_at FROM entitlements WHERE user_id = ?",
                        (uid,)).fetchone()[0]
    n = con.execute("SELECT COUNT(*) FROM webhook_events WHERE event_id = ?",
                    ("stripe:evt_%s_dupe" % RUN,)).fetchone()[0]
    con.close()
    check("THE SAME EVENT ID TWICE is accepted but processed once",
          r.status_code == 200 and before == after, "%s -> %s" % (before, after))
    check("...and recorded in webhook_events exactly once", n == 1, str(n))

    post("charge.refunded", {"id": "ch_dupe", "payment_intent": "pi_dupe", "amount": 100,
                             "amount_refunded": 100, "refunded": True})
    post("checkout.session.completed", obj, event_id="dupe_late",
         created=int(time.time()) - 3600)
    check("A LATE completed event does not resurrect a refunded pack",
          "pack:starter" not in held(uid), str(held(uid)))

    uid, _ = make_user("retry")
    boom = {"n": 0}
    real_guarded = app.grant_entitlement_guarded

    def flaky(*a, **kw):
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("simulated handler failure")
        return real_guarded(*a, **kw)

    app.grant_entitlement_guarded = flaky
    try:
        obj = session(uid, "cs_retry", STARTER, "payment", payment_intent="pi_retry")
        r1 = post("checkout.session.completed", obj, event_id="retry")
        con = sqlite3.connect(DB)
        left = con.execute("SELECT COUNT(*) FROM webhook_events WHERE event_id = ?",
                           ("stripe:evt_%s_retry" % RUN,)).fetchone()[0]
        con.close()
        check("a failing handler returns 500 so Stripe retries", r1.status_code == 500,
              str(r1.status_code))
        check("...and does NOT leave the event id claimed", left == 0, str(left))
        r2 = post("checkout.session.completed", obj, event_id="retry")
        check("THE RETRY IS PROCESSED, and the buyer gets access",
              r2.status_code == 200 and "pack:starter" in held(uid), str(held(uid)))
    finally:
        app.grant_entitlement_guarded = real_guarded

    uid, _ = make_user("lookupfail")
    obj = session(uid, "cs_lf", STARTER, "payment", payment_intent="pi_lf")
    FAKE.on("GET", "/v1/checkout/sessions/cs_lf/line_items",
            lambda kw: (500, {"error": {"message": "stripe is down"}}))
    r = post("checkout.session.completed", obj, event_id="lookupfail")
    con = sqlite3.connect(DB)
    left = con.execute("SELECT COUNT(*) FROM webhook_events WHERE event_id = ?",
                       ("stripe:evt_%s_lookupfail" % RUN,)).fetchone()[0]
    con.close()
    check("a FAILED line-item lookup is a 500 (retry), not a dropped purchase",
          r.status_code == 500 and left == 0, "%s claimed=%s" % (r.status_code, left))
    line_items("cs_lf", STARTER)
    r = post("checkout.session.completed", obj, event_id="lookupfail")
    check("...and the retry grants", r.status_code == 200 and "pack:starter" in held(uid))

    # ── 7. ACCOUNT DELETION ───────────────────────────────────────────────────
    print("\naccount deletion")
    uid, name = make_user("delete")
    set_user(uid, password_hash=app.hash_pw("pw123456"), is_pro=0,
             stripe_subscription_id="sub_live", stripe_customer_id="cus_live",
             polar_subscription_id="polar_should_not_be_used")
    cancelled = {}

    async def fake_cancel(subscription_id):
        cancelled["id"] = subscription_id

    real_cancel = app._cancel_subscription
    app._cancel_subscription = fake_cancel
    try:
        r = client.request("DELETE", "/api/account",
                           headers={"Authorization": "Bearer %s" % app.make_token(uid, name)},
                           json={"password": "pw123456", "confirm_username": name})
    finally:
        app._cancel_subscription = real_cancel
    check("deletion cancels the STRIPE subscription (not another provider's id)",
          cancelled.get("id") == "sub_live", str(cancelled))
    check("...and then erases the account", r.status_code == 200, str(r.status_code))

    uid, name = make_user("deletefail")
    set_user(uid, password_hash=app.hash_pw("pw123456"), stripe_subscription_id="sub_x")
    FAKE.on("DELETE", "/v1/subscriptions/sub_x",
            lambda kw: (500, {"error": {"message": "down"}}))
    r = client.request("DELETE", "/api/account",
                       headers={"Authorization": "Bearer %s" % app.make_token(uid, name)},
                       json={"password": "pw123456", "confirm_username": name})
    check("a FAILED cancel refuses the deletion (502) and keeps the account",
          r.status_code == 502 and user_row(uid)[2] == "sub_x", str(r.status_code))

    def cancel_result(delete_resp, get_resp=None):
        FAKE.on("DELETE", "/v1/subscriptions/sub_c", lambda kw: delete_resp)
        if get_resp:
            FAKE.on("GET", "/v1/subscriptions/sub_c", lambda kw: get_resp)
        else:
            FAKE.routes.pop(("GET", "/v1/subscriptions/sub_c"), None)
        try:
            asyncio.run(app._cancel_stripe_subscription("sub_c"))
            return None
        except app.SubscriptionCancelError as exc:
            return str(exc)

    check("cancel: 200 is success",
          cancel_result((200, {"id": "sub_c", "status": "canceled"})) is None)
    check("cancel: 404 resource_missing is success",
          cancel_result((404, {"error": {"code": "resource_missing"}})) is None)
    check("cancel: a 404 WITHOUT Stripe's error code aborts",
          cancel_result((404, {"detail": "Not Found"})) is not None)
    check("cancel: 401 (rotated key) aborts",
          cancel_result((401, {"error": {"type": "invalid_request_error"}})) is not None)
    check("cancel: 403 (restricted key) aborts",
          cancel_result((403, {"error": {"code": "secret_key_required"}})) is not None)
    check("cancel: 400 on an ALREADY-canceled subscription is success",
          cancel_result((400, {"error": {"message": "already canceled"}}),
                        (200, {"id": "sub_c", "status": "canceled"})) is None)
    check("cancel: 400 on a subscription that is still active aborts",
          cancel_result((400, {"error": {"message": "?"}}),
                        (200, {"id": "sub_c", "status": "active"})) is not None)
    FAKE.on("DELETE", "/v1/subscriptions/sub_c", lambda kw: ConnectionError("offline"))
    try:
        asyncio.run(app._cancel_stripe_subscription("sub_c"))
        err = None
    except app.SubscriptionCancelError as exc:
        err = str(exc)
    check("cancel: unreachable Stripe aborts", err is not None, str(err))
    calls = [c for c in FAKE.calls if c[:2] == ("DELETE", "/v1/subscriptions/sub_c")]
    check("cancel is immediate: DELETE, not cancel_at_period_end", bool(calls))

    # ── 8. PORTAL ─────────────────────────────────────────────────────────────
    print("\ncustomer portal")
    uid, name = make_user("portal")
    tok = {"Authorization": "Bearer %s" % app.make_token(uid, name)}
    r = client.post("/api/billing/portal", headers=tok)
    check("no Stripe customer yet -> 404, not a broken link", r.status_code == 404,
          str(r.status_code))
    set_user(uid, stripe_customer_id="cus_portal")
    FAKE.on("POST", "/v1/billing_portal/sessions", lambda kw: (200, {
        "url": "https://billing.stripe.com/p/session/x"}
        if kw["data"]["customer"] == "cus_portal" else (400, {})))
    was_open = app.BILLING_OPEN
    app.BILLING_OPEN = False
    try:
        r = client.post("/api/billing/portal", headers=tok)
    finally:
        app.BILLING_OPEN = was_open
    check("the portal returns a fresh Stripe session URL for the user's customer",
          r.status_code == 200 and r.json().get("url", "").startswith(
              "https://billing.stripe.com/"), "%s %s" % (r.status_code, r.text[:80]))
    check("...even with BILLING_OPEN off: closing the shop never closes the way out",
          r.status_code == 200)
    r = client.post("/api/billing/portal")
    check("the portal needs a login", r.status_code in (401, 403), str(r.status_code))

    # ── 9. PLANS, PRICES, GATES ───────────────────────────────────────────────
    print("\nplans and prices")
    app._STRIPE_PRICE_CACHE.update({"at": 0.0, "variants": {}})
    prices = {
        STARTER: {"id": STARTER, "type": "one_time", "unit_amount": 100, "active": True,
                  "livemode": False, "recurring": None,
                  "product": {"name": "Mirror Starter Pack"}},
        MONTHLY: {"id": MONTHLY, "type": "recurring", "unit_amount": 899, "active": True,
                  "livemode": False, "recurring": {"interval": "month", "interval_count": 1},
                  "product": {"name": "Mirror Pro"}},
        YEARLY:  {"id": YEARLY, "type": "recurring", "unit_amount": 4788, "active": True,
                  "livemode": False, "recurring": {"interval": "year", "interval_count": 1},
                  "product": {"name": "Mirror Pro"}},
    }
    for pid, body in prices.items():
        FAKE.on("GET", "/v1/prices/%s" % pid, lambda kw, b=body: (200, b))
    body = client.get("/api/billing/plans").json()
    by = {p["key"]: p for p in body["plans"]}
    check("prices are read from the Stripe API", body["prices_live"] is True)
    check("...$1.00 one-time starter",
          by["starter"]["price_cents"] == 100 and by["starter"]["is_subscription"] is False)
    check("...$8.99 monthly",
          by["monthly"]["price_cents"] == 899 and by["monthly"]["interval"] == "month"
          and by["monthly"]["is_subscription"] is True)
    check("...$47.88 yearly",
          by["yearly"]["price_cents"] == 4788 and by["yearly"]["interval"] == "year")

    uid, name = make_user("owns")
    tok = {"Authorization": "Bearer %s" % app.make_token(uid, name)}
    subscribe(uid, "sub_own")
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "monthly"})
    check("a Pro subscriber is refused a second subscription", r.status_code == 409)
    r = client.post("/api/billing/checkout", headers=tok, json={"plan_key": "starter"})
    check("...and the pack Pro already includes", r.status_code == 409)

    app.BILLING_OPEN = False
    try:
        uid, name = make_user("closed")
        r = client.post("/api/billing/checkout",
                        headers={"Authorization": "Bearer %s" % app.make_token(uid, name)},
                        json={"plan_key": "starter"})
        check("BILLING_OPEN off still refuses checkout (503)", r.status_code == 503)
    finally:
        app.BILLING_OPEN = was_open

    uid, _ = make_user("gate")
    post("checkout.session.completed",
         session(uid, "cs_g", STARTER, "payment", payment_intent="pi_g"))
    con = sqlite3.connect(DB)
    cur = con.cursor()
    s = app._STARTER_SCENE_IDS[0]
    check("a stripe-granted pack:starter opens a starter scene",
          app.can_access_scene(cur, uid, s) is True, s)
    con.close()

    # ── 10. THE OTHER TWO PROVIDERS ARE UNTOUCHED ─────────────────────────────
    print("\nthe other providers")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "main.py"), encoding="utf-8").read()
    check("BILLING_PROVIDER still defaults to lemonsqueezy",
          'os.getenv("BILLING_PROVIDER", "lemonsqueezy")' in src)
    check("Polar and Lemon Squeezy handlers are both still routed",
          "return await _polar_webhook(request)" in src
          and "return await _lemonsqueezy_webhook(request)" in src)
    check("every webhook grant still goes through the guarded path",
          src.count("grant_entitlement(cur, user_id") == 2,
          "%d occurrences" % src.count("grant_entitlement(cur, user_id"))

    # Clean up
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM entitlements WHERE user_id IN "
                "(SELECT id FROM users WHERE username LIKE 'stripetest_%')")
    con.execute("DELETE FROM user_missions WHERE username LIKE 'stripetest_%'")
    con.execute("DELETE FROM user_streak WHERE username LIKE 'stripetest_%'")
    con.execute("DELETE FROM users WHERE username LIKE 'stripetest_%'")
    con.execute("DELETE FROM webhook_events WHERE event_id LIKE ?", ("stripe:evt_run_%",))
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
